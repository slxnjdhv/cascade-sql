"""Stage `generate`: run one model on one split and cache a JSONL record per question.

Per question:
  1. 1 greedy + N sampled candidates           (N, temperature from config)
  2. execute each on a read-only copy of the DB
  3. one repair round for candidates that errored (execution feedback)
  4. cluster executable candidates by result hash; the largest cluster wins
     (ties: the cluster containing the greedy answer, then the higher mean log-prob)
  5. chosen SQL = greedy if it is in the winning cluster, else that cluster's best-log-prob member

Runs are resumable: questions already present in the output file are skipped.
"""
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional

from . import execute as ex
from .io import append_jsonl, read_jsonl
from .llm import Backend
from .prompts import (approx_tokens, extract_sql, generation_messages, repair_messages,
                      schema_ddl, schema_with_rows)


def _cand(sql: str, comp, res: ex.ExecResult, kind: str, repaired: bool = False) -> dict:
    return {"sql": sql, "kind": kind, "repaired": repaired,
            "logprob": comp.sql_logprob_mean, "n_tokens": comp.sql_n_tokens,
            "exec_ok": res.ok, "error": res.error, "timed_out": res.timed_out,
            "empty": res.empty if res.ok else None, "truncated": res.truncated,
            "result_hash": res.result_hash(), "exec_s": round(res.seconds, 4),
            "n_rows": len(res.rows) if res.ok else None,
            "token_data": comp.token_data}


def choose(cands: List[dict]) -> Optional[int]:
    """Index of the chosen candidate, or None if nothing executed."""
    clusters: Dict[str, List[int]] = defaultdict(list)
    for i, c in enumerate(cands):
        if c["exec_ok"]:
            clusters[c["result_hash"]].append(i)
    if not clusters:
        return 0 if cands else None

    def lp(i):
        v = cands[i]["logprob"]
        return v if v is not None else -1e9

    def key(h):
        idx = clusters[h]
        has_greedy = any(cands[i]["kind"] == "greedy" for i in idx)
        mean_lp = sum(lp(i) for i in idx) / len(idx)
        return (len(idx), has_greedy, mean_lp)

    best = max(clusters, key=key)
    idx = clusters[best]
    for i in idx:
        if cands[i]["kind"] == "greedy":
            return i
    return max(idx, key=lp)


def process_question(item: dict, backend: Backend, gcfg: dict, seed: int = 0) -> dict:
    n = int(gcfg.get("n_samples", 8))
    temp = float(gcfg.get("temperature", 0.8))
    max_tokens = int(gcfg.get("max_tokens", 1024))
    timeout_s = float(gcfg.get("exec_timeout_s", 30))
    # repair: "all" (every failing candidate), "greedy" (only the greedy answer, max 1 per question;
    # laptop default) or false. `true` means "all" for backward compatibility.
    rmode = gcfg.get("repair", "all")
    rmode = "all" if rmode is True else ("off" if rmode in (False, None, "off") else rmode)
    do_repair = rmode != "off"
    include_greedy = bool(gcfg.get("greedy", True))
    style = gcfg.get("prompt_style", "omnisql")

    schema = (schema_with_rows(item["db_path"]) if style == "maleki"
              else schema_ddl(item["db_path"], int(gcfg.get("n_example_values", 3))))
    msgs = generation_messages(schema, item["question"], item.get("evidence"), style=style)
    meta = item.get("mock_meta")
    t0 = time.monotonic()
    prompt_tokens = output_tokens = 0
    calls = []
    if include_greedy:
        calls.append(("greedy", backend.generate(msgs, n=1, temperature=0.0, max_tokens=max_tokens,
                                                 seed=seed, meta=meta)))
    if n > 0:
        calls.append(("sample", backend.generate(msgs, n=n, temperature=temp, max_tokens=max_tokens,
                                                 seed=seed, meta=meta)))
    cands = []
    for kind, comps in calls:
        prompt_tokens += comps[0].prompt_tokens if comps else 0
        for c in comps:
            output_tokens += c.output_tokens
            sql = extract_sql(c.text)
            cands.append(_cand(sql, c, ex.execute(item["db_path"], sql, timeout_s), kind))

    n_repairs = 0
    if do_repair:
        for i, c in enumerate(cands):
            if c["exec_ok"] or (rmode == "greedy" and c["kind"] != "greedy"):
                continue
            rm = repair_messages(schema, item["question"], item.get("evidence"), c["sql"], c["error"] or "",
                                 style=style)
            comp = backend.generate(rm, n=1, temperature=0.0, max_tokens=max_tokens,
                                    seed=seed + 1000 + i, meta=meta)[0]
            n_repairs += 1
            prompt_tokens += comp.prompt_tokens
            output_tokens += comp.output_tokens
            sql = extract_sql(comp.text)
            pre = {k: c.get(k) for k in ("sql", "error", "logprob", "token_data")}
            cands[i] = _cand(sql, comp, ex.execute(item["db_path"], sql, timeout_s), c["kind"], repaired=True)
            cands[i]["pre_repair"] = pre  # original attempt (failed to execute => incorrect), kept for R0

    ci = choose(cands)
    return {
        "qid": item["qid"], "split": item["split"], "db_id": item["db_id"],
        "model": backend.name, "seed": seed,
        "question": item["question"], "evidence": item.get("evidence"),
        "candidates": cands, "chosen_index": ci,
        "chosen_sql": cands[ci]["sql"] if ci is not None else "",
        "n_repairs": n_repairs,
        "schema_tokens": approx_tokens(schema),
        "prompt_tokens": prompt_tokens or approx_tokens(msgs[-1]["content"]) * (1 + (n > 0)),
        "output_tokens": output_tokens,
        "latency_s": round(time.monotonic() - t0, 3),
        "gen_config": {"n_samples": n, "temperature": temp, "repair": rmode, "prompt_style": style},
    }


def run(items: List[dict], backend: Backend, gcfg: dict, out_path: str, seed: int = 0,
        limit: Optional[int] = None, log_every: int = 25) -> int:
    done = {r["qid"] for r in read_jsonl(out_path)}
    todo = [it for it in items if it["qid"] not in done]
    if limit:
        todo = todo[:limit]
    t0 = time.monotonic()
    for k, it in enumerate(todo, 1):
        try:
            rec = process_question(it, backend, gcfg, seed=seed)
        except Exception as e:  # keep going; failures are logged and retried on the next run
            print(f"[generate] {it['qid']}: {type(e).__name__}: {e}")
            continue
        append_jsonl(out_path, rec)
        if k % log_every == 0 or k == len(todo):
            rate = k / max(1e-9, time.monotonic() - t0)
            print(f"[generate] {k}/{len(todo)} done ({rate:.2f} q/s) -> {os.path.basename(out_path)}")
    ex.close_all()
    return len(todo)
