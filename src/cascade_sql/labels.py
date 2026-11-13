"""Stage `label`: correctness of each record under every label version.

A manifest item carries `gold: {version: gold_sql}`, e.g.
    {"orig": "...", "arcwise": "...", "bird2026": "..."}
Each version is scored by executing gold and chosen SQL and comparing results
(`bird` = set equality as in the official BIRD script, `strict` = multiset/ordered).

Labels that need an external evaluator (e.g. Spider test-suite accuracy, which needs
many database variants) are imported from a CSV/JSONL `qid,label` file with
`--extra name=path`.
"""
import csv
import json
from typing import Dict, Optional

from . import execute as ex

_gold_cache: Dict[tuple, ex.ExecResult] = {}


def _gold(db_path: str, sql: str, timeout_s: float) -> ex.ExecResult:
    k = (db_path, sql)
    if k not in _gold_cache:
        _gold_cache[k] = ex.execute(db_path, sql, timeout_s)
    return _gold_cache[k]


GOLD_TIMEOUT_S = 180.0  # a few BIRD gold queries run > 60 s; gold gets a generous limit


def correct(db_path: str, pred_sql: str, gold_sql: str, mode: str, timeout_s: float = 30.0) -> Optional[int]:
    g = _gold(db_path, gold_sql, max(timeout_s, GOLD_TIMEOUT_S))
    if not g.ok:
        return None  # broken gold: excluded from that label version, and counted in the report
    p = ex.execute(db_path, pred_sql, timeout_s)
    if not p.ok:
        return 0
    return int(ex.results_match(p.rows, g.rows, mode=mode, gold_ordered=g.ordered))


def label_record(rec: dict, item: dict, mode: str, timeout_s: float = 30.0,
                 all_candidates: bool = True) -> dict:
    labels, cand_labels = {}, {}
    for version, gold_sql in (item.get("gold") or {}).items():
        if not gold_sql:
            continue
        labels[version] = correct(item["db_path"], rec["chosen_sql"], gold_sql, mode, timeout_s)
        if all_candidates:
            # one label per distinct result hash keeps this cheap
            by_hash = {}
            for c in rec["candidates"]:
                h = c["result_hash"]
                if h is not None and h not in by_hash:
                    by_hash[h] = correct(item["db_path"], c["sql"], gold_sql, mode, timeout_s)
            cand_labels[version] = [by_hash.get(c["result_hash"], 0) if c["exec_ok"] else 0
                                    for c in rec["candidates"]]
    rec["labels"] = {**rec.get("labels", {}), **labels}
    if all_candidates:
        rec["candidate_labels"] = cand_labels
    return rec


def load_extra_labels(path: str) -> Dict[str, int]:
    out = {}
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    out[str(d["qid"])] = int(d["label"])
    else:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                out[str(row["qid"])] = int(row["label"])
    return out
