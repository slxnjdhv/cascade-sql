"""Stage `features`: confidence features f1..f10 (see Experiment_Design.md §2.1).

All features are computed from the cached record and the schema, so they are
almost free at inference time.
"""
import math
from collections import Counter
from typing import Dict, List, Optional

import sqlglot
from sqlglot import exp

from .prompts import schema_stats
from .sac import METHODS as LOGIT_METHODS
from .sac import score_candidate

CLAUSES = ("select", "from", "where", "group", "having", "order", "limit")

FEATURE_GROUPS = {
    "prior": ["f9_n_tables", "f9_n_columns", "f9_q_len", "f9_ev_len"],
    "consistency": ["f1_top_share", "f2_n_clusters", "f3_greedy_agrees"],
    "structure": ["f4_scf_mean", "f4_scf_min", "f10_n_joins", "f10_n_subq"],
    "probability": ["f5_logprob"],
    "execution": ["f6_exec_ok", "f7_empty", "f8_repaired"],
    # base paper (Maleki et al., AAAI 2026): FTC/SLC/SAC x Avg/Prod on the chosen query
    "logit": ["l_" + m.lower().replace("-", "_") for m in LOGIT_METHODS],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]


def clause_signature(sql: str) -> Optional[Dict[str, str]]:
    """Normalised text of each top-level clause, or None if the SQL does not parse."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None
    if tree is None:
        return None
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    if sel is None:
        return None

    def s(node):
        if node is None:
            return ""
        if isinstance(node, list):
            return " | ".join(sorted(x.sql(dialect="sqlite", normalize=True) for x in node))
        return node.sql(dialect="sqlite", normalize=True)

    from_part = s(sel.args.get("from_") or sel.args.get("from"))
    joins = sel.args.get("joins") or []
    if joins:
        from_part += " || " + s(joins)
    return {"select": s(sel.expressions), "from": from_part, "where": s(sel.args.get("where")),
            "group": s(sel.args.get("group")), "having": s(sel.args.get("having")),
            "order": s(sel.args.get("order")), "limit": s(sel.args.get("limit"))}


def structure_counts(sql: str):
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return float("nan"), float("nan")
    if tree is None:
        return float("nan"), float("nan")
    n_joins = len(list(tree.find_all(exp.Join)))
    n_sel = len(list(tree.find_all(exp.Select)))
    return float(n_joins), float(max(0, n_sel - 1))


def scf(cands: List[dict], chosen: int):
    """Sub-clause frequency (Liu et al. 2025): for each clause of the chosen SQL, the share
    of candidates whose same clause is identical. Returns (mean, min) over non-empty clauses."""
    sigs = [clause_signature(c["sql"]) for c in cands]
    ref = sigs[chosen]
    if ref is None:
        return float("nan"), float("nan")
    valid = [s for s in sigs if s is not None]
    fr = []
    for cl in CLAUSES:
        if ref[cl] == "" and cl not in ("select", "from"):
            continue
        fr.append(sum(1 for s in valid if s[cl] == ref[cl]) / len(sigs))
    return (sum(fr) / len(fr), min(fr)) if fr else (float("nan"), float("nan"))


def compute(rec: dict, db_path: str) -> Dict[str, float]:
    cands = rec["candidates"]
    ci = rec["chosen_index"]
    n = len(cands)
    nan = float("nan")
    if not cands or ci is None:
        return {f: nan for f in ALL_FEATURES}
    hashes = [c["result_hash"] for c in cands if c["exec_ok"]]
    counts = Counter(hashes)
    chosen = cands[ci]
    top = counts.get(chosen["result_hash"], 0) if chosen["exec_ok"] else 0
    greedy = next((c for c in cands if c["kind"] == "greedy"), None)
    st = schema_stats(db_path)
    sc_mean, sc_min = scf(cands, ci)
    nj, nsq = structure_counts(chosen["sql"])
    lp = chosen.get("logprob")
    logit = {"l_" + k.lower().replace("-", "_"): v for k, v in score_candidate(chosen).items()}
    return {**logit,
        "f1_top_share": top / n,
        "f2_n_clusters": len(counts) / n,
        "f3_greedy_agrees": float(greedy is not None and greedy["exec_ok"]
                                  and greedy["result_hash"] == chosen["result_hash"]),
        "f4_scf_mean": sc_mean,
        "f4_scf_min": sc_min,
        "f5_logprob": lp if lp is not None else nan,
        "f6_exec_ok": sum(c["exec_ok"] for c in cands) / n,
        "f7_empty": float(bool(chosen.get("empty"))),
        "f8_repaired": float(bool(chosen.get("repaired"))),
        "f9_n_tables": float(st["n_tables"]),
        "f9_n_columns": float(math.log1p(st["n_columns"])),
        "f9_q_len": float(len(rec["question"].split())),
        "f9_ev_len": float(len((rec.get("evidence") or "").split())),
        "f10_n_joins": nj,
        "f10_n_subq": nsq,
    }
