"""Offline simulation of answer / escalate / abstain policies over cached records.

Inputs are aligned per-question arrays:
    p   : gate confidence that the LOCAL answer is correct
    lc  : 1 if the local answer is correct
    gc  : 1 if the large model's answer is correct
    cost: per-question dicts (local/large latency and tokens, schema tokens)
Decisions: 0 = answer locally, 1 = escalate (use the large model's answer), 2 = abstain.
"""
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

ANSWER, ESCALATE, ABSTAIN = 0, 1, 2


def decide(p: np.ndarray, tau_hi: float, tau_lo: float = -np.inf) -> np.ndarray:
    d = np.full(len(p), ESCALATE)
    d[p >= tau_hi] = ANSWER
    d[p < tau_lo] = ABSTAIN
    return d


def outcome(d: np.ndarray, lc: np.ndarray, gc: np.ndarray) -> np.ndarray:
    """+1 correct shown, -1 wrong shown, 0 abstained (per question)."""
    shown_correct = np.where(d == ANSWER, lc, gc)
    o = np.where(shown_correct == 1, 1, -1)
    o[d == ABSTAIN] = 0
    return o


def summarize(d: np.ndarray, lc: np.ndarray, gc: np.ndarray, costs: Optional[dict] = None,
              lambdas: Iterable[float] = (0, 1, 5, 10)) -> Dict[str, float]:
    o = outcome(d, lc, gc)
    n = len(d)
    res = {"n": n,
           "ex": float((o == 1).mean()),
           "answer_rate": float((d == ANSWER).mean()),
           "escalation_rate": float((d == ESCALATE).mean()),
           "abstain_rate": float((d == ABSTAIN).mean()),
           "selective_acc": float((o[d != ABSTAIN] == 1).mean()) if (d != ABSTAIN).any() else float("nan"),
           "wrong_rate": float((o == -1).mean())}
    for lam in lambdas:
        res[f"rs_l{lam:g}"] = float(((o == 1).sum() - lam * (o == -1).sum()) / n)
    if costs:
        esc = d == ESCALATE
        res["pe_q"] = float(esc.mean())                                   # share of questions sent off-prem
        st = np.asarray(costs["schema_tokens"], float)
        res["pe_s"] = float(st[esc].sum() / max(1.0, st.sum()))           # share of schema tokens sent
        lat = np.asarray(costs["local_latency"], float) + esc * np.asarray(costs["large_latency"], float)
        res["lat_p50"] = float(np.percentile(lat, 50))
        res["lat_p95"] = float(np.percentile(lat, 95))
        res["api_in_tokens"] = float(np.asarray(costs["large_prompt_tokens"], float)[esc].sum())
        res["api_out_tokens"] = float(np.asarray(costs["large_output_tokens"], float)[esc].sum())
        if costs.get("price_in_per_m") is not None:
            res["api_usd"] = (res["api_in_tokens"] * costs["price_in_per_m"]
                              + res["api_out_tokens"] * costs["price_out_per_m"]) / 1e6
    return res


def budget_decisions(p: np.ndarray, rate: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Escalate exactly round(rate*n) questions with the lowest confidence (ties broken randomly)."""
    n = len(p)
    k = int(round(rate * n))
    rng = rng or np.random.default_rng(0)
    order = np.lexsort((rng.random(n), p))  # ascending p
    d = np.full(n, ANSWER)
    d[order[:k]] = ESCALATE
    return d


def escalation_curve(p, lc, gc, rates: Sequence[float] = tuple(np.linspace(0, 1, 21))) -> List[dict]:
    p, lc, gc = map(np.asarray, (p, lc, gc))
    out = []
    for r in rates:
        d = budget_decisions(p, r)
        out.append({"rate": float(r), "ex": float(np.where(d == ESCALATE, gc, lc).mean())})
    return out


def curve_area(curve: List[dict]) -> float:
    r = np.array([c["rate"] for c in curve])
    e = np.array([c["ex"] for c in curve])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(e, r) / (r.max() - r.min()))


def gap_recovered(ex_cascade: float, ex_local: float, ex_large: float) -> float:
    denom = ex_large - ex_local
    return float("nan") if abs(denom) < 1e-12 else (ex_cascade - ex_local) / denom


def choose_thresholds(p, lc, gc, lam: float, esc_cost: float, three_way: bool = True,
                      grid: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Maximise mean utility on the calibration split:
       U = +1 correct, -lam wrong, 0 abstain, minus esc_cost per escalation."""
    p, lc, gc = map(np.asarray, (p, lc, gc))
    grid = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(p, np.linspace(0, 1, 101))])) \
        if grid is None else np.asarray(grid)
    best = {"utility": -np.inf}
    for hi in grid:
        los = [g for g in grid if g <= hi] if three_way else [-np.inf]
        for lo in los:
            d = decide(p, hi, lo)
            o = outcome(d, lc, gc)
            u = ((o == 1).sum() - lam * (o == -1).sum() - esc_cost * (d == ESCALATE).sum()) / len(p)
            if u > best["utility"]:
                best = {"utility": float(u), "tau_hi": float(hi), "tau_lo": float(lo)}
    return best


def decide_answer_abstain(p: np.ndarray, tau: float) -> np.ndarray:
    """No large model available: answer locally if p >= tau, otherwise abstain."""
    return np.where(np.asarray(p) >= tau, ANSWER, ABSTAIN)


def choose_answer_abstain(p, lc, lam: float) -> Dict[str, float]:
    p, lc = np.asarray(p), np.asarray(lc)
    best = {"utility": -np.inf}
    for tau in np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(p, np.linspace(0, 1, 201))])):
        ans = p >= tau
        u = (lc[ans].sum() - lam * (1 - lc[ans]).sum()) / len(p)
        if u > best["utility"]:
            best = {"utility": float(u), "tau": float(tau)}
    return best


def risk_coverage(p, lc) -> List[dict]:
    """Selective accuracy of the LOCAL model when answering the top-coverage share by confidence."""
    p, lc = np.asarray(p), np.asarray(lc)
    order = np.argsort(-p)
    cum = np.cumsum(lc[order])
    n = len(p)
    return [{"coverage": (i + 1) / n, "risk": 1 - cum[i] / (i + 1)} for i in range(n)]
