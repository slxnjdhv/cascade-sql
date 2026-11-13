"""Seed stability: escalation-curve areas (gate vs. consistency vs. router) per sampling seed, paired bootstrap.
usage: PYTHONPATH=src python scripts/seed_stability.py [--seeds 0 1] > results/analysis/seed_stability.md"""
import argparse
import pickle

import numpy as np

from cascade_sql import metrics as M
from cascade_sql import policy as P
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES, FEATURE_GROUPS
from cascade_sql.gate import Gate

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
ap.add_argument("--local", default="qwen-coder-1.5b-mlx")
ap.add_argument("--large", default="qwen-coder-3b-mlx")
a = ap.parse_args()
R, RATES = "results", tuple(np.linspace(0, 1, 21))
L = {r["qid"]: r for r in load(R, a.large, "test_b", 0).values()}


def area(s, lc, gc):
    return P.curve_area(P.escalation_curve(s, lc, gc, RATES))


print("# Seed stability (1.5B -> 3B, test_b, original labels)\n")
print("| seed | B2 acc | area gate | area consistency | area router | gate−cons [95% CI] | gate−router [95% CI] |")
print("|---|---|---|---|---|---|---|")
for s in a.seeds:
    gate = Gate.load(f"{R}/gates/{a.local}.seed{s}/gate.pkl")
    router = pickle.load(open(f"{R}/gates/{a.local}.seed{s}/router.pkl", "rb"))
    te = list(load(R, a.local, "test_b", s).values())
    X, y, texts, keep = matrix(te, ALL_FEATURES, "orig")
    recs = [r for r, k in zip(te, keep) if k]
    gc = np.array([L[r["qid"]]["labels"]["orig"] or 0 for r in recs])
    p = gate.predict(X)
    f1 = X[:, ALL_FEATURES.index("f1_top_share")]
    Xp, _, _, _ = matrix(te, FEATURE_GROUPS["prior"], "orig")
    pr = router.predict(texts, Xp)
    dc = M.bootstrap_ci(lambda i: area(p[i], y[i], gc[i]) - area(f1[i], y[i], gc[i]), len(y), n_boot=2000)
    dr = M.bootstrap_ci(lambda i: area(p[i], y[i], gc[i]) - area(pr[i], y[i], gc[i]), len(y), n_boot=2000)
    print(f"| {s} | {100*y.mean():.1f} | {area(p, y, gc):.3f} | {area(f1, y, gc):.3f} | {area(pr, y, gc):.3f} | "
          f"{dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}] | {dr[0]:+.4f} [{dr[1]:+.4f}, {dr[2]:+.4f}] |")
