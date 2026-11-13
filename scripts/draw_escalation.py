"""Paper figure: accuracy vs. escalation (left) and gate reliability with bin counts (right), 1.5B -> 3B, test split.
usage: PYTHONPATH=src python scripts/draw_escalation.py ../paper/figures/escalation_curve"""
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cascade_sql import baselines as B
from cascade_sql import metrics as M
from cascade_sql import policy as P
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES, FEATURE_GROUPS
from cascade_sql.gate import Gate

out = sys.argv[1] if len(sys.argv) > 1 else "escalation_curve"
R, SM, LG = "results", "qwen-coder-1.5b-mlx", "qwen-coder-3b-mlx"
gate = Gate.load(f"{R}/gates/{SM}.seed0/gate.pkl")
router = pickle.load(open(f"{R}/gates/{SM}.seed0/router.pkl", "rb"))
te = list(load(R, SM, "test_b", 0).values())
L = {r["qid"]: r for r in load(R, LG, "test_b", 0).values()}
X, y, texts, keep = matrix(te, ALL_FEATURES, "orig")
recs = [r for r, k in zip(te, keep) if k]
gc = np.array([L[r["qid"]]["labels"]["orig"] or 0 for r in recs])
p = gate.predict(X)
Xp, _, _, _ = matrix(te, FEATURE_GROUPS["prior"], "orig")
pr = router.predict(texts, Xp)
f1 = X[:, ALL_FEATURES.index("f1_top_share")]
rates = np.linspace(0, 1, 21)
rng = np.random.default_rng(0)
curves = {"Hybrid gate (ours)": p, "Execution consistency": f1, "Pre-generation router": pr,
          "Random": rng.random(len(y)), "Oracle": B.oracle_scores(y, gc)}
style = {"Hybrid gate (ours)": dict(color="#1f5fa8", lw=2.0, ls="-"),
         "Execution consistency": dict(color="#2e8b57", lw=1.5, ls="-."),
         "Pre-generation router": dict(color="#d9822b", lw=1.5, ls="--"),
         "Random": dict(color="#8a8a8a", lw=1.2, ls=":"),
         "Oracle": dict(color="#7b4fa0", lw=1.2, ls=(0, (1, 1)))}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig = plt.figure(figsize=(7.2, 3.0))
ax = fig.add_axes([0.07, 0.15, 0.42, 0.78])
for name, sc in curves.items():
    ex = [100 * c["ex"] for c in P.escalation_curve(sc, y, gc, rates)]
    ax.plot(100 * rates, ex, label=name, **style[name])
ax.axhline(100 * y.mean(), color="#444", lw=0.7, ls=(0, (4, 3)))
ax.axhline(100 * gc.mean(), color="#444", lw=0.7, ls=(0, (4, 3)))
ax.text(101, 100 * y.mean(), " B2 (1.5B)", va="center", fontsize=7, color="#444")
ax.text(101, 100 * gc.mean(), " B3 (3B)", va="center", fontsize=7, color="#444")
ax.set_xlabel("Questions escalated to the 3B model (%)")
ax.set_ylabel("Execution accuracy (%)")
ax.set_xlim(0, 100)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.07))

ax2 = fig.add_axes([0.66, 0.38, 0.31, 0.55])
bins = M.reliability_bins(p, y, 10)
mp = [b[0] for b in bins]
acc = [b[1] for b in bins]
cnt = [b[2] for b in bins]
ax2.plot([0, 1], [0, 1], color="#aaa", lw=0.8)
ax2.plot(mp, acc, "o-", color="#1f5fa8", ms=3.5, lw=1.4)
ax2.set_ylabel("Observed accuracy")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xticklabels([])
ax2.spines[["top", "right"]].set_visible(False)
ax3 = fig.add_axes([0.66, 0.15, 0.31, 0.18])
ax3.bar(mp, cnt, width=0.07, color="#9db7d5")
ax3.set_xlim(0, 1)
ax3.set_xlabel("Predicted P(correct)")
ax3.set_ylabel("Count", fontsize=7)
ax3.spines[["top", "right"]].set_visible(False)
fig.savefig(out + ".pdf")
fig.savefig(out + ".png", dpi=200)
print(out)
