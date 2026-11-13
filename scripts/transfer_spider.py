"""E5: out-of-domain transfer. The gate is trained on BIRD (g_train) and calibrated on BIRD (calib), then
applied unchanged to Spider dev (20 unseen databases, no evidence text). Reports signal quality, the
1.5B->3B escalation curve, and answer/abstain reliability with thresholds chosen on BIRD calib.

usage: PYTHONPATH=src python scripts/transfer_spider.py [--local qwen-coder-1.5b-mlx --large qwen-coder-3b-mlx]
"""
import argparse
import json
import os

import numpy as np

from cascade_sql import metrics as M
from cascade_sql import policy as P
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES, FEATURE_GROUPS
from cascade_sql.gate import Gate

ap = argparse.ArgumentParser()
ap.add_argument("--local", default="qwen-coder-1.5b-mlx")
ap.add_argument("--large", default="qwen-coder-3b-mlx")
ap.add_argument("--results", default="results")
a = ap.parse_args()

gate = Gate.load(os.path.join(a.results, "gates", f"{a.local}.seed0", "gate.pkl"))
sp = list(load(a.results, a.local, "spider_dev", 0).values())
L = {r["qid"]: r for r in load(a.results, a.large, "spider_dev", 0).values()}
X, y, _, keep = matrix(sp, ALL_FEATURES, "orig")
recs = [r for r, k in zip(sp, keep) if k]
yl = np.array([L[r["qid"]]["labels"]["orig"] or 0 for r in recs])
p = gate.predict(X)
n = len(y)
col = {f: X[:, ALL_FEATURES.index(f)] for f in ALL_FEATURES}

# in-domain refit on Spider for reference (5-fold out-of-fold), shows the cost of transfer
from sklearn.model_selection import StratifiedKFold
p_in = np.zeros(n)
for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    g = Gate(features=ALL_FEATURES).fit(X[tr], y[tr])
    p_in[va] = g.raw(X[va])

out = {"n": n, "acc_small": float(y.mean()), "acc_large": float(yl.mean()),
       "oracle": float(np.maximum(y, yl).mean()), "signals": {}}
md = [f"# E5 transfer to Spider dev: gate trained/calibrated on BIRD (n={n})\n\n",
      f"Accuracy: {a.local} self-consistency {100 * y.mean():.1f}%, {a.large} {100 * yl.mean():.1f}%, "
      f"oracle {100 * np.maximum(y, yl).mean():.1f}%\n\n",
      "| signal | AUROC | AUPRC(wrong) | ECE | Brier |\n|---|---|---|---|---|\n"]
rows = [("SAC-Prod (base paper)", np.nan_to_num(col["l_sac_prod"], nan=0.0)),
        ("SAC-Avg (base paper)", np.nan_to_num(col["l_sac_avg"], nan=0.0)),
        ("execution consistency", col["f1_top_share"]),
        ("hybrid gate, BIRD-trained (transfer)", p),
        ("hybrid gate, Spider 5-fold (in-domain reference)", p_in)]
for name, s in rows:
    q = M.signal_quality(s, y)
    out["signals"][name] = q
    f = lambda v: "-" if not np.isfinite(v) else f"{100 * v:.1f}"
    md.append(f"| {name} | {f(q['auroc'])} | {f(q['auprc_wrong'])} | {f(q['ece'])} | {f(q['brier'])} |\n")


def curve(score):
    o = np.argsort(score)
    xs = np.linspace(0, 1, 21)
    ys = []
    for r in xs:
        esc = np.zeros(n, bool)
        esc[o[: int(round(r * n))]] = True
        ys.append(np.where(esc, yl, y).mean())
    return xs, np.array(ys)


md.append("\n| escalation policy | EX@10% | EX@20% | EX@30% | area |\n|---|---|---|---|---|\n")
rng = np.random.default_rng(0)
for name, s in (("hybrid gate (transfer)", p), ("execution consistency", col["f1_top_share"]),
                ("random", rng.random(n)), ("oracle", y - yl * 0.5)):
    xs, ys = curve(s)
    area = float(np.trapezoid(ys, xs)) if hasattr(np, "trapezoid") else float(np.trapz(ys, xs))
    out.setdefault("curves", {})[name] = {"x": xs.tolist(), "y": ys.tolist(), "area": area}
    md.append(f"| {name} | {100 * ys[2]:.1f} | {100 * ys[4]:.1f} | {100 * ys[6]:.1f} | {area:.3f} |\n")

# answer/abstain with thresholds chosen on BIRD calib
ca = list(load(a.results, a.local, "calib", 0).values())
Xc, yc, _, _ = matrix(ca, ALL_FEATURES, "orig")
pc = gate.predict(Xc)
md.append("\n| answer/abstain (τ from BIRD calib) | correct | abstain | wrong | RS1 | RS5 |\n|---|---|---|---|---|---|\n")
for lam in (1, 5):
    tau = P.choose_answer_abstain(pc, yc, lam)["tau"]
    ans = p >= tau
    c, w = (ans & (y == 1)).mean(), (ans & (y == 0)).mean()
    out[f"abstain_l{lam}"] = {"tau": tau, "correct": float(c), "abstain": float(1 - ans.mean()), "wrong": float(w)}
    md.append(f"| tuned λ={lam} | {100 * c:.1f} | {100 * (1 - ans.mean()):.1f} | {100 * w:.1f} | {c - w:+.3f} | {c - 5 * w:+.3f} |\n")
md.append(f"| always answer | {100 * y.mean():.1f} | 0.0 | {100 * (1 - y.mean()):.1f} | {2 * y.mean() - 1:+.3f} | "
          f"{y.mean() - 5 * (1 - y.mean()):+.3f} |\n")


# paired bootstrap CIs (2000 resamples)
from sklearn.metrics import roc_auc_score
sac = np.nan_to_num(col["l_sac_prod"], nan=0.0)
f1 = col["f1_top_share"]
auc = lambda s, i: roc_auc_score(y[i], s[i])
area_ = lambda s, i: P.curve_area(P.escalation_curve(s[i], y[i], yl[i], tuple(np.linspace(0, 1, 21))))
md.append("\n| quantity | value [95% CI] |\n|---|---|\n")
for name, fn in (("gate AUROC", lambda i: auc(p, i)),
                 ("gate − SAC-Prod AUROC", lambda i: auc(p, i) - auc(sac, i)),
                 ("gate − consistency AUROC", lambda i: auc(p, i) - auc(f1, i)),
                 ("gate − consistency area", lambda i: area_(p, i) - area_(f1, i))):
    v, lo, hi = M.bootstrap_ci(fn, n, n_boot=2000)
    out.setdefault("ci", {})[name] = (v, lo, hi)
    k = 100 if "AUROC" in name else 1
    md.append(f"| {name} | {k * v:+.3f} [{k * lo:+.3f}, {k * hi:+.3f}] |\n")
md.append(f"\nMean predicted P(correct) {100 * p.mean():.1f}% vs. accuracy {100 * y.mean():.1f}% "
          f"(BIRD calib accuracy {100 * yc.mean():.1f}%)\n")
out["mean_p"] = float(p.mean())

od = os.path.join(a.results, "analysis")
os.makedirs(od, exist_ok=True)
json.dump(out, open(os.path.join(od, "transfer_spider.json"), "w"), indent=1, default=float)
open(os.path.join(od, "transfer_spider.md"), "w").writelines(md)
print("".join(md))
