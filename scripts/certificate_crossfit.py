"""Exchangeable control for the risk certificate: split test_b at random into two halves, certify the
answer/abstain threshold on one half, measure realised risk on the other; repeat R times.
usage: PYTHONPATH=src python scripts/certificate_crossfit.py --local qwen-coder-1.5b-mlx"""
import argparse, json, os
import numpy as np
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES
from cascade_sql.gate import Gate, risk_threshold
ap = argparse.ArgumentParser(); ap.add_argument("--local", required=True); ap.add_argument("--reps", type=int, default=500)
ap.add_argument("--results", default="results"); a = ap.parse_args()
gate = Gate.load(os.path.join(a.results, "gates", f"{a.local}.seed0", "gate.pkl"))
te = list(load(a.results, a.local, "test_b", 0).values())
out, md = [], [f"# Certificate cross-fit on test_b halves ({a.local}, {a.reps} random splits, delta=0.05)\n\n",
               "| labels | α | certified (share of splits) | mean coverage | mean realised risk | violation rate (risk>α) |\n|---|---|---|---|---|---|\n"]
for lab in ("orig", "arcwise", "bird_official"):
    X, y, _, _ = matrix(te, ALL_FEATURES, lab); p = gate.predict(X); n = len(y)
    for alpha in (0.2, 0.3, 0.4):
        rng = np.random.default_rng(0); cert = covs = risks = viol = 0; covl, riskl = [], []
        for _ in range(a.reps):
            idx = rng.permutation(n); c, t = idx[: n // 2], idx[n // 2:]
            r = risk_threshold(p[c], y[c], alpha=alpha, delta=0.05)
            if not np.isfinite(r["tau"]): continue
            ans = p[t] >= r["tau"]
            if not ans.any(): continue
            cert += 1; covl.append(ans.mean()); rk = 1 - y[t][ans].mean(); riskl.append(rk); viol += rk > alpha
        row = {"labels": lab, "alpha": alpha, "certified": cert / a.reps,
               "coverage": float(np.mean(covl)) if covl else float("nan"),
               "risk": float(np.mean(riskl)) if riskl else float("nan"), "violation": viol / max(cert, 1)}
        out.append(row)
        md.append(f"| {lab} | {alpha} | {row['certified']:.2f} | {row['coverage']:.3f} | {row['risk']:.3f} | {row['violation']:.3f} |\n")
os.makedirs(os.path.join(a.results, "analysis"), exist_ok=True)
json.dump(out, open(os.path.join(a.results, "analysis", f"cert_crossfit_{a.local}.json"), "w"), indent=1)
open(os.path.join(a.results, "analysis", f"cert_crossfit_{a.local}.md"), "w").writelines(md); print("".join(md))
