"""Local-model-only analysis (needs no escalation target):
  E1  signal quality of the hybrid gate vs. the base paper's single signals (AUROC/AUPRC/ECE/Brier)
  E6  feature-group ablation (gate retrained on g_train, Platt-calibrated on calib)
  E3' answer/abstain risk certificate (Clopper-Pearson, fixed-sequence) at several risk levels
All on test_b, under every label version (orig / arcwise / bird_official) -> N4 robustness.

usage: PYTHONPATH=src python scripts/analyze_local.py --local qwen-coder-1.5b-mlx
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from cascade_sql import metrics as M
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES, FEATURE_GROUPS
from cascade_sql.gate import Gate, risk_threshold

LABELS = ("orig", "arcwise", "bird_official")
SINGLE = ["l_sac_prod", "l_sac_avg", "l_ftc_prod", "l_slc_prod", "f1_top_share", "f4_scf_mean", "f5_logprob"]
G = FEATURE_GROUPS
ABLATIONS = {
    "SAC only (base paper, Platt)": G["logit"],
    "consistency only": G["consistency"],
    "consistency+structure": G["consistency"] + G["structure"],
    "all - logit": [f for f in ALL_FEATURES if f not in G["logit"]],
    "all - consistency": [f for f in ALL_FEATURES if f not in G["consistency"]],
    "all - execution": [f for f in ALL_FEATURES if f not in G["execution"]],
    "all - prior": [f for f in ALL_FEATURES if f not in G["prior"]],
    "hybrid gate (all)": ALL_FEATURES,
}


def auroc_ci(p, y, n_boot=2000):
    def stat(i):
        yy = y[i]
        return roc_auc_score(yy, p[i]) if 0 < yy.sum() < len(yy) else np.nan
    return M.bootstrap_ci(stat, len(y), n_boot=n_boot)


def auroc_diff(pa, pb, y, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    d0 = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    ds = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            ds.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    ds = np.array(ds)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return {"diff": d0, "lo": lo, "hi": hi, "p": float(min(1, 2 * min((ds <= 0).mean(), (ds >= 0).mean())))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tr = list(load(a.results, a.local, "g_train", a.seed).values())
    ca = list(load(a.results, a.local, "calib", a.seed).values())
    te = list(load(a.results, a.local, "test_b", a.seed).values())
    out = {"local": a.local, "e1": [], "e6": [], "cert": [], "acc": {}}

    gates = {}
    for name, feats in ABLATIONS.items():
        Xtr, ytr, _, _ = matrix(tr, feats, "orig")
        Xca, yca, _, _ = matrix(ca, feats, "orig")
        gates[name] = (feats, Gate(features=feats).fit(Xtr, ytr).calibrate(Xca, yca))
    feats, hyb = gates["hybrid gate (all)"]
    Xca, yca, _, _ = matrix(ca, feats, "orig")
    p_ca = hyb.predict(Xca)

    md = [f"# Local-model analysis: {a.local} (test_b)\n"]
    for lab in LABELS:
        X, y, _, _ = matrix(te, ALL_FEATURES, lab)
        if len(y) == 0:
            continue
        out["acc"][lab] = float(y.mean())
        col = {f: X[:, ALL_FEATURES.index(f)] for f in ALL_FEATURES}
        p_h = hyb.predict(X)
        base = np.nan_to_num(col["l_sac_prod"], nan=0.0)
        md.append(f"\n## labels = {lab}  (n={len(y)}, accuracy={100 * y.mean():.1f}%)\n")
        md.append("\n### E1 signal quality\n\n| signal | AUROC [95% CI] | AUPRC(wrong) | ECE | Brier |\n|---|---|---|---|---|\n")
        rows = [(s, np.nan_to_num(col[s], nan=np.nanmin(col[s]) if np.isfinite(np.nanmin(col[s])) else 0)) for s in SINGLE]
        rows.append(("hybrid gate (calibrated)", p_h))
        for s, p in rows:
            q = M.signal_quality(p, y)
            v, lo, hi = auroc_ci(p, y)
            r = {"labels": lab, "signal": s, "auroc": v, "lo": lo, "hi": hi, "auprc_wrong": q["auprc_wrong"],
                 "ece": q["ece"], "brier": q["brier"]}
            out["e1"].append(r)
            f = lambda x: "-" if x is None or not np.isfinite(x) else f"{100 * x:.1f}"
            md.append(f"| {s} | {f(v)} [{f(lo)}, {f(hi)}] | {f(q['auprc_wrong'])} | {f(q['ece'])} | {f(q['brier'])} |\n")
        d = auroc_diff(p_h, base, y)
        out["e1"].append({"labels": lab, "signal": "gate - SAC-Prod", **d})
        md.append(f"\nGate − SAC-Prod AUROC: {100 * d['diff']:+.1f} pts [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}], p={d['p']:.4f}\n")

        md.append("\n### E6 ablation (gate retrained per feature set)\n\n| features | AUROC | ECE | Brier |\n|---|---|---|---|\n")
        for name, (fs, g) in gates.items():
            Xs, ys, _, _ = matrix(te, fs, lab)
            q = M.signal_quality(g.predict(Xs), ys)
            out["e6"].append({"labels": lab, "features": name, **q})
            md.append(f"| {name} | {100 * q['auroc']:.1f} | {100 * q['ece']:.1f} | {100 * q['brier']:.1f} |\n")

        md.append("\n### Answer/abstain risk certificate (calibrated on calib, applied to test)\n\n"
                  "| target risk α | τ | calib coverage | calib UCB | test coverage | test risk |\n|---|---|---|---|---|---|\n")
        for alpha in (0.10, 0.20, 0.30, 0.40):
            c = risk_threshold(p_ca, yca, alpha=alpha, delta=0.05)
            ans = p_h >= c["tau"]
            cov = float(ans.mean())
            risk = float(1 - y[ans].mean()) if ans.any() else float("nan")
            out["cert"].append({"labels": lab, "alpha": alpha, "tau": c["tau"], "calib_cov": c.get("coverage"),
                                "calib_ucb": c.get("ucb"), "test_cov": cov, "test_risk": risk})
            fmt = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.3f}"
            md.append(f"| {alpha:.2f} | {fmt(c['tau'])} | {fmt(c.get('coverage'))} | {fmt(c.get('ucb'))} | "
                      f"{cov:.3f} | {fmt(risk)} |\n")

    od = os.path.join(a.results, "analysis")
    os.makedirs(od, exist_ok=True)
    tag = a.local + (f".seed{a.seed}" if a.seed else "")
    with open(os.path.join(od, f"local_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    with open(os.path.join(od, f"local_{tag}.md"), "w") as f:
        f.writelines(md)
    print("".join(md))


if __name__ == "__main__":
    main()
