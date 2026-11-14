"""Paired bootstrap CIs requested in pre-submission review:
(a) RS_1 of the two-stage cascade (tuned for lambda=1) minus RS_1 of the 3B model with its own gate (B8, tuned
    for lambda=1), per sampling seed and label set;
(b) area under the accuracy-escalation curve of the hybrid gate minus that of random escalation (expected value).
usage: PYTHONPATH=src python scripts/paired_reliability.py > results/analysis/paired_reliability.md"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cascade_sql import metrics as M
from cascade_sql import policy as P
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES
from cascade_sql.gate import Gate

import importlib.util, os, sys
spec = importlib.util.spec_from_file_location("tw", os.path.join(os.path.dirname(__file__), "three_way.py"))
tw = importlib.util.module_from_spec(spec); spec.loader.exec_module(tw)

R, SM, LG, KAPPA, LAM = "results", "qwen-coder-1.5b-mlx", "qwen-coder-3b-mlx", 0.1, 1
RATES = tuple(np.linspace(0, 1, 21))
Lca = {r["qid"]: r for r in load(R, LG, "calib", 0).values()}
Lte = {r["qid"]: r for r in load(R, LG, "test_b", 0).values()}
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def z3(recs, L):
    rows = []
    for r in recs:
        c = L[r["qid"]]["candidates"][L[r["qid"]]["chosen_index"]]
        lp = c.get("logprob")
        rows.append([float(c["exec_ok"]), float(bool(c.get("empty"))), float(bool(c.get("repaired"))),
                     lp if lp is not None else -5.0])
    return np.array(rows)


def score(correct, wrong):  # per-question contribution to RS_1
    return correct.astype(float) - LAM * wrong.astype(float)


print("# Paired bootstrap (2,000 resamples over test questions)\n")
print("| seed | labels | RS1 two-stage | RS1 B8 | difference [95% CI] | gate area − random area [95% CI] |")
print("|---|---|---|---|---|---|")
for seed in (0, 1):
    gate = Gate.load(f"{R}/gates/{SM}.seed{seed}/gate.pkl")
    ca = list(load(R, SM, "calib", seed).values())
    te = list(load(R, SM, "test_b", seed).values())
    Xc, yc, _, kc = matrix(ca, ALL_FEATURES, "orig")
    ca = [r for r, k in zip(ca, kc) if k]
    p1c = gate.predict(Xc)
    yLc = np.array([Lca[r["qid"]]["labels"]["orig"] or 0 for r in ca])
    Zc = tw.stage2_features(ca, Lca, p1c)
    p2c = np.zeros(len(yLc))
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(Zc, yLc):
        p2c[b_] = mk().fit(Zc[a_], yLc[a_]).predict_proba(Zc[b_])[:, 1]
    m2 = mk().fit(Zc, yLc)
    g1 = np.unique(np.concatenate([[np.inf], np.quantile(p1c, np.linspace(0, 1, 51))]))
    g2 = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(p2c, np.linspace(0, 1, 51))]))
    t1, t2 = max(((a, b) for a in g1 for b in g2),
                 key=lambda t: tw.utility(p1c, p2c, yc, yLc, t[0], t[1], LAM, KAPPA)["utility"])
    # B8 (3B + own gate), fitted on calib with out-of-fold threshold selection
    Z3c = z3(ca, Lca)
    oof = np.zeros(len(yLc))
    for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(Z3c, yLc):
        oof[b_] = mk().fit(Z3c[a_], yLc[a_]).predict_proba(Z3c[b_])[:, 1]
    m3 = mk().fit(Z3c, yLc)
    tau3 = P.choose_answer_abstain(oof, yLc, LAM)["tau"]
    for lab in ("orig", "arcwise", "bird_official"):
        Xt, yt, _, kt = matrix(te, ALL_FEATURES, lab)
        tr_ = [r for r, k in zip(te, kt) if k]
        p1t = gate.predict(Xt)
        yLt = np.array([Lte[r["qid"]]["labels"][lab] or 0 for r in tr_])
        p2t = m2.predict_proba(tw.stage2_features(tr_, Lte, p1t))[:, 1]
        ans = p1t >= t1
        show = (~ans) & (p2t >= t2)
        s_c = score((ans & (yt == 1)) | (show & (yLt == 1)), (ans & (yt == 0)) | (show & (yLt == 0)))
        a3 = m3.predict_proba(z3(tr_, Lte))[:, 1] >= tau3
        s_b = score(a3 & (yLt == 1), a3 & (yLt == 0))
        d = M.bootstrap_ci(lambda i: s_c[i].mean() - s_b[i].mean(), len(yt), n_boot=2000)
        ar = M.bootstrap_ci(lambda i: P.curve_area(P.escalation_curve(p1t[i], yt[i], yLt[i], RATES))
                            - ((yt[i] + yLt[i]) / 2).mean(), len(yt), n_boot=2000)
        print(f"| {seed} | {lab} | {s_c.mean():+.3f} | {s_b.mean():+.3f} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | "
              f"{ar[0]:+.4f} [{ar[1]:+.4f}, {ar[2]:+.4f}] |")
