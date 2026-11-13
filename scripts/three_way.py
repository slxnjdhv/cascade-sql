"""Three-way cascade with a second-stage gate (answer / escalate / abstain).

Stage 1: the local gate p1 (trained on g_train, calibrated on calib) decides ANSWER locally if p1 >= t1.
Stage 2: otherwise the question is escalated; a second gate p2 scores the LARGE model's answer from
         features available after escalation, and the system shows it if p2 >= t2, else ABSTAINS.
Stage-2 features: p1, large exec_ok / empty / repaired / log-prob, agreement between the large result
and the local chosen result, and the share of local candidates that return the large result.
p2 is fitted on calib (out-of-fold predictions on calib for threshold selection), then refit on all of
calib and applied to test_b. Thresholds (t1, t2) maximise U = +1 correct, -lam wrong, 0 abstain,
-kappa per escalation on calib.

usage: PYTHONPATH=src python scripts/three_way.py --local qwen-coder-1.5b-mlx --large qwen-coder-3b-mlx
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES
from cascade_sql.gate import Gate

S2 = ["p1", "l_exec_ok", "l_empty", "l_repaired", "l_logprob", "agree", "local_share_of_large"]


def stage2_features(local_recs, large, p1):
    rows = []
    for r, p in zip(local_recs, p1):
        L = large[r["qid"]]
        c = L["candidates"][L["chosen_index"]]
        h = c["result_hash"] if c["exec_ok"] else None
        lc = r["candidates"][r["chosen_index"]]
        agree = float(h is not None and lc["exec_ok"] and lc["result_hash"] == h)
        share = (sum(1 for x in r["candidates"] if x["exec_ok"] and x["result_hash"] == h) / len(r["candidates"])
                 if h is not None else 0.0)
        lp = c.get("logprob")
        rows.append([p, float(c["exec_ok"]), float(bool(c.get("empty"))), float(bool(c.get("repaired"))),
                     lp if lp is not None else -5.0, agree, share])
    return np.array(rows, float)


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def utility(p1, p2, yl, yL, t1, t2, lam, kappa):
    ans = p1 >= t1
    show = (~ans) & (p2 >= t2)
    correct = (ans & (yl == 1)) | (show & (yL == 1))
    wrong = (ans & (yl == 0)) | (show & (yL == 0))
    esc = ~ans
    n = len(p1)
    return {"utility": float((correct.sum() - lam * wrong.sum() - kappa * esc.sum()) / n),
            "ex": float(correct.mean()), "wrong_rate": float(wrong.mean()),
            "abstain_rate": float((esc & ~show).mean()), "escalation_rate": float(esc.mean()),
            **{f"rs_l{l}": float((correct.sum() - l * wrong.sum()) / n) for l in (1, 5, 10)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True)
    ap.add_argument("--large", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--kappa", type=float, default=0.1)
    a = ap.parse_args()
    gate = Gate.load(os.path.join(a.results, "gates", f"{a.local}.seed0", "gate.pkl"))
    large_ca = {r["qid"]: r for r in load(a.results, a.large, "calib", 0).values()}
    large_te = {r["qid"]: r for r in load(a.results, a.large, "test_b", 0).values()}
    ca = list(load(a.results, a.local, "calib", 0).values())
    te = list(load(a.results, a.local, "test_b", 0).values())

    Xc, yc, _, kc = matrix(ca, ALL_FEATURES, "orig")
    ca = [r for r, k in zip(ca, kc) if k]
    p1c = gate.predict(Xc)
    yLc = np.array([large_ca[r["qid"]]["labels"]["orig"] or 0 for r in ca])
    Zc = stage2_features(ca, large_ca, p1c)
    # out-of-fold stage-2 predictions on calib for threshold selection
    p2c = np.zeros(len(yLc))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Zc, yLc):
        p2c[va] = model().fit(Zc[tr], yLc[tr]).predict_proba(Zc[va])[:, 1]
    m2 = model().fit(Zc, yLc)
    coef = dict(zip(S2, m2[-1].coef_[0].round(3).tolist()))

    grid1 = np.unique(np.concatenate([[np.inf], np.quantile(p1c, np.linspace(0, 1, 51))]))
    grid2 = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(p2c, np.linspace(0, 1, 51))]))
    out = {"local": a.local, "large": a.large, "kappa": a.kappa, "stage2_coef": coef, "results": []}
    md = [f"# Three-way cascade with stage-2 gate: {a.local} -> {a.large} (kappa={a.kappa})\n\n",
          f"Stage-2 coefficients (standardised): {coef}\n"]
    for lab in ("orig", "arcwise", "bird_official"):
        Xt, yt, _, kt = matrix(te, ALL_FEATURES, lab)
        tr_ = [r for r, k in zip(te, kt) if k]
        p1t = gate.predict(Xt)
        yLt = np.array([large_te[r["qid"]]["labels"][lab] or 0 for r in tr_])
        p2t = m2.predict_proba(stage2_features(tr_, large_te, p1t))[:, 1]
        from sklearn.metrics import roc_auc_score
        auc2 = roc_auc_score(yLt, p2t)
        md.append(f"\n## labels = {lab}: stage-2 AUROC on test (large answer correct) = {100 * auc2:.1f}\n\n"
                  "| policy (tuned for λ) | EX | escalate | abstain | wrong | RS1 | RS5 | RS10 |\n|---|---|---|---|---|---|---|---|\n")
        for lam in (1, 5, 10):
            best = max(((t1, t2) for t1 in grid1 for t2 in grid2),
                       key=lambda t: utility(p1c, p2c, yc, yLc, t[0], t[1], lam, a.kappa)["utility"])
            r = utility(p1t, p2t, yt, yLt, best[0], best[1], lam, a.kappa)
            out["results"].append({"labels": lab, "lam": lam, "t1": best[0], "t2": best[1], "auc2": auc2, **r})
            md.append(f"| 3-way, λ={lam} | {100 * r['ex']:.1f} | {100 * r['escalation_rate']:.1f} | "
                      f"{100 * r['abstain_rate']:.1f} | {100 * r['wrong_rate']:.1f} | {r['rs_l1']:+.3f} | "
                      f"{r['rs_l5']:+.3f} | {r['rs_l10']:+.3f} |\n")
    od = os.path.join(a.results, "analysis")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"three_way_{a.local}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    with open(os.path.join(od, f"three_way_{a.local}.md"), "w") as f:
        f.writelines(md)
    print("".join(md))


if __name__ == "__main__":
    main()
