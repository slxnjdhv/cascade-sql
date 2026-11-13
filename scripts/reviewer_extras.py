"""Additional analyses requested in internal review (all from existing outputs; no new generation).
usage: PYTHONPATH=src python scripts/reviewer_extras.py > results/analysis/reviewer_extras.md
"""
import json
import os

import numpy as np
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cascade_sql import metrics as M
from cascade_sql import policy as P
from cascade_sql.evaluate import load, matrix
from cascade_sql.features import ALL_FEATURES, FEATURE_GROUPS
from cascade_sql.gate import Gate, risk_threshold

R = "results"
SM, LG = "qwen-coder-1.5b-mlx", "qwen-coder-3b-mlx"
RATES = tuple(np.linspace(0, 1, 21))
out = {}
gate = Gate.load(f"{R}/gates/{SM}.seed0/gate.pkl")
tr = list(load(R, SM, "g_train", 0).values())
ca = list(load(R, SM, "calib", 0).values())
te = list(load(R, SM, "test_b", 0).values())
Lte = {r["qid"]: r for r in load(R, LG, "test_b", 0).values()}
Lca = {r["qid"]: r for r in load(R, LG, "calib", 0).values()}


def split(recs, lab="orig", feats=ALL_FEATURES):
    X, y, _, k = matrix(recs, feats, lab)
    return X, y, [r for r, kk in zip(recs, k) if kk]


def area(score, lc, gc):
    return P.curve_area(P.escalation_curve(score, lc, gc, RATES))


print("# Reviewer-requested analyses (1.5B -> 3B)\n")
# 1. gate vs execution-consistency area, with CI, per label set
print("## 1. Area difference gate − execution consistency (paired bootstrap, 2000)\n")
for lab in ("orig", "bird_official", "arcwise"):
    X, y, recs = split(te, lab)
    gc = np.array([Lte[r["qid"]]["labels"][lab] or 0 for r in recs])
    p = gate.predict(X)
    f1 = X[:, ALL_FEATURES.index("f1_top_share")]
    d = lambda i: area(p[i], y[i], gc[i]) - area(f1[i], y[i], gc[i])
    v, lo, hi = M.bootstrap_ci(d, len(y), n_boot=2000)
    print(f"- {lab}: {v:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    out[f"area_gate_minus_cons_{lab}"] = (v, lo, hi)

# 2. budgets from calibration quantiles (non-transductive)
print("\n## 2. Budgets set by calibration-split thresholds (realized test rate)\n")
Xc, yc, rc = split(ca)
pc = gate.predict(Xc)
X, y, recs = split(te)
p = gate.predict(X)
gc = np.array([Lte[r["qid"]]["labels"]["orig"] or 0 for r in recs])
for r_ in (0.1, 0.2, 0.3):
    t = np.quantile(pc, r_)
    esc = p < t
    ex = np.where(esc, gc, y).mean()
    print(f"- target {r_:.0%}: threshold {t:.3f}, realized escalation {esc.mean():.1%}, EX {100 * ex:.1f}")
    out[f"calib_budget_{r_}"] = (float(esc.mean()), float(ex))

# 3. ablations: GBM, all-minus-structure, SAC-only content
print("\n## 3. Extra ablations (test, orig)\n")
Xt, yt, _ = split(tr)
for name, feats, learner in (("GBM, all features", ALL_FEATURES, "gbm"),
                             ("logistic, all - structure", [f for f in ALL_FEATURES if f not in FEATURE_GROUPS["structure"]], "logistic"),
                             ("logistic, SAC-Prod only", ["l_sac_prod"], "logistic"),
                             ("logistic, top-cluster share only", ["f1_top_share"], "logistic")):
    A, ya, _ = split(tr, feats=feats)
    B, yb, _ = split(ca, feats=feats)
    T, yt2, _ = split(te, feats=feats)
    g = Gate(features=feats, learner=learner).fit(A, ya).calibrate(B, yb)
    q = M.signal_quality(g.predict(T), yt2)
    print(f"- {name}: AUROC {100 * q['auroc']:.1f}, ECE {100 * q['ece']:.1f}, Brier {100 * q['brier']:.1f}")
    out[f"abl_{name}"] = q
print(f"- (the 'SAC only' row of the paper uses the 6 token features {FEATURE_GROUPS['logit']})")

# 4. 3B + answer/abstain gate (no small-model features) vs two-stage
print("\n## 4. 3B-only with its own answer/abstain gate\n")


def z3(recs, L):
    rows = []
    for r in recs:
        c = L[r["qid"]]["candidates"][L[r["qid"]]["chosen_index"]]
        lp = c.get("logprob")
        rows.append([float(c["exec_ok"]), float(bool(c.get("empty"))), float(bool(c.get("repaired"))),
                     lp if lp is not None else -5.0])
    return np.array(rows)


Zc = z3(rc, Lca)
yLc = np.array([Lca[r["qid"]]["labels"]["orig"] or 0 for r in rc])
Zt = z3(recs, Lte)
oof = np.zeros(len(yLc))
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(Zc, yLc):
    oof[b_] = mk().fit(Zc[a_], yLc[a_]).predict_proba(Zc[b_])[:, 1]
m3 = mk().fit(Zc, yLc)
p3 = m3.predict_proba(Zt)[:, 1]
print(f"- 3B answer/abstain gate AUROC on test: {100 * roc_auc_score(gc, p3):.1f}")
for lam in (1, 5, 10):
    tau = P.choose_answer_abstain(oof, yLc, lam)["tau"]
    ans = p3 >= tau
    c, w = (ans & (gc == 1)).mean(), (ans & (gc == 0)).mean()
    print(f"- tuned λ={lam}: correct {100 * c:.1f}, abstain {100 * (1 - ans.mean()):.1f}, wrong {100 * w:.1f}, "
          f"RS1 {c - w:+.3f}, RS5 {c - 5 * w:+.3f}, RS10 {c - 10 * w:+.3f}")
    out[f"b3_abstain_l{lam}"] = dict(correct=float(c), abstain=float(1 - ans.mean()), wrong=float(w))
n = len(y)
print(f"- exact always-answer RS1 = ({int(y.sum())} - {int(n - y.sum())})/{n} = {(2 * y.sum() - n) / n:+.4f}")

# 5. cost table
print("\n## 5. Cost per question (test)\n")
ls = np.array([r["latency_s"] for r in recs])
ll = np.array([Lte[r["qid"]]["latency_s"] for r in recs])
ts = np.array([r["output_tokens"] for r in recs])
tl = np.array([Lte[r["qid"]]["output_tokens"] for r in recs])
pts = np.array([r["prompt_tokens"] for r in recs])
ptl = np.array([Lte[r["qid"]]["prompt_tokens"] for r in recs])
print(f"- mean prompt tokens processed: 1.5B {pts.mean():.0f}, 3B {ptl.mean():.0f}; schema tokens mean "
      f"{np.mean([r['schema_tokens'] for r in recs]):.0f}")
print("| policy | median s | p95 s | mean s | gen. tokens |\n|---|---|---|---|---|")
pol = {"B2 1.5B self-consistency": (ls, ts), "B3 3B only": (ll, tl)}
o = np.argsort(p)
for r_ in (0.1, 0.2, 0.3):
    esc = np.zeros(n, bool)
    esc[o[: int(round(r_ * n))]] = True
    pol[f"gate cascade @{r_:.0%}"] = (ls + esc * ll, ts + esc * tl)
for k, (lat, tok) in pol.items():
    print(f"| {k} | {np.median(lat):.1f} | {np.percentile(lat, 95):.1f} | {lat.mean():.1f} | {tok.mean():.0f} |")

# 6. kappa sensitivity of the two-stage policy (reuse three_way logic)
print("\n## 6. κ sensitivity, two-stage policy tuned for λ=1\n")
import importlib.util
spec = importlib.util.spec_from_file_location("tw", "scripts/three_way.py")
tw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tw)
Zc2 = tw.stage2_features(rc, Lca, pc)
p2c = np.zeros(len(yLc))
for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(Zc2, yLc):
    p2c[b_] = tw.model().fit(Zc2[a_], yLc[a_]).predict_proba(Zc2[b_])[:, 1]
m2 = tw.model().fit(Zc2, yLc)
p2t = m2.predict_proba(tw.stage2_features(recs, Lte, p))[:, 1]
g1 = np.unique(np.concatenate([[np.inf], np.quantile(pc, np.linspace(0, 1, 51))]))
g2 = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(p2c, np.linspace(0, 1, 51))]))
for kappa in (0.0, 0.05, 0.1, 0.2, 0.5):
    best = max(((a1, a2) for a1 in g1 for a2 in g2),
               key=lambda t: tw.utility(pc, p2c, yc, yLc, t[0], t[1], 1, kappa)["utility"])
    r = tw.utility(p, p2t, y, gc, best[0], best[1], 1, kappa)
    print(f"- κ={kappa}: escalate {100 * r['escalation_rate']:.1f}, abstain {100 * r['abstain_rate']:.1f}, "
          f"correct {100 * r['ex']:.1f}, wrong {100 * r['wrong_rate']:.1f}, RS1 {r['rs_l1']:+.3f}")

# 7. leave-one-database-out gate
print("\n## 7. Leave-one-database-out gate (train/calibrate without the test database)\n")
dbs = sorted(set(r["db_id"] for r in recs))
pl = np.zeros(n)
for db in dbs:
    A, ya, _ = split([r for r in tr if r["db_id"] != db])
    B, yb, _ = split([r for r in ca if r["db_id"] != db])
    g = Gate(features=ALL_FEATURES).fit(A, ya).calibrate(B, yb)
    m = np.array([r["db_id"] == db for r in recs])
    pl[m] = g.predict(X[m])
q_lodo, q_in = M.signal_quality(pl, y), M.signal_quality(p, y)
print(f"- pooled AUROC: in-domain {100 * q_in['auroc']:.1f}, leave-db-out {100 * q_lodo['auroc']:.1f}; "
      f"ECE {100 * q_in['ece']:.1f} -> {100 * q_lodo['ece']:.1f}")
out["lodo"] = (q_in["auroc"], q_lodo["auroc"], q_lodo["ece"])

# 8. certificate: CI of realized risk, and repeated calibration (bootstrap of calib) -> Mini-Dev
print("\n## 8. Risk certificate (answer/abstain)\n")
for alpha in (0.3, 0.4):
    c = risk_threshold(pc, yc, alpha=alpha, delta=0.05)
    ans = p >= c["tau"]
    k, m_ = int((y[ans] == 0).sum()), int(ans.sum())
    lo = beta.ppf(0.025, k, m_ - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, m_ - k)
    print(f"- α={alpha}: answered {m_}, wrong {k}, realized risk {k / m_:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")
rng = np.random.default_rng(0)
for alpha in (0.3, 0.4):
    cert = viol = 0
    risks = []
    for _ in range(500):
        i = rng.integers(0, len(yc), len(yc))
        c = risk_threshold(pc[i], yc[i], alpha=alpha, delta=0.05)
        if not np.isfinite(c["tau"]):
            continue
        ans = p >= c["tau"]
        if not ans.any():
            continue
        cert += 1
        rk = 1 - y[ans].mean()
        risks.append(rk)
        viol += rk > alpha
    print(f"- α={alpha}, 500 bootstrap calibrations -> Mini-Dev: certified {cert / 500:.2f}, "
          f"violations among certified {viol / max(cert, 1):.2f}, overall {viol / 500:.3f}, mean risk {np.mean(risks):.3f}")

# 9. reliability bins with counts
print("\n## 9. Gate reliability bins (test, orig): mean p, accuracy, count\n")
for mp, acc, cnt in M.reliability_bins(p, y, 10):
    print(f"- p={mp:.2f}: acc {acc:.2f} (n={cnt})")
json.dump(out, open(f"{R}/analysis/reviewer_extras.json", "w"), indent=1, default=float)
