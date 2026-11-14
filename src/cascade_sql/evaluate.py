"""Gate fitting and experiments E1-E4 over annotated (labelled + featured) records.

File convention:  results/<model>/<split>.seed<k>.ann.jsonl
"""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import baselines as B
from . import metrics as M
from . import policy as P
from .features import ALL_FEATURES, FEATURE_GROUPS
from .gate import Gate, risk_threshold
from .io import read_jsonl

RATES = tuple(np.round(np.linspace(0, 1, 21), 3))
REPORT_RATES = (0.1, 0.2, 0.3)


def ann_path(results: str, model: str, split: str, seed: int) -> str:
    return os.path.join(results, model, f"{split}.seed{seed}.ann.jsonl")


def load(results: str, model: str, split: str, seed: int) -> Dict[str, dict]:
    path = ann_path(results, model, split, seed)
    recs = read_jsonl(path)
    if not recs:
        raise FileNotFoundError(f"no annotated records at {path}; run generate + annotate first")
    return {r["qid"]: r for r in recs}


def matrix(recs: List[dict], feats: List[str], label: str) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    X = np.array([[r["features"].get(f, np.nan) for f in feats] for r in recs], float)
    y = np.array([np.nan if r["labels"].get(label) is None else r["labels"][label] for r in recs], float)
    texts = [r["question"] for r in recs]
    keep = ~np.isnan(y)
    return X[keep], y[keep].astype(int), [t for t, k in zip(texts, keep) if k], keep


# ------------------------------------------------------------------ fit
def fit_gate(results: str, local: str, seed: int, out_dir: str, label: str = "orig",
             learner: str = "logistic", features: Optional[List[str]] = None) -> dict:
    feats = features or ALL_FEATURES
    tr = list(load(results, local, "g_train", seed).values())
    ca = list(load(results, local, "calib", seed).values())
    Xtr, ytr, ttr, _ = matrix(tr, feats, label)
    Xca, yca, _, _ = matrix(ca, feats, label)
    gate = Gate(features=feats, learner=learner).fit(Xtr, ytr).calibrate(Xca, yca)
    prior = FEATURE_GROUPS["prior"]
    Xtr_p, _, _, _ = matrix(tr, prior, label)
    router = B.Router().fit(ttr, Xtr_p, ytr)
    os.makedirs(out_dir, exist_ok=True)
    gate.save(os.path.join(out_dir, "gate.pkl"))
    import pickle
    with open(os.path.join(out_dir, "router.pkl"), "wb") as f:
        pickle.dump(router, f)
    info = {"local": local, "seed": seed, "label": label, "learner": learner, "features": feats,
            "n_train": int(len(ytr)), "train_acc": float(ytr.mean()),
            "n_calib": int(len(yca)), "calib_acc": float(yca.mean()),
            "coefficients": gate.coefficients()}
    with open(os.path.join(out_dir, "gate_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


# ------------------------------------------------------------------ evaluate
def _aligned(local: Dict[str, dict], large: Dict[str, dict], label: str):
    qids = [q for q in local if q in large and local[q]["labels"].get(label) is not None
            and large[q]["labels"].get(label) is not None]
    return [local[q] for q in qids], [large[q] for q in qids]


def _costs(lrecs, grecs, price_in, price_out):
    return {"schema_tokens": [r["schema_tokens"] for r in lrecs],
            "local_latency": [r["latency_s"] for r in lrecs],
            "large_latency": [r["latency_s"] for r in grecs],
            "large_prompt_tokens": [r["prompt_tokens"] for r in grecs],
            "large_output_tokens": [r["output_tokens"] for r in grecs],
            "price_in_per_m": price_in, "price_out_per_m": price_out}


def _greedy_correct(rec: dict, label: str) -> int:
    cl = (rec.get("candidate_labels") or {}).get(label)
    if not cl:
        return int(rec["labels"][label])
    for i, c in enumerate(rec["candidates"]):
        if c["kind"] == "greedy":
            return int(cl[i])
    return int(rec["labels"][label])


def _single_signal_orientation(cal_recs, label) -> Dict[str, float]:
    """+1/-1 per feature so that higher oriented value = more likely correct (chosen on calib)."""
    sign = {}
    for f in ALL_FEATURES:
        X, y, _, _ = matrix(cal_recs, [f], label)
        q = M.signal_quality(X[:, 0], y)
        sign[f] = -1.0 if (q["auroc"] == q["auroc"] and q["auroc"] < 0.5) else 1.0
    return sign


def evaluate(results: str, local: str, large: str, seed: int, gate_dir: str, out_dir: str,
             test_splits: List[str], train_label: str = "orig", price_in: Optional[float] = None,
             price_out: Optional[float] = None, lambdas=(0, 1, 5, 10), esc_costs=(0.0, 0.1, 0.3),
             alphas=(0.05, 0.10), delta: float = 0.05, n_boot: int = 1000, plots: bool = True) -> dict:
    import pickle
    os.makedirs(out_dir, exist_ok=True)
    gate = Gate.load(os.path.join(gate_dir, "gate.pkl"))
    with open(os.path.join(gate_dir, "router.pkl"), "rb") as f:
        router = pickle.load(f)
    feats = gate.features
    prior = FEATURE_GROUPS["prior"]

    # ---- calibration split: thresholds + certificates (original labels only)
    cal_l, cal_g = _aligned(load(results, local, "calib", seed), load(results, large, "calib", seed), train_label)
    Xc, yc, _, _ = matrix(cal_l, feats, train_label)
    pc = gate.predict(Xc)
    gc_cal = np.array([r["labels"][train_label] for r in cal_g], int)
    sign = _single_signal_orientation(cal_l, train_label)
    thresholds = {f"l{lam:g}_c{c:g}": P.choose_thresholds(pc, yc, gc_cal, lam, c, three_way=True)
                  for lam in lambdas for c in esc_costs}
    thresholds_2way = {f"l{lam:g}_c{c:g}": P.choose_thresholds(pc, yc, gc_cal, lam, c, three_way=False)
                       for lam in lambdas for c in esc_costs}
    answer_abstain = {lam: P.choose_answer_abstain(pc, yc, lam) for lam in lambdas}
    certs = {f"a{a:g}": risk_threshold(pc, yc, a, delta) for a in alphas}

    # per-group gates for E1 (fit on g_train, calibrated on calib)
    tr = list(load(results, local, "g_train", seed).values())
    group_gates = {}
    for gname in ["prior", "consistency", "structure", "probability", "execution"]:
        fs = FEATURE_GROUPS[gname]
        try:
            Xg, yg, _, _ = matrix(tr, fs, train_label)
            Xgc, ygc, _, _ = matrix(cal_l, fs, train_label)
            group_gates[gname] = (fs, Gate(fs).fit(Xg, yg).calibrate(Xgc, ygc))
        except ValueError:
            pass

    report = {"local": local, "large": large, "seed": seed, "thresholds": thresholds,
              "thresholds_2way": thresholds_2way, "answer_abstain": {f"l{k:g}": v for k, v in answer_abstain.items()},
              "certificates": certs, "splits": {}}
    rows_e1, rows_e2, rows_e3, rows_e4 = [], [], [], []

    for split in test_splits:
        L, G = load(results, local, split, seed), load(results, large, split, seed)
        versions = sorted({v for r in L.values() for v in r["labels"]})
        for ver in versions:
            lr, gr = _aligned(L, G, ver)
            if len(lr) < 10:
                continue
            X, lc, texts, _ = matrix(lr, feats, ver)
            gc = np.array([r["labels"][ver] for r in gr], int)
            p = gate.predict(X)
            Xp, _, _, _ = matrix(lr, prior, ver)
            pr = router.predict(texts, Xp)
            costs = _costs(lr, gr, price_in, price_out)
            tag = f"{split}/{ver}"

            # ---------------- E1 signal quality
            rows_e1.append({"split": split, "labels": ver, "signal": "gate(all)", **M.signal_quality(p, lc)})
            rows_e1.append({"split": split, "labels": ver, "signal": "router(pre-gen)", **M.signal_quality(pr, lc)})
            for gname, (fs, gg) in group_gates.items():
                Xg, _, _, _ = matrix(lr, fs, ver)
                rows_e1.append({"split": split, "labels": ver, "signal": f"gate({gname})",
                                **M.signal_quality(gg.predict(Xg), lc)})
            for f in ALL_FEATURES:
                Xf, _, _, _ = matrix(lr, [f], ver)
                q = M.signal_quality(sign[f] * Xf[:, 0], lc)
                q["ece"] = q["brier"] = float("nan")  # raw features are not probabilities
                rows_e1.append({"split": split, "labels": ver, "signal": f, **q})

            # ---------------- E2 cascade curves
            f1 = np.nan_to_num(np.array([r["features"]["f1_top_share"] for r in lr]), nan=0.0)
            curves = {"gate": P.escalation_curve(p, lc, gc, RATES),
                      "router": P.escalation_curve(pr, lc, gc, RATES),
                      "consistency_only": P.escalation_curve(f1, lc, gc, RATES),
                      "random": B.random_curve(lc, gc, RATES),
                      "oracle": P.escalation_curve(B.oracle_scores(lc, gc), lc, gc, RATES)}
            ex_local, ex_large = float(lc.mean()), float(gc.mean())
            greedy = np.array([_greedy_correct(r, ver) for r in lr])

            def area_diff(idx):
                return (P.curve_area(P.escalation_curve(p[idx], lc[idx], gc[idx], RATES))
                        - P.curve_area(P.escalation_curve(pr[idx], lc[idx], gc[idx], RATES)))

            ad, ad_lo, ad_hi = M.bootstrap_ci(area_diff, len(lc), n_boot=n_boot, seed=seed)
            split_rep = {"n": int(len(lc)), "ex_local_greedy(B1)": float(greedy.mean()),
                         "ex_local_sc(B2)": ex_local, "ex_large(B3)": ex_large,
                         "curve_area": {k: P.curve_area(v) for k, v in curves.items()},
                         "area_gate_minus_router": {"diff": ad, "lo": ad_lo, "hi": ad_hi}, "curves": curves}
            for rate in REPORT_RATES:
                for name, score in [("gate", p), ("router", pr), ("consistency_only", f1)]:
                    d = P.budget_decisions(score, rate)
                    s = P.summarize(d, lc, gc, costs, lambdas)
                    s["gap_recovered"] = P.gap_recovered(s["ex"], ex_local, ex_large)
                    rows_e2.append({"split": split, "labels": ver, "policy": name, "budget": rate, **s})
                dg = P.budget_decisions(p, rate)
                dr = P.budget_decisions(pr, rate)
                ok_g = (P.outcome(dg, lc, gc) == 1).astype(int)
                ok_r = (P.outcome(dr, lc, gc) == 1).astype(int)
                split_rep[f"mcnemar_gate_vs_router@{rate}"] = M.mcnemar(ok_g, ok_r)

            # ---------------- E3 reliability (thresholds from calib)
            for key, th in thresholds.items():
                for mode, tdict in (("3way", th), ("2way", thresholds_2way[key])):
                    d = P.decide(p, tdict["tau_hi"], tdict["tau_lo"])
                    rows_e3.append({"split": split, "labels": ver, "setting": key, "policy": mode,
                                    **P.summarize(d, lc, gc, costs, lambdas)})
            for lam, th in answer_abstain.items():  # no large model: answer locally or abstain
                d_abst = P.decide_answer_abstain(p, th["tau"])
                rows_e3.append({"split": split, "labels": ver, "setting": f"l{lam:g}", "policy": "answer/abstain",
                                **P.summarize(d_abst, lc, gc, costs, lambdas)})
            rows_e3.append({"split": split, "labels": ver, "setting": "-", "policy": "local only (B2)",
                            **P.summarize(np.full(len(lc), P.ANSWER), lc, gc, costs, lambdas)})
            rows_e3.append({"split": split, "labels": ver, "setting": "-", "policy": "large only (B3)",
                            **P.summarize(np.full(len(lc), P.ESCALATE), lc, gc, costs, lambdas)})

            # ---------------- E4 certificates under this label version
            for key, cert in certs.items():
                ans = p >= cert["tau"]
                m = int(ans.sum())
                rows_e4.append({"split": split, "labels": ver, "alpha": key, "tau": cert["tau"],
                                "calib_coverage": cert["coverage"], "calib_ucb": cert["ucb"],
                                "test_coverage": m / len(p),
                                "test_risk": float((lc[ans] == 0).mean()) if m else float("nan"),
                                "violated": bool(m and (lc[ans] == 0).mean() > float(key[1:]))})
            report["splits"][tag] = split_rep
            if plots:
                _plots(out_dir, tag.replace("/", "_"), curves, p, lc)

    _write_csv(os.path.join(out_dir, "e1_signals.csv"), rows_e1)
    _write_csv(os.path.join(out_dir, "e2_budgets.csv"), rows_e2)
    _write_csv(os.path.join(out_dir, "e3_reliability.csv"), rows_e3)
    _write_csv(os.path.join(out_dir, "e4_certificates.csv"), rows_e4)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    _markdown(out_dir, report, rows_e1, rows_e2, rows_e3, rows_e4)
    return report


def _write_csv(path, rows):
    import csv
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _markdown(out_dir, report, e1, e2, e3, e4):
    lines = [f"# Evaluation report: local=`{report['local']}`, large=`{report['large']}`, seed={report['seed']}\n"]
    for tag, s in report["splits"].items():
        a = s["area_gate_minus_router"]
        lines += [f"## {tag} (n={s['n']})",
                  f"- EX: local greedy (B1) {s['ex_local_greedy(B1)']:.3f} · local self-consistency (B2) "
                  f"{s['ex_local_sc(B2)']:.3f} · large only (B3) {s['ex_large(B3)']:.3f}",
                  "- Area under accuracy–escalation curve: " + ", ".join(
                      f"{k} {v:.3f}" for k, v in s["curve_area"].items()),
                  f"- Gate − router area: {a['diff']:+.4f} (95% CI {a['lo']:+.4f} … {a['hi']:+.4f})", ""]
        split, ver = tag.split("/")
        lines += ["**E1 signal quality**", "",
                  _table([r for r in e1 if r["split"] == split and r["labels"] == ver],
                         ["signal", "n", "acc", "auroc", "auprc_wrong", "ece", "brier"]), "",
                  "**E2 budgets**", "",
                  _table([r for r in e2 if r["split"] == split and r["labels"] == ver],
                         ["policy", "budget", "ex", "gap_recovered", "pe_q", "pe_s", "lat_p95", "rs_l5"]), "",
                  "**E3 reliability (thresholds chosen on calib)**", "",
                  _table([r for r in e3 if r["split"] == split and r["labels"] == ver
                          and (r["setting"] in ("-",) or r["setting"].endswith("_c0.1") or r["policy"] == "answer/abstain")],
                         ["policy", "setting", "ex", "escalation_rate", "abstain_rate", "wrong_rate", "rs_l1", "rs_l5", "rs_l10"]), "",
                  "**E4 certificates**", "",
                  _table([r for r in e4 if r["split"] == split and r["labels"] == ver],
                         ["alpha", "tau", "calib_coverage", "calib_ucb", "test_coverage", "test_risk", "violated"]), ""]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def _plots(out_dir, tag, curves, p, lc):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    styles = {"gate": "-", "router": "--", "consistency_only": "-.", "random": ":", "oracle": (0, (1, 1))}
    for k, c in curves.items():
        ax[0].plot([x["rate"] for x in c], [x["ex"] for x in c], linestyle=styles.get(k, "-"), label=k)
    ax[0].set_xlabel("share of questions escalated")
    ax[0].set_ylabel("execution accuracy")
    ax[0].set_title("Accuracy vs escalation")
    ax[0].legend(frameon=False, fontsize=8)
    bins = M.reliability_bins(p, lc, 10)
    ax[1].plot([0, 1], [0, 1], color="0.7", linewidth=1)
    ax[1].plot([b[0] for b in bins], [b[1] for b in bins], marker="o")
    ax[1].set_xlabel("predicted P(correct)")
    ax[1].set_ylabel("observed accuracy")
    ax[1].set_title("Gate reliability")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_{tag}.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ R0: base-paper reproduction
def reproduce(results: str, model: str, split: str, seed: int, out_dir: str, label: str = "orig") -> list:
    """Base-paper protocol (Maleki et al., AAAI 2026): ONE generation per question at the
    sampling temperature, scored with FTC/SLC/SAC x Avg/Prod; report EX, AUROC and ECE (10 bins).
    We use the first sampled candidate of each record (temperature from the generation config)
    and, as a secondary row, the greedy candidate."""
    from .sac import METHODS, score_candidate
    recs = list(load(results, model, split, seed).values())
    rows = []
    for which in ("sample", "greedy"):
        scores, ys = {m: [] for m in METHODS}, []
        for r in recs:
            cl = (r.get("candidate_labels") or {}).get(label)
            idx = next((i for i, c in enumerate(r["candidates"]) if c["kind"] == which), None)
            if idx is None or cl is None or (r.get("labels") or {}).get(label) is None:
                continue
            sc = score_candidate(r["candidates"][idx])
            if any(np.isnan(v) for v in sc.values()):
                continue
            ys.append(int(cl[idx]))
            for m in METHODS:
                scores[m].append(sc[m])
        if len(set(ys)) < 2:
            continue
        for m in METHODS:
            q = M.signal_quality(scores[m], ys)
            rows.append({"model": model, "split": split, "labels": label, "candidate": which, "method": m,
                         "n": q["n"], "EX": round(100 * q["acc"], 2), "AUROC": round(100 * q["auroc"], 2),
                         "ECE": round(100 * M.ece(scores[m], ys, n_bins=10), 2)})
    os.makedirs(out_dir, exist_ok=True)
    _write_csv(os.path.join(out_dir, f"r0_{model}_{split}_{label}.csv"), rows)
    with open(os.path.join(out_dir, f"r0_{model}_{split}_{label}.md"), "w") as f:
        f.write(f"# R0 reproduction of Maleki et al. (AAAI 2026): {model}, {split}, labels={label}\n\n")
        f.write(_table(rows, ["candidate", "method", "n", "EX", "AUROC", "ECE"]))
    return rows
