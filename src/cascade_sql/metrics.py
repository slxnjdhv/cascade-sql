"""Metrics and statistics: AUROC/AUPRC/ECE/Brier, paired bootstrap, McNemar, Holm."""
from typing import Callable, Dict, Sequence

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ece(p: Sequence[float], y: Sequence[int], n_bins: int = 15) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_bins(p, y, n_bins: int = 10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    return [(float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum()))
            for b in range(n_bins) if (idx == b).any()]


def signal_quality(p, y) -> Dict[str, float]:
    p, y = np.asarray(p, float), np.asarray(y, int)
    ok = ~np.isnan(p)
    p, y = p[ok], y[ok]
    if len(set(y.tolist())) < 2:
        return {"n": int(len(y)), "auroc": float("nan"), "auprc_wrong": float("nan"),
                "ece": float("nan"), "brier": float("nan"), "acc": float(y.mean()) if len(y) else float("nan")}
    return {"n": int(len(y)), "acc": float(y.mean()),
            "auroc": float(roc_auc_score(y, p)),
            "auprc_wrong": float(average_precision_score(1 - y, -p)),
            "ece": ece(np.clip(p, 0, 1), y) if p.min() >= 0 and p.max() <= 1 else float("nan"),
            "brier": float(brier_score_loss(y, np.clip(p, 0, 1))) if p.min() >= 0 and p.max() <= 1 else float("nan")}


def bootstrap_ci(stat: Callable[[np.ndarray], float], n: int, n_boot: int = 10_000,
                 seed: int = 0, level: float = 0.95):
    """Percentile CI of stat(indices) over resampled question indices."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(stat(np.arange(n))), float(lo), float(hi)


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """Mean(a) - mean(b) for paired per-question scores, with 95% CI and two-sided p-value."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p": float(min(1.0, p))}


def mcnemar(a_correct: Sequence[int], b_correct: Sequence[int]) -> Dict[str, float]:
    a, b = np.asarray(a_correct, int), np.asarray(b_correct, int)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    return {"n01": n01, "n10": n10, "p": float(binomtest(n10, n01 + n10, 0.5).pvalue)}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
