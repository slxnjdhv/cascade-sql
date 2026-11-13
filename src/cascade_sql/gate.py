"""Confidence gate: P(local answer is correct | features).

* fit on G-train (BIRD train sample), Platt-calibrate on Calib (BIRD dev minus Mini-Dev)
* learner: logistic regression (main, interpretable) or gradient boosting (ablation)
* risk certificate: highest-coverage threshold whose Clopper-Pearson upper bound on the
  error rate among answered questions is <= alpha. Thresholds are tested in a fixed
  sequence from strict to loose and testing stops at the first failure
  (Learn-then-Test, fixed-sequence), which controls the probability of a false
  certificate at level delta.
"""
import pickle
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Gate:
    features: List[str]
    learner: str = "logistic"
    model: Optional[object] = None
    platt: Optional[LogisticRegression] = None

    def _build(self):
        if self.learner == "logistic":
            return Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                             ("scale", StandardScaler()),
                             ("clf", LogisticRegression(C=1.0, max_iter=2000))])
        if self.learner == "gbm":
            return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                  l2_regularization=1.0)
        raise ValueError(self.learner)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Gate":
        if len(set(y.tolist())) < 2:
            raise ValueError("gate training labels contain a single class")
        self.model = self._build().fit(X, y)
        return self

    def raw(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> "Gate":
        z = _logit(self.raw(X)).reshape(-1, 1)
        self.platt = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.raw(X)
        if self.platt is not None:
            p = self.platt.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        return p

    def coefficients(self):
        if self.learner != "logistic":
            return None
        clf = self.model.named_steps["clf"]
        return dict(zip(self.features, clf.coef_[0].round(4).tolist()))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "Gate":
        with open(path, "rb") as f:
            return pickle.load(f)


def cp_upper(k: int, m: int, delta: float) -> float:
    """One-sided Clopper-Pearson upper bound on a binomial rate."""
    if m == 0:
        return 1.0
    if k >= m:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, m - k))


def min_answerable(alpha: float, delta: float) -> int:
    """Smallest number of answered questions for which even 0 errors can certify risk <= alpha."""
    return int(np.ceil(np.log(delta) / np.log(1 - alpha)))


def risk_threshold(p: Sequence[float], y: Sequence[int], alpha: float, delta: float = 0.05,
                   n_grid: int = 41, method: str = "fixed_sequence", min_coverage: float = 0.05) -> dict:
    """Lowest threshold tau (max coverage) with certified error <= alpha among p >= tau.

    The tau grid is fixed in advance (1.0 -> 0.0 in steps of 1/(n_grid-1)), independent of the data.
    method="fixed_sequence": test from strict to loose, stop at the first failure. Thresholds
        answering fewer than max(min_answerable(alpha, delta), min_coverage * n) questions are
        skipped rather than tested: with very few answered questions one early error would end
        the sequence. Pre-register `min_coverage` and state this choice in the paper.
    method="bonferroni": test every grid point at delta / n_grid; most conservative.
    """
    p, y = np.asarray(p, float), np.asarray(y, int)
    grid = np.linspace(1.0, 0.0, n_grid)
    m_min = max(min_answerable(alpha, delta), int(np.ceil(min_coverage * len(p))))
    level = delta if method == "fixed_sequence" else delta / n_grid
    best = {"tau": float("inf"), "coverage": 0.0, "emp_risk": None, "ucb": None,
            "method": method, "m_min": m_min}
    for tau in grid:
        ans = p >= tau
        m = int(ans.sum())
        if m < m_min:
            continue
        k = int((y[ans] == 0).sum())
        ucb = cp_upper(k, m, level)
        if ucb > alpha:
            if method == "fixed_sequence":
                break
            continue
        if m / len(p) >= best["coverage"]:
            best = {"tau": float(tau), "coverage": m / len(p), "emp_risk": k / m, "ucb": ucb,
                    "method": method, "m_min": m_min}
    return best
