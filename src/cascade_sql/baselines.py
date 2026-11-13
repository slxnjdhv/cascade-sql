"""Baselines B4 (random), B5 (pre-generation router), B6 (single signal), B7 (oracle).

B1-B3 (local greedy, local self-consistency, large only) come directly from the
cached records, see `evaluate.py`.
"""
from typing import List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_GROUPS


def random_curve(lc, gc, rates: Sequence[float]) -> List[dict]:
    """Expected EX when a random share `rate` of questions is escalated."""
    lc, gc = np.asarray(lc, float), np.asarray(gc, float)
    return [{"rate": float(r), "ex": float((1 - r) * lc.mean() + r * gc.mean())} for r in rates]


def oracle_scores(lc, gc) -> np.ndarray:
    """Confidence an oracle would assign: escalate first where local wrong & large right,
    then where both are wrong, never where local is right."""
    lc, gc = np.asarray(lc), np.asarray(gc)
    s = np.ones(len(lc))
    s[(lc == 0) & (gc == 1)] = 0.0
    s[(lc == 0) & (gc == 0)] = 0.5
    return s


class Router:
    """Pre-generation router (Malekpour et al. 2024 style, classification-based):
    predicts whether the LOCAL model will be correct from information available
    BEFORE generation: question text (TF-IDF) + schema/question size features."""

    prior = FEATURE_GROUPS["prior"]

    def __init__(self, max_features: int = 5000, C: float = 1.0):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features)
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scale = StandardScaler()
        self.clf = LogisticRegression(C=C, max_iter=3000)

    def _X(self, texts, Xprior, fit=False):
        if fit:
            T = self.tfidf.fit_transform(texts)
            P = self.scale.fit_transform(self.imp.fit_transform(Xprior))
        else:
            T = self.tfidf.transform(texts)
            P = self.scale.transform(self.imp.transform(Xprior))
        return sparse.hstack([T, sparse.csr_matrix(P)]).tocsr()

    def fit(self, texts, Xprior, y):
        try:
            X = self._X(texts, Xprior, fit=True)
        except ValueError:  # vocabulary too small (tiny datasets): fall back to min_df=1
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            X = self._X(texts, Xprior, fit=True)
        self.clf.fit(X, y)
        return self

    def predict(self, texts, Xprior):
        return self.clf.predict_proba(self._X(texts, Xprior))[:, 1]
