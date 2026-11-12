"""OLS with absorbed fixed effects and cluster-robust standard errors.

Fixed effects are removed by alternating projections (repeated group demeaning), which by the
Frisch-Waugh-Lovell theorem gives the same slope coefficients as including one dummy per group.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _codes(g) -> np.ndarray:
    _, inv = np.unique(np.asarray(g), return_inverse=True)
    return inv


def demean(M: np.ndarray, groups: list[np.ndarray], tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Project columns of M off the span of all group indicator sets."""
    M = np.array(M, dtype=float, copy=True)
    if M.ndim == 1:
        M = M[:, None]
    if not groups:
        return M - M.mean(axis=0)
    codes = [_codes(g) for g in groups]
    counts = [np.bincount(c) for c in codes]
    for _ in range(max_iter):
        before = M.copy()
        for c, n in zip(codes, counts):
            for j in range(M.shape[1]):
                M[:, j] -= (np.bincount(c, weights=M[:, j]) / n)[c]
        if len(codes) == 1 or np.max(np.abs(M - before)) < tol:
            break
    return M


@dataclass
class FEResult:
    names: list[str]
    coef: np.ndarray
    se: np.ndarray
    n_obs: int
    n_clusters: int

    def ci(self, name: str, level: float = 0.95) -> tuple[float, float]:
        i = self.names.index(name)
        z = stats.t.ppf(0.5 + level / 2, df=self.n_clusters - 1)
        return float(self.coef[i] - z * self.se[i]), float(self.coef[i] + z * self.se[i])

    def get(self, name: str) -> dict:
        i = self.names.index(name)
        lo, hi = self.ci(name)
        p = 2 * stats.t.sf(abs(self.coef[i] / self.se[i]), df=self.n_clusters - 1)
        return {"estimate": float(self.coef[i]), "se": float(self.se[i]), "ci95": [lo, hi],
                "p_value": float(p), "n_obs": self.n_obs, "n_clusters": self.n_clusters}


def fe_ols(y, X, names: list[str], fixed_effects: list | None = None, cluster=None) -> FEResult:
    """Regress y on X after absorbing fixed effects; cluster-robust (CR1) standard errors.

    With no fixed effects an intercept is absorbed (plain demeaning).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    fes = [np.asarray(f) for f in (fixed_effects or [])]
    Z = demean(np.column_stack([y, X]), fes)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtX_inv = np.linalg.pinv(Xt.T @ Xt)
    beta = XtX_inv @ Xt.T @ yt
    u = yt - Xt @ beta
    n, k = Xt.shape
    if cluster is None:
        cluster = np.arange(n)
    cl = _codes(cluster)
    G = int(cl.max()) + 1
    scores = np.zeros((G, k))
    for j in range(k):
        scores[:, j] = np.bincount(cl, weights=Xt[:, j] * u, minlength=G)
    meat = scores.T @ scores
    # absorbed FE nested in clusters do not count toward k (standard practice, e.g. reghdfe)
    adj = G / (G - 1) * (n - 1) / max(n - k - 1, 1)
    V = adj * XtX_inv @ meat @ XtX_inv
    return FEResult(list(names), beta, np.sqrt(np.clip(np.diag(V), 0, None)), n, G)
