"""Classical recommenders: global mean, user/item mean, user-kNN, item-kNN, biased-SVD MF.

Every model exposes:
    predict(users, items) -> np.ndarray of predicted ratings (clipped to [1,5])
    score_all(u) -> np.ndarray of shape (n_items,) for Top-N ranking
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
class GlobalMean:
    name = "GlobalMean"

    def fit(self, train_df):
        self.mu = float(train_df.rating.mean())
        return self

    def predict(self, users, items):
        return np.full(len(users), self.mu, dtype=np.float32)

    def score_all(self, u):
        return np.clip(np.full(self.n_items, self.mu, dtype=np.float32), 1.0, 5.0)

    def register_shape(self, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items


class UserItemMean:
    """r_hat = mu + b_u + b_i, solved by alternating closed-form updates."""
    name = "BiasOnly"

    def __init__(self, reg: float = 5.0, n_iter: int = 15):
        self.reg = reg
        self.n_iter = n_iter

    def fit(self, train_df, n_users: int, n_items: int):
        self.n_users, self.n_items = n_users, n_items
        u = train_df.u.to_numpy()
        i = train_df.i.to_numpy()
        r = train_df.rating.to_numpy().astype(np.float32)
        self.mu = float(r.mean())
        bu = np.zeros(n_users, dtype=np.float32)
        bi = np.zeros(n_items, dtype=np.float32)
        # Alternating ridge updates
        u_counts = np.bincount(u, minlength=n_users)
        i_counts = np.bincount(i, minlength=n_items)
        for _ in range(self.n_iter):
            # b_u
            num = np.bincount(u, weights=r - self.mu - bi[i], minlength=n_users)
            bu = num / (u_counts + self.reg)
            # b_i
            num = np.bincount(i, weights=r - self.mu - bu[u], minlength=n_items)
            bi = num / (i_counts + self.reg)
        self.bu, self.bi = bu, bi
        return self

    def predict(self, users, items):
        out = self.mu + self.bu[users] + self.bi[items]
        return np.clip(out, 1.0, 5.0).astype(np.float32)

    def score_all(self, u):
        return np.clip(self.mu + self.bu[u] + self.bi, 1.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Neighborhood CF
# ---------------------------------------------------------------------------
def _build_csr(train_df, n_users, n_items, center: bool = True):
    """Returns (R_csr_centered, user_means) where unobserved entries are implicit zeros.

    Centering subtracts each user's mean so implicit-zeros behave as "no signal"
    rather than biasing similarity toward the global mean.
    """
    u = train_df.u.to_numpy()
    i = train_df.i.to_numpy()
    r = train_df.rating.to_numpy().astype(np.float32)
    user_sum = np.bincount(u, weights=r, minlength=n_users)
    user_cnt = np.bincount(u, minlength=n_users).astype(np.float32)
    user_mean = np.divide(user_sum, user_cnt, out=np.zeros_like(user_sum), where=user_cnt > 0)
    r_c = r - user_mean[u] if center else r.copy()
    R = sp.csr_matrix((r_c, (u, i)), shape=(n_users, n_items))
    R_raw = sp.csr_matrix((r, (u, i)), shape=(n_users, n_items))
    return R, R_raw, user_mean


class _KNNBase:
    # Shared pickle hygiene: drop the 6040x3416 float32 score cache before pickling
    # (recomputed lazily after load).
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_scores_full", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._scores_full = None


class ItemKNN(_KNNBase):
    """Item-based CF with adjusted cosine similarity and top-K neighbours.

    Prediction (Sarwar et al., 2001):
        r_hat(u, i) = mu_u + sum_{j in N_k(i) ∩ I_u} sim(i,j) * (r(u,j) - mu_u)
                             / sum |sim(i,j)|
    """
    name = "ItemKNN"

    def __init__(self, k: int = 50, shrinkage: float = 100.0):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, train_df, n_users: int, n_items: int):
        self.n_users, self.n_items = n_users, n_items
        R, R_raw, mu_u = _build_csr(train_df, n_users, n_items, center=True)
        self.R_csr = R.tocsr()            # users x items, centered
        self.R_raw_csr = R_raw.tocsr()
        self.R_csc = R.tocsc()
        self.user_mean = mu_u
        # Cosine sim on centered columns = item vectors
        norms = np.sqrt((self.R_csc.multiply(self.R_csc)).sum(axis=0)).A1 + 1e-9
        # Compute top-k neighbours per item (block-wise for memory)
        print(f"[ItemKNN] computing top-{self.k} neighbours for {n_items} items...")
        neighbours = np.full((n_items, self.k), -1, dtype=np.int32)
        sims = np.zeros((n_items, self.k), dtype=np.float32)
        # Use support count (co-rating count) for shrinkage
        binary = (self.R_csc != 0).astype(np.float32)
        block = 256
        for start in tqdm(range(0, n_items, block)):
            end = min(start + block, n_items)
            cols = self.R_csc[:, start:end]            # users x block
            num = (self.R_csc.T @ cols).toarray()       # items x block
            support = (binary.T @ binary[:, start:end]).toarray()
            denom = np.outer(norms, norms[start:end]) + 1e-9
            sim = num / denom
            sim *= support / (support + self.shrinkage)  # shrinkage
            # zero out self-sim
            for j, col in enumerate(range(start, end)):
                sim[col, j] = 0.0
            # top-k
            for j, col in enumerate(range(start, end)):
                s = sim[:, j]
                if self.k < n_items:
                    idx = np.argpartition(-s, self.k)[: self.k]
                else:
                    idx = np.arange(n_items)
                order = np.argsort(-s[idx])
                neighbours[col] = idx[order]
                sims[col] = s[idx[order]]
        self.neighbours = neighbours
        self.sims = sims
        # Build sparse top-k similarity matrix W: items x items
        rows = np.repeat(np.arange(n_items), self.k)
        cols = neighbours.reshape(-1)
        vals = sims.reshape(-1)
        mask = cols >= 0
        self.W = sp.csr_matrix((vals[mask], (rows[mask], cols[mask])), shape=(n_items, n_items))
        return self

    def _ensure_full(self):
        if getattr(self, "_scores_full", None) is None:
            W = self.W
            W_abs = W.copy(); W_abs.data = np.abs(W_abs.data)
            R = self.R_csr
            R_bin = (R != 0).astype(np.float32)
            num = (R @ W).toarray()                 # users x items
            denom = (R_bin @ W_abs).toarray() + 1e-9
            self._scores_full = self.user_mean[:, None] + num / denom
            self._scores_full = np.clip(self._scores_full, 1.0, 5.0).astype(np.float32)

    def predict(self, users, items):
        self._ensure_full()
        return self._scores_full[np.asarray(users), np.asarray(items)]

    def score_all(self, u):
        self._ensure_full()
        return self._scores_full[u].copy()


class UserKNN(_KNNBase):
    """User-based CF with mean-centered cosine similarity and significance weighting.

    (We use global per-user mean-centering rather than per-pair co-rating centering
    for efficiency; shrinkage by co-rating support count compensates for the
    resulting bias on low-overlap pairs.)
    """
    name = "UserKNN"

    def __init__(self, k: int = 50, shrinkage: float = 100.0):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, train_df, n_users: int, n_items: int):
        self.n_users, self.n_items = n_users, n_items
        R, R_raw, mu_u = _build_csr(train_df, n_users, n_items, center=True)
        self.R_csr = R.tocsr()
        self.R_raw_csr = R_raw.tocsr()
        self.user_mean = mu_u
        norms = np.sqrt((self.R_csr.multiply(self.R_csr)).sum(axis=1)).A1 + 1e-9
        binary = (self.R_csr != 0).astype(np.float32)
        print(f"[UserKNN] computing top-{self.k} neighbours for {n_users} users...")
        neighbours = np.full((n_users, self.k), -1, dtype=np.int32)
        sims = np.zeros((n_users, self.k), dtype=np.float32)
        block = 128
        for start in tqdm(range(0, n_users, block)):
            end = min(start + block, n_users)
            rows = self.R_csr[start:end]
            num = (rows @ self.R_csr.T).toarray()
            support = (binary[start:end] @ binary.T).toarray()
            denom = np.outer(norms[start:end], norms) + 1e-9
            sim = num / denom
            sim *= support / (support + self.shrinkage)
            for j, u in enumerate(range(start, end)):
                sim[j, u] = 0.0
            for j, u in enumerate(range(start, end)):
                s = sim[j]
                if self.k < n_users:
                    idx = np.argpartition(-s, self.k)[: self.k]
                else:
                    idx = np.arange(n_users)
                order = np.argsort(-s[idx])
                neighbours[u] = idx[order]
                sims[u] = s[idx[order]]
        self.neighbours = neighbours
        self.sims = sims
        rows = np.repeat(np.arange(n_users), self.k)
        cols = neighbours.reshape(-1)
        vals = sims.reshape(-1)
        mask = cols >= 0
        self.W = sp.csr_matrix((vals[mask], (rows[mask], cols[mask])), shape=(n_users, n_users))
        return self

    def _ensure_full(self):
        if getattr(self, "_scores_full", None) is None:
            W = self.W
            W_abs = W.copy(); W_abs.data = np.abs(W_abs.data)
            R = self.R_csr
            R_bin = (R != 0).astype(np.float32)
            num = (W @ R).toarray()
            denom = (W_abs @ R_bin).toarray() + 1e-9
            self._scores_full = self.user_mean[:, None] + num / denom
            self._scores_full = np.clip(self._scores_full, 1.0, 5.0).astype(np.float32)

    def predict(self, users, items):
        self._ensure_full()
        return self._scores_full[np.asarray(users), np.asarray(items)]

    def score_all(self, u):
        self._ensure_full()
        return self._scores_full[u].copy()


# ---------------------------------------------------------------------------
# Matrix Factorization — biased SVD trained with SGD (Koren, 2009)
# ---------------------------------------------------------------------------
@dataclass
class BiasedSVD:
    n_factors: int = 64
    n_epochs: int = 25
    lr: float = 0.005
    reg: float = 0.05
    seed: int = 0
    name: str = "BiasedSVD"
    history: list = field(default_factory=list)

    def fit(self, train_df, n_users: int, n_items: int, verbose: bool = True):
        self.n_users, self.n_items = n_users, n_items
        rng = np.random.default_rng(self.seed)
        self.mu = float(train_df.rating.mean())
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors)).astype(np.float32)
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors)).astype(np.float32)
        u = train_df.u.to_numpy()
        i = train_df.i.to_numpy()
        r = train_df.rating.to_numpy().astype(np.float32)
        perm = np.arange(len(r))
        for epoch in range(self.n_epochs):
            rng.shuffle(perm)
            _sgd_epoch(
                perm, u, i, r, self.mu, self.bu, self.bi, self.P, self.Q,
                self.lr, self.reg,
            )
            if verbose:
                pred = self.predict(u, i)
                err = float(np.sqrt(np.mean((pred - r) ** 2)))
                self.history.append(err)
                print(f"[BiasedSVD] epoch {epoch+1}/{self.n_epochs} train_rmse={err:.4f}")
        return self

    def predict(self, users, items):
        out = self.mu + self.bu[users] + self.bi[items] + np.sum(self.P[users] * self.Q[items], axis=1)
        return np.clip(out, 1.0, 5.0).astype(np.float32)

    def score_all(self, u):
        out = self.mu + self.bu[u] + self.bi + self.P[u] @ self.Q.T
        return np.clip(out, 1.0, 5.0).astype(np.float32)


def _sgd_epoch(perm, u, i, r, mu, bu, bi, P, Q, lr, reg):
    # Plain-Python loop — for ML-1M (~800k) this is a few seconds per epoch.
    # (A Cython/Numba kernel would speed this up but keeps deps minimal.)
    for idx in perm:
        uu = u[idx]; ii = i[idx]
        pred = mu + bu[uu] + bi[ii] + P[uu] @ Q[ii]
        err = r[idx] - pred
        bu[uu] += lr * (err - reg * bu[uu])
        bi[ii] += lr * (err - reg * bi[ii])
        p_uu = P[uu].copy()
        P[uu] += lr * (err * Q[ii] - reg * P[uu])
        Q[ii] += lr * (err * p_uu - reg * Q[ii])
