"""PyTorch implementations of MF, NeuMF, BPR-MF, LightGCN.

All models expose the same interface as the classical ones:
  fit(train_df, n_users, n_items, ...)
  predict(users, items) -> np.ndarray (ratings 1..5 if rating model, else scores)
  score_all(u) -> np.ndarray
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sample_negatives(u_np: np.ndarray, pos_mask: sp.csr_matrix, n_items: int,
                      rng: np.random.Generator, max_tries: int = 10) -> np.ndarray:
    """Vectorised rejection sampling of one negative per positive.

    On the first pass we draw one random item per position; any collision with a
    training positive is rerolled. ML-1M is ~5% dense, so almost all samples are
    accepted on the first try, and max_tries=10 handles the tail with p≈0.95^10≈0.6
    probability of remaining collision — good enough; any residuals are tolerated.
    """
    n = len(u_np)
    neg = rng.integers(0, n_items, size=n)
    for _ in range(max_tries):
        # boolean array of collisions using sparse indexing
        coll = np.asarray(pos_mask[u_np, neg]).ravel()
        if not coll.any():
            break
        idx = np.where(coll)[0]
        neg[idx] = rng.integers(0, n_items, size=idx.size)
    return neg


# ---------------------------------------------------------------------------
# Biased MF trained with mini-batch SGD in PyTorch
# ---------------------------------------------------------------------------
class MF(nn.Module):
    name = "MF"

    def __init__(self, n_users: int, n_items: int, n_factors: int = 64, mu: float = 3.5):
        super().__init__()
        self.n_users, self.n_items, self.n_factors = n_users, n_items, n_factors
        self.mu = mu
        self.P = nn.Embedding(n_users, n_factors)
        self.Q = nn.Embedding(n_items, n_factors)
        self.bu = nn.Embedding(n_users, 1)
        self.bi = nn.Embedding(n_items, 1)
        nn.init.normal_(self.P.weight, std=0.1)
        nn.init.normal_(self.Q.weight, std=0.1)
        nn.init.zeros_(self.bu.weight)
        nn.init.zeros_(self.bi.weight)

    def forward(self, u, i):
        return (
            self.mu
            + self.bu(u).squeeze(-1)
            + self.bi(i).squeeze(-1)
            + (self.P(u) * self.Q(i)).sum(-1)
        )

    def fit(self, train_df, n_users: int, n_items: int, *, epochs: int = 20,
            batch: int = 8192, lr: float = 0.01, reg: float = 1e-4, verbose: bool = True):
        dev = _device()
        self.to(dev)
        self.mu = float(train_df.rating.mean())
        u = torch.tensor(train_df.u.to_numpy(), dtype=torch.long, device=dev)
        i = torch.tensor(train_df.i.to_numpy(), dtype=torch.long, device=dev)
        r = torch.tensor(train_df.rating.to_numpy(), dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        n = len(r)
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            total = 0.0
            for start in range(0, n, batch):
                b = perm[start:start + batch]
                pred = self(u[b], i[b])
                loss = F.mse_loss(pred, r[b])
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(b)
            if verbose:
                print(f"[MF] epoch {ep+1}/{epochs} train_mse={total/n:.4f}")
        self.eval()
        return self

    @torch.no_grad()
    def predict(self, users, items) -> np.ndarray:
        dev = _device()
        u = torch.tensor(users, dtype=torch.long, device=dev)
        i = torch.tensor(items, dtype=torch.long, device=dev)
        out = self(u, i).cpu().numpy()
        return np.clip(out, 1.0, 5.0).astype(np.float32)

    @torch.no_grad()
    def score_all(self, u: int) -> np.ndarray:
        dev = _device()
        user = torch.tensor([u], dtype=torch.long, device=dev)
        P_u = self.P(user)                  # 1 x d
        bu = self.bu(user).squeeze()
        scores = self.mu + bu + self.bi.weight.squeeze() + (self.Q.weight @ P_u.T).squeeze()
        return np.clip(scores.cpu().numpy(), 1.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Neural Collaborative Filtering (NeuMF), He et al. 2017
# ---------------------------------------------------------------------------
class NeuMF(nn.Module):
    """Combines GMF (generalized matrix factorization) and an MLP tower."""
    name = "NeuMF"

    def __init__(self, n_users, n_items, gmf_dim: int = 32, mlp_dim: int = 32,
                 hidden: tuple = (64, 32, 16), dropout: float = 0.1, mu: float = 3.5):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.mu = mu
        self.P_gmf = nn.Embedding(n_users, gmf_dim)
        self.Q_gmf = nn.Embedding(n_items, gmf_dim)
        self.P_mlp = nn.Embedding(n_users, mlp_dim)
        self.Q_mlp = nn.Embedding(n_items, mlp_dim)
        dims = [2 * mlp_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(gmf_dim + hidden[-1], 1)
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.05)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u, i):
        g = self.P_gmf(u) * self.Q_gmf(i)
        h = self.mlp(torch.cat([self.P_mlp(u), self.Q_mlp(i)], dim=-1))
        return self.mu + self.out(torch.cat([g, h], dim=-1)).squeeze(-1)

    def fit(self, train_df, n_users: int, n_items: int, *, epochs: int = 20,
            batch: int = 8192, lr: float = 0.001, reg: float = 1e-5, verbose: bool = True):
        dev = _device()
        self.to(dev)
        self.mu = float(train_df.rating.mean())
        u = torch.tensor(train_df.u.to_numpy(), dtype=torch.long, device=dev)
        i = torch.tensor(train_df.i.to_numpy(), dtype=torch.long, device=dev)
        r = torch.tensor(train_df.rating.to_numpy(), dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        n = len(r)
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            tot = 0.0
            for start in range(0, n, batch):
                b = perm[start:start + batch]
                pred = self(u[b], i[b])
                loss = F.mse_loss(pred, r[b])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item() * len(b)
            if verbose:
                print(f"[NeuMF] epoch {ep+1}/{epochs} train_mse={tot/n:.4f}")
        self.eval()
        return self

    @torch.no_grad()
    def predict(self, users, items) -> np.ndarray:
        dev = _device()
        u = torch.tensor(users, dtype=torch.long, device=dev)
        i = torch.tensor(items, dtype=torch.long, device=dev)
        # chunk to avoid blowing memory on 200k test set at once
        out = []
        for s in range(0, len(u), 200_000):
            out.append(self(u[s:s+200_000], i[s:s+200_000]).cpu().numpy())
        return np.clip(np.concatenate(out), 1.0, 5.0).astype(np.float32)

    @torch.no_grad()
    def score_all(self, u: int) -> np.ndarray:
        dev = _device()
        all_items = torch.arange(self.n_items, device=dev)
        users = torch.full_like(all_items, u)
        out = self(users, all_items).cpu().numpy()
        return np.clip(out, 1.0, 5.0).astype(np.float32)


# ---------------------------------------------------------------------------
# BPR-MF (Rendle et al., 2009) — pairwise ranking, trained with uniform negative sampling
# ---------------------------------------------------------------------------
class BPRMF(nn.Module):
    """BPR-MF with a scalar item bias added to the bilinear score.

    The original Rendle et al. (2009) formulation defines the score as
    p_u^T q_i. In practice an item-popularity bias improves convergence on
    implicit feedback and is used in the BPR-MF code released with MyMediaLite
    and most subsequent re-implementations. We retain it here for that reason
    and cite the deviation explicitly in the report.
    """
    name = "BPR-MF"

    def __init__(self, n_users: int, n_items: int, n_factors: int = 64):
        super().__init__()
        self.n_users, self.n_items, self.n_factors = n_users, n_items, n_factors
        self.P = nn.Embedding(n_users, n_factors)
        self.Q = nn.Embedding(n_items, n_factors)
        self.bi = nn.Embedding(n_items, 1)  # item-popularity bias (small deviation from Rendle 2009)
        nn.init.normal_(self.P.weight, std=0.05)
        nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.zeros_(self.bi.weight)

    def score(self, u, i):
        return (self.P(u) * self.Q(i)).sum(-1) + self.bi(i).squeeze(-1)

    def fit(self, train_df, n_users: int, n_items: int, *, epochs: int = 20,
            batch: int = 4096, lr: float = 0.01, reg: float = 1e-5, verbose: bool = True,
            seed: int = 0):
        dev = _device()
        self.to(dev)
        u_np = train_df.u.to_numpy()
        i_np = train_df.i.to_numpy()
        # Sparse positive mask for O(1) vectorised rejection sampling
        pos_mask = sp.csr_matrix(
            (np.ones(len(u_np), dtype=bool), (u_np, i_np)),
            shape=(n_users, n_items),
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        rng = np.random.default_rng(seed)
        n = len(u_np)
        u_all = torch.tensor(u_np, dtype=torch.long, device=dev)
        i_all = torch.tensor(i_np, dtype=torch.long, device=dev)
        for ep in range(epochs):
            perm = rng.permutation(n)
            neg = _sample_negatives(u_np, pos_mask, n_items, rng)
            neg_t = torch.tensor(neg, dtype=torch.long, device=dev)
            total = 0.0
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                idx_t = torch.as_tensor(idx, dtype=torch.long, device=dev)
                u = u_all[idx_t]; pos = i_all[idx_t]; n_ = neg_t[idx_t]
                diff = self.score(u, pos) - self.score(u, n_)
                loss = -F.logsigmoid(diff).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            if verbose:
                print(f"[BPR] epoch {ep+1}/{epochs} loss={total/n:.4f}")
        self.eval()
        return self

    @torch.no_grad()
    def score_all(self, u: int) -> np.ndarray:
        dev = _device()
        user = torch.tensor([u], dtype=torch.long, device=dev)
        scores = (self.P(user) @ self.Q.weight.T).squeeze() + self.bi.weight.squeeze()
        return scores.cpu().numpy().astype(np.float32)

    # BPR is a ranking model, but we expose predict for interface parity:
    @torch.no_grad()
    def predict(self, users, items) -> np.ndarray:
        dev = _device()
        u = torch.tensor(users, dtype=torch.long, device=dev)
        i = torch.tensor(items, dtype=torch.long, device=dev)
        return self.score(u, i).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# LightGCN (He et al., SIGIR 2020)
# ---------------------------------------------------------------------------
class LightGCN(nn.Module):
    """Simplified GCN for CF: no self-loop, no feature transform, no non-linearity.
    Propagates embeddings on the user-item bipartite graph.
    Trained with BPR loss.
    """
    name = "LightGCN"

    def __init__(self, n_users: int, n_items: int, n_factors: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users, self.n_items, self.k = n_users, n_items, n_factors
        self.n_layers = n_layers
        self.E_u = nn.Embedding(n_users, n_factors)
        self.E_i = nn.Embedding(n_items, n_factors)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.A_hat: Optional[torch.Tensor] = None  # normalized adj, filled in fit()

    def _build_adj(self, train_df):
        """Return symmetric normalized bipartite adjacency A_hat as a torch sparse tensor.

        A_hat = D^{-1/2} A D^{-1/2}, where A is the (N+M) x (N+M) block matrix
        [[0, R], [R^T, 0]] of user-item interactions.
        """
        u = train_df.u.to_numpy()
        i = train_df.i.to_numpy() + self.n_users  # offset items
        n = self.n_users + self.n_items
        rows = np.concatenate([u, i])
        cols = np.concatenate([i, u])
        data = np.ones(len(rows), dtype=np.float32)
        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
        deg = np.asarray(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        D = sp.diags(d_inv_sqrt)
        A_hat = D @ A @ D
        A_hat = A_hat.tocoo()
        idx = torch.tensor(np.vstack([A_hat.row, A_hat.col]), dtype=torch.long)
        vals = torch.tensor(A_hat.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, vals, size=(n, n)).coalesce()

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        E = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        out = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.A_hat, E)
            out.append(E)
        E_final = torch.stack(out, dim=0).mean(dim=0)
        return E_final[: self.n_users], E_final[self.n_users:]

    def fit(self, train_df, n_users: int, n_items: int, *, epochs: int = 30,
            batch: int = 4096, lr: float = 0.001, reg: float = 1e-4, verbose: bool = True,
            seed: int = 0):
        dev = _device()
        self.to(dev)
        self.A_hat = self._build_adj(train_df).to(dev)
        u_np = train_df.u.to_numpy()
        i_np = train_df.i.to_numpy()
        pos_mask = sp.csr_matrix(
            (np.ones(len(u_np), dtype=bool), (u_np, i_np)),
            shape=(n_users, n_items),
        )
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=reg)
        rng = np.random.default_rng(seed)
        n = len(u_np)
        for ep in range(epochs):
            perm = rng.permutation(n)
            neg = _sample_negatives(u_np, pos_mask, n_items, rng)
            u_t = torch.tensor(u_np, dtype=torch.long, device=dev)
            i_t = torch.tensor(i_np, dtype=torch.long, device=dev)
            neg_t = torch.tensor(neg, dtype=torch.long, device=dev)
            total = 0.0
            for start in range(0, n, batch):
                idx = torch.as_tensor(perm[start:start + batch], dtype=torch.long, device=dev)
                E_U, E_I = self.propagate()
                u = u_t[idx]; pos = i_t[idx]; nn_ = neg_t[idx]
                pu, pi, pn = E_U[u], E_I[pos], E_I[nn_]
                score_pos = (pu * pi).sum(-1)
                score_neg = (pu * pn).sum(-1)
                bpr = -F.logsigmoid(score_pos - score_neg).mean()
                # L2 reg on embeddings only (LightGCN recipe)
                reg_l = (self.E_u(u).pow(2).sum() + self.E_i(pos).pow(2).sum()
                         + self.E_i(nn_).pow(2).sum()) / len(u)
                loss = bpr + reg * reg_l
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * len(idx)
            if verbose:
                print(f"[LightGCN] epoch {ep+1}/{epochs} loss={total/n:.4f}")
        with torch.no_grad():
            E_U, E_I = self.propagate()
            self.E_U_final = E_U.detach()
            self.E_I_final = E_I.detach()
        self.eval()
        return self

    @torch.no_grad()
    def score_all(self, u: int) -> np.ndarray:
        scores = self.E_U_final[u] @ self.E_I_final.T
        return scores.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict(self, users, items) -> np.ndarray:
        dev = _device()
        u = torch.tensor(users, dtype=torch.long, device=dev)
        i = torch.tensor(items, dtype=torch.long, device=dev)
        pu = self.E_U_final[u]; pi = self.E_I_final[i]
        return (pu * pi).sum(-1).cpu().numpy().astype(np.float32)
