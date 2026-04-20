"""Exposure-fairness audit of the seven recommenders.

Recommender systems tend to exhibit \emph{popularity bias}: rare items are
starved of exposure while head items are repeatedly surfaced, compounding a
feedback loop that harms long-tail producers and user discovery alike. We
audit the models on three complementary metrics:

  (1) Item Coverage@K: fraction of catalog items recommended to at least
      one user within their top-K list.
  (2) Gini coefficient of the item-exposure distribution: 0 = uniform
      exposure across items; 1 = all exposure concentrated on one item.
  (3) Long-tail share@K: fraction of Top-K slots that land on items in
      the bottom 80% by training popularity. Higher is fairer to tail.

Output: results/fairness.csv and results/fairness.pdf (bar chart).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data_utils import load_split
from src.pipeline import MODELS_DIR, RESULTS_DIR

ORDER = ["GlobalMean", "BiasOnly", "ItemKNN", "UserKNN", "MF", "NeuMF", "BPR-MF", "LightGCN"]


def _gini(exposure: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution. 0 = perfect equality."""
    x = np.sort(exposure.astype(np.float64))
    n = x.size
    if x.sum() == 0:
        return 0.0
    # Using the standard formula: G = (sum_i (2i - n - 1) x_i) / (n sum x_i)
    idx = np.arange(1, n + 1)
    return float((np.sum((2 * idx - n - 1) * x)) / (n * x.sum()))


def audit(k: int = 10) -> pd.DataFrame:
    ds = load_split()
    n_items = ds.n_items
    item_pop = np.bincount(ds.train.i.to_numpy(), minlength=n_items)
    # Bottom 80% by popularity defines the long tail
    pop_order = np.argsort(-item_pop)
    head_size = int(0.2 * n_items)
    head_set = set(pop_order[:head_size].tolist())

    seen = sp.csr_matrix(
        (np.ones(len(ds.train), dtype=bool), (ds.train.u.to_numpy(), ds.train.i.to_numpy())),
        shape=(ds.n_users, n_items),
    )
    # users with >=1 test record (same cohort as topk_metrics)
    test_users = sorted(ds.test.u.unique().tolist())

    # Precompute a mask matrix (users x items, -inf for seen)
    head_mask = np.zeros(n_items, dtype=bool)
    for h in head_set:
        head_mask[h] = True

    rows = []
    test_users_arr = np.asarray(test_users, dtype=np.int64)
    for name in ORDER:
        p = MODELS_DIR / f"{name}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as f:
            model = pickle.load(f)
        print(f"[fairness] scoring {name} ...", flush=True)

        exposure = np.zeros(n_items, dtype=np.int64)
        tail_hits = 0
        total_slots = 0
        for idx, u in enumerate(test_users_arr):
            scores = model.score_all(int(u)).astype(np.float32).copy()
            s, e = seen.indptr[int(u)], seen.indptr[int(u) + 1]
            scores[seen.indices[s:e]] = -np.inf
            top_idx = np.argpartition(-scores, k)[:k]
            top = top_idx[np.argsort(-scores[top_idx])]
            exposure[top] += 1
            tail_hits += int((~head_mask[top]).sum())
            total_slots += k
            if (idx + 1) % 1000 == 0:
                print(f"  {name}: {idx+1}/{len(test_users_arr)}", flush=True)

        coverage = float((exposure > 0).sum() / n_items)
        gini = _gini(exposure)
        tail_share = tail_hits / total_slots if total_slots else 0.0
        rows.append({
            "model": name,
            "Coverage@K": coverage,
            "Gini": gini,
            "LongTailShare@K": tail_share,
        })
        print(f"[fairness] {name}: cov={coverage:.3f} gini={gini:.3f} tail={tail_share:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "fairness.csv", index=False)
    return df


def plot(df: pd.DataFrame):
    order = [m for m in ORDER if m in df.model.values]
    d = df.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    axes[0].bar(order, d["Coverage@K"], color="#1e3a8a")
    axes[0].set_title("Item Coverage@10 (higher is fairer)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(order, d["Gini"], color="#3b82f6")
    axes[1].set_title("Exposure Gini (lower is fairer)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(order, d["LongTailShare@K"], color="#10b981")
    axes[2].set_title("Long-tail Share@10 (higher is fairer)")
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fairness.pdf")
    fig.savefig(RESULTS_DIR / "fairness.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = audit(k=10)
    plot(df)
    print(df.to_string(index=False))
