"""Evaluation metrics for rating prediction and top-N recommendation.

Rating prediction: MAE, RMSE.
Top-N: Precision@K, Recall@K, F1@K, NDCG@K — computed per user and averaged.
"""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _dcg(relevances: np.ndarray) -> float:
    # relevances: 1-d array of binary (or graded) gains, in predicted order
    if relevances.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances * discounts))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    rel = np.array([1.0 if i in relevant else 0.0 for i in recommended[:k]])
    dcg = _dcg(rel)
    ideal_hits = min(len(relevant), k)
    idcg = _dcg(np.ones(ideal_hits)) if ideal_hits > 0 else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def topk_metrics(
    recs_per_user: dict[int, list[int]],
    test_items_per_user: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float]:
    """Return mean Precision@k, Recall@k, F1@k, NDCG@k over users with >=1 test item."""
    precisions, recalls, ndcgs = [], [], []
    for u, test_items in test_items_per_user.items():
        if not test_items:
            continue
        rec = recs_per_user.get(u, [])[:k]
        hits = sum(1 for i in rec if i in test_items)
        p = hits / k
        r = hits / len(test_items)
        precisions.append(p)
        recalls.append(r)
        ndcgs.append(ndcg_at_k(rec, test_items, k))
    p = float(np.mean(precisions)) if precisions else 0.0
    r = float(np.mean(recalls)) if recalls else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    n = float(np.mean(ndcgs)) if ndcgs else 0.0
    return {"Precision@K": p, "Recall@K": r, "F1@K": f, "NDCG@K": n, "K": k}
