"""Train every model, produce MAE/RMSE and Top-10 metrics, persist results.

Usage:
    python -m src.pipeline                # full run, all models
    python -m src.pipeline --models MF    # single model
    python -m src.pipeline --quick        # sanity run with 1-2 epochs
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import load_split, Dataset  # noqa: E402
from src.metrics import mae, rmse, topk_metrics  # noqa: E402
from src.models_classical import GlobalMean, UserItemMean, ItemKNN, UserKNN  # noqa: E402
from src.models_torch import MF, NeuMF, BPRMF, LightGCN  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def train_positives(train_df: pd.DataFrame, n_users: int) -> list[np.ndarray]:
    """List of item arrays already seen per user (to exclude from Top-N)."""
    return [g.i.to_numpy() for _, g in train_df.groupby("u", sort=True)]


def test_items_per_user(test_df: pd.DataFrame) -> dict[int, set[int]]:
    return {u: set(g.i.to_numpy().tolist()) for u, g in test_df.groupby("u", sort=True)}


def evaluate_rating(model, test_df: pd.DataFrame) -> dict[str, float]:
    y = test_df.rating.to_numpy(dtype=np.float32)
    users = test_df.u.to_numpy()
    items = test_df.i.to_numpy()
    yhat = model.predict(users, items)
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat)}


def evaluate_topn(model, train_df: pd.DataFrame, test_df: pd.DataFrame,
                  n_users: int, n_items: int, k: int = 10) -> dict[str, float]:
    """For every user in test, rank all items, mask seen-in-train, take top-K."""
    t0 = time.time()
    # Build seen-in-train mask as CSR
    seen = sp.csr_matrix(
        (np.ones(len(train_df), dtype=bool), (train_df.u.to_numpy(), train_df.i.to_numpy())),
        shape=(n_users, n_items),
    )
    test_items = test_items_per_user(test_df)
    recs = {}
    for u in test_items.keys():
        scores = model.score_all(u).astype(np.float32).copy()
        # mask items already seen in training
        s, e = seen.indptr[u], seen.indptr[u + 1]
        scores[seen.indices[s:e]] = -np.inf
        idx = np.argpartition(-scores, k)[:k]
        top = idx[np.argsort(-scores[idx])]
        recs[u] = top.tolist()
    metrics = topk_metrics(recs, test_items, k=k)
    metrics["topN_sec"] = time.time() - t0
    return metrics


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def build_model(name: str, quick: bool):
    # Classical models are shape-agnostic at construction; torch models are built in fit_model.
    if name == "GlobalMean":
        return GlobalMean()
    if name == "BiasOnly":
        return UserItemMean(reg=5.0, n_iter=15)
    if name == "ItemKNN":
        return ItemKNN(k=50, shrinkage=100.0)
    if name == "UserKNN":
        return UserKNN(k=50, shrinkage=100.0)
    if name in {"MF", "NeuMF", "BPR-MF", "LightGCN"}:
        return None  # constructed in fit_model once shapes are known
    raise ValueError(name)


def fit_model(name: str, model, ds: Dataset, quick: bool):
    if name == "GlobalMean":
        model.fit(ds.train); model.register_shape(ds.n_users, ds.n_items); return model
    if name == "BiasOnly":
        return model.fit(ds.train, ds.n_users, ds.n_items)
    if name in {"ItemKNN", "UserKNN"}:
        return model.fit(ds.train, ds.n_users, ds.n_items)
    # Torch models need shape-aware reconstruction since build_model passed None,None
    if name == "MF":
        model = MF(ds.n_users, ds.n_items, n_factors=64)
        return model.fit(ds.train, ds.n_users, ds.n_items,
                         epochs=1 if quick else 20, lr=0.01, reg=1e-4)
    if name == "NeuMF":
        model = NeuMF(ds.n_users, ds.n_items, gmf_dim=32, mlp_dim=32, hidden=(64, 32, 16))
        return model.fit(ds.train, ds.n_users, ds.n_items,
                         epochs=1 if quick else 20, lr=1e-3, reg=1e-5)
    if name == "BPR-MF":
        model = BPRMF(ds.n_users, ds.n_items, n_factors=64)
        return model.fit(ds.train, ds.n_users, ds.n_items,
                         epochs=1 if quick else 20, lr=0.01, reg=1e-5)
    if name == "LightGCN":
        model = LightGCN(ds.n_users, ds.n_items, n_factors=64, n_layers=3)
        return model.fit(ds.train, ds.n_users, ds.n_items,
                         epochs=1 if quick else 15, batch=16384, lr=1e-3, reg=1e-4)
    raise ValueError(name)


# Pointwise rating models: both metrics meaningful
RATING_MODELS = {"GlobalMean", "BiasOnly", "ItemKNN", "UserKNN", "MF", "NeuMF"}
# Ranking-only models: MAE/RMSE not meaningful (scores not calibrated to 1-5)
RANKING_MODELS = {"BPR-MF", "LightGCN"}


def run(models: list[str], quick: bool = False, k: int = 10) -> pd.DataFrame:
    ds = load_split()
    rows = []
    for name in models:
        print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
        t0 = time.time()
        m = build_model(name, quick)
        m = fit_model(name, m, ds, quick)
        train_sec = time.time() - t0

        row = {"model": name, "train_sec": round(train_sec, 2)}
        if name in RATING_MODELS:
            r = evaluate_rating(m, ds.test)
            row.update(r)
        else:
            row["MAE"] = float("nan")
            row["RMSE"] = float("nan")
        tn = evaluate_topn(m, ds.train, ds.test, ds.n_users, ds.n_items, k=k)
        row.update(tn)
        rows.append(row)
        # Persist the trained model for the demo
        try:
            with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
                pickle.dump(m, f)
        except Exception as e:
            print(f"[warn] could not pickle {name}: {e}")
        # Incrementally save after each model so interrupts don't lose work
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    df = pd.DataFrame(rows)
    print("\n=== Final results ===")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[
        "GlobalMean", "BiasOnly", "ItemKNN", "UserKNN",
        "MF", "NeuMF", "BPR-MF", "LightGCN",
    ])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    run(args.models, quick=args.quick, k=args.k)
