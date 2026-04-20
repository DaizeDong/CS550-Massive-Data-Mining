"""Data loading and user-wise 80/20 train-test split for MovieLens-1M.

Implements the preprocessing strategy required by the CS550 project spec:
for each user, randomly hold out 20% of interactions as test.

We also emit contiguous integer id mappings (userId, movieId -> 0..N-1)
so that downstream models can use dense embedding tables.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "ml-1m")
SPLIT_DIR = os.path.join(DATA_DIR, "split")


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    user_map: dict
    item_map: dict
    movies: pd.DataFrame  # item metadata for the demo and content features


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(
        os.path.join(RAW_DIR, "ratings.dat"),
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        os.path.join(RAW_DIR, "movies.dat"),
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )
    return ratings, movies


def build_split(seed: int = 42, test_frac: float = 0.2) -> Dataset:
    ratings, movies = _load_raw()

    # Filter to users and items with enough interactions so that holding out
    # 20% still leaves a meaningful training signal. ML-1M is already 20-core
    # on the user side; we keep items with >= 5 ratings to stabilise CF.
    item_counts = ratings.groupby("movieId").size()
    keep_items = item_counts[item_counts >= 5].index
    ratings = ratings[ratings.movieId.isin(keep_items)].reset_index(drop=True)

    # Dense id remap
    uniq_users = np.sort(ratings.userId.unique())
    uniq_items = np.sort(ratings.movieId.unique())
    user_map = {int(u): i for i, u in enumerate(uniq_users)}
    item_map = {int(m): i for i, m in enumerate(uniq_items)}
    ratings["u"] = ratings.userId.map(user_map).astype(np.int32)
    ratings["i"] = ratings.movieId.map(item_map).astype(np.int32)
    ratings["rating"] = ratings.rating.astype(np.float32)

    # Per-user stratified split
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for _, g in ratings.groupby("u", sort=False):
        idx = g.index.to_numpy().copy()
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        # Guard: always keep >= 1 train record (true for ML-1M 20-core)
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    train = ratings.loc[train_idx].reset_index(drop=True)
    test = ratings.loc[test_idx].reset_index(drop=True)

    n_users, n_items = len(user_map), len(item_map)
    print(
        f"[split] users={n_users} items={n_items} "
        f"train={len(train):,} test={len(test):,} "
        f"sparsity={1 - len(ratings) / (n_users * n_items):.4f}"
    )

    os.makedirs(SPLIT_DIR, exist_ok=True)
    train.reset_index(drop=True).to_pickle(os.path.join(SPLIT_DIR, "train.pkl"))
    test.reset_index(drop=True).to_pickle(os.path.join(SPLIT_DIR, "test.pkl"))
    movies.reset_index(drop=True).to_pickle(os.path.join(SPLIT_DIR, "movies.pkl"))
    with open(os.path.join(SPLIT_DIR, "maps.json"), "w") as f:
        json.dump({"user_map": user_map, "item_map": item_map}, f)

    return Dataset(train, test, n_users, n_items, user_map, item_map, movies)


def load_split() -> Dataset:
    if not os.path.exists(os.path.join(SPLIT_DIR, "train.pkl")):
        return build_split()
    train = pd.read_pickle(os.path.join(SPLIT_DIR, "train.pkl"))
    test = pd.read_pickle(os.path.join(SPLIT_DIR, "test.pkl"))
    movies = pd.read_pickle(os.path.join(SPLIT_DIR, "movies.pkl"))
    with open(os.path.join(SPLIT_DIR, "maps.json")) as f:
        maps = json.load(f)
    user_map = {int(k): v for k, v in maps["user_map"].items()}
    item_map = {int(k): v for k, v in maps["item_map"].items()}
    return Dataset(train, test, len(user_map), len(item_map), user_map, item_map, movies)


if __name__ == "__main__":
    d = build_split()
    print(d.train.head())
