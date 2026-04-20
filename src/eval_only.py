"""Re-evaluate every pickled model in models/ without retraining.

Used when we want to refresh metrics.csv after a code change that affects
inference but not training (e.g., the __getstate__ fix that strips cached
score matrices).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.data_utils import load_split
from src.pipeline import (
    RATING_MODELS, evaluate_rating, evaluate_topn, RESULTS_DIR, MODELS_DIR,
)


ORDER = ["GlobalMean", "BiasOnly", "ItemKNN", "UserKNN", "MF", "NeuMF", "BPR-MF", "LightGCN"]


def main():
    ds = load_split()
    rows = []
    for name in ORDER:
        p = MODELS_DIR / f"{name}.pkl"
        if not p.exists():
            print(f"[eval] skip {name} (no pickle)")
            continue
        with open(p, "rb") as f:
            m = pickle.load(f)
        row = {"model": name, "train_sec": float("nan")}
        if name in RATING_MODELS:
            row.update(evaluate_rating(m, ds.test))
        else:
            row["MAE"] = float("nan"); row["RMSE"] = float("nan")
        row.update(evaluate_topn(m, ds.train, ds.test, ds.n_users, ds.n_items, k=10))
        rows.append(row)
        print(f"[eval] {name}: {row}")

    df = pd.DataFrame(rows)
    # Restore training times from the frozen train_times.csv if present so that
    # repeated evaluations don't lose the recorded wall-clock cost.
    tt_path = RESULTS_DIR / "train_times.csv"
    if tt_path.exists():
        tt = pd.read_csv(tt_path).set_index("model")
        for m in df.model:
            if m in tt.index:
                df.loc[df.model == m, "train_sec"] = tt.loc[m, "train_sec"]
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
