"""Re-train LightGCN with a compute budget matched to BPR-MF.

Matched setting:
  - batch = 4096 (same as BPR-MF)
  - epochs = 20 (same as BPR-MF)
  - everything else identical to the original LightGCN run.

The motivation is the audit finding that the original run used batch=16384,
epochs=15 (~750 gradient steps) against BPR-MF's ~3900 gradient steps;
comparing them under that imbalance does not isolate the architectural effect.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from src.data_utils import load_split
from src.metrics import mae, rmse
from src.models_torch import LightGCN
from src.pipeline import MODELS_DIR, RESULTS_DIR, evaluate_rating, evaluate_topn


def main():
    ds = load_split()
    m = LightGCN(ds.n_users, ds.n_items, n_factors=64, n_layers=3)
    t0 = time.time()
    m.fit(ds.train, ds.n_users, ds.n_items,
          epochs=20, batch=4096, lr=1e-3, reg=1e-4)
    train_sec = time.time() - t0
    print(f"[LightGCN matched] train_sec={train_sec:.1f}")
    with open(MODELS_DIR / "LightGCN.pkl", "wb") as f:
        pickle.dump(m, f)
    # Re-evaluate
    tn = evaluate_topn(m, ds.train, ds.test, ds.n_users, ds.n_items, k=10)
    row = {"model": "LightGCN", "train_sec": round(train_sec, 2),
           "MAE": float("nan"), "RMSE": float("nan"), **tn}
    print(row)

    # Update metrics.csv in-place
    df = pd.read_csv(RESULTS_DIR / "metrics.csv")
    df = df[df.model != "LightGCN"]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    # Preserve canonical row order
    order = ["GlobalMean", "BiasOnly", "ItemKNN", "UserKNN", "MF", "NeuMF", "BPR-MF", "LightGCN"]
    df["__ord"] = df.model.map({m: i for i, m in enumerate(order)})
    df = df.sort_values("__ord").drop(columns="__ord").reset_index(drop=True)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    # Also refresh train_times.csv
    tt = pd.read_csv(RESULTS_DIR / "train_times.csv")
    tt.loc[tt.model == "LightGCN", "train_sec"] = round(train_sec, 2)
    tt.to_csv(RESULTS_DIR / "train_times.csv", index=False)
    print("[rerun] updated metrics.csv and train_times.csv")


if __name__ == "__main__":
    main()
