"""Flask demo for the CS550 recommender system project.

UI lets the user pick a user ID and a model, then shows:
  - top-10 recommended movies (with predicted score)
  - the user's actual training history (for qualitative inspection)
  - per-model metrics table on the test set
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_utils import load_split  # noqa: E402

app = Flask(__name__, template_folder="templates", static_folder="static")

print("[demo] loading dataset and models ...")
DS = load_split()
MOVIES = DS.movies.set_index("movieId")

# inverse id map (dense -> raw movieId)
ITEM_INV = {v: k for k, v in DS.item_map.items()}

# Train-seen mask to exclude from recommendations
SEEN = sp.csr_matrix(
    (np.ones(len(DS.train), dtype=bool), (DS.train.u.to_numpy(), DS.train.i.to_numpy())),
    shape=(DS.n_users, DS.n_items),
)

MODELS_DIR = ROOT / "models"
METRICS = pd.read_csv(ROOT / "results" / "metrics.csv") if (ROOT / "results" / "metrics.csv").exists() else pd.DataFrame()


def available_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted([p.stem for p in MODELS_DIR.glob("*.pkl")])


def load_model(name: str):
    with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def topn(u: int, name: str, k: int = 10) -> list[dict]:
    model = load_model(name)
    scores = model.score_all(u).astype(np.float32).copy()
    s, e = SEEN.indptr[u], SEEN.indptr[u + 1]
    scores[SEEN.indices[s:e]] = -np.inf
    idx = np.argpartition(-scores, k)[:k]
    top = idx[np.argsort(-scores[idx])]
    out = []
    for i in top:
        raw = ITEM_INV[int(i)]
        meta = MOVIES.loc[raw] if raw in MOVIES.index else None
        out.append({
            "movieId": int(raw),
            "title": str(meta["title"]) if meta is not None else f"movie {raw}",
            "genres": str(meta["genres"]) if meta is not None else "",
            "score": float(scores[i]),
        })
    return out


def user_history(u_raw: int, limit: int = 15) -> list[dict]:
    if u_raw not in DS.user_map:
        return []
    u = DS.user_map[u_raw]
    rows = DS.train[DS.train.u == u].sort_values("rating", ascending=False).head(limit)
    out = []
    for _, r in rows.iterrows():
        raw = int(r.movieId)
        meta = MOVIES.loc[raw] if raw in MOVIES.index else None
        out.append({
            "movieId": raw,
            "title": str(meta["title"]) if meta is not None else f"movie {raw}",
            "genres": str(meta["genres"]) if meta is not None else "",
            "rating": float(r.rating),
        })
    return out


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    sample_users = sorted(list(DS.user_map.keys()))[:50]
    return render_template(
        "index.html",
        models=available_models(),
        sample_users=sample_users,
        metrics=METRICS.to_dict(orient="records"),
    )


@app.route("/api/recommend")
def api_recommend():
    u_raw = int(request.args.get("user_id", 1))
    name = request.args.get("model", "MF")
    k = int(request.args.get("k", 10))
    if u_raw not in DS.user_map:
        return jsonify({"error": f"user {u_raw} not in dataset"}), 400
    u = DS.user_map[u_raw]
    return jsonify({
        "user_id": u_raw,
        "model": name,
        "recs": topn(u, name, k=k),
        "history": user_history(u_raw, limit=15),
    })


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS.to_dict(orient="records"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
