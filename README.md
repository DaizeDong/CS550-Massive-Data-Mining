# CS 550 course project — seven recommenders on MovieLens-1M

Baselines through LightGCN on the same split, with matching evaluation. Everything written from NumPy/PyTorch primitives; no recommender library is imported.

## Layout

```
src/
  data_utils.py        MovieLens-1M loader, per-user 80/20 split
  metrics.py           MAE, RMSE, P@K, R@K, F1@K, NDCG@K
  models_classical.py  GlobalMean, BiasOnly, ItemKNN, UserKNN
  models_torch.py      MF, NeuMF, BPR-MF, LightGCN
  pipeline.py          trains all models end-to-end, writes results/metrics.csv
  eval_only.py         re-evaluates already-pickled models
  rerun_lightgcn.py    LightGCN only, at matched (BPR-equivalent) budget
  fairness.py          coverage / Gini / long-tail share audit
  make_artifacts.py    LaTeX tables + plots from the CSVs
demo/                  Flask single-page demo
report/                LaTeX report (report.pdf)
slides/                Beamer slides (slides.pdf)
data/ml-1m/            raw MovieLens-1M
models/                pickled checkpoints
results/               metrics.csv, fairness.csv, PDF/PNG plots
run_all.sh             end-to-end reproduction
```

## Reproduce

```bash
pip install numpy pandas scipy scikit-learn torch tqdm matplotlib flask
python -m src.pipeline           # trains all seven, ~12 min CPU
python -m src.fairness           # exposure-fairness audit, ~3 min
python -m src.make_artifacts     # regenerates LaTeX tables and plots
cd report  && pdflatex -interaction=nonstopmode report.tex
cd ../slides && pdflatex -interaction=nonstopmode slides.tex
python -m demo.app               # http://localhost:5000
```

`run_all.sh` runs everything except the demo.

## Notes on design choices

The split is per-user uniform-random, as the spec requires. For ML-1M (20-core on users) this leaves every user with $\ge$1 training and $\ge$1 test row, so no special casing.

BPR-MF and LightGCN produce uncalibrated scores, not 1–5 ratings. Their MAE/RMSE cells are reported as N/A rather than as misleading scalars.

LightGCN is evaluated twice: a short run (15 epochs, batch 16 384) and a matched-budget run (20 epochs, batch 4 096, same gradient-step count as BPR-MF). Both are stored; the matched run is the one cited in the report.

The pickled kNN checkpoints strip a cached score matrix (~82 MB) via `__getstate__`; it is rebuilt on first use after load.

## What is not done

- Cold-start serving (models are transductive).
- Timestamps are unused.
- Single seed per model, no bootstrap confidence intervals.
- LightGCN layer count is fixed at 3; a sweep is the obvious next thing to try.
