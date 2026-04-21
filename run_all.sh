#!/bin/bash
# End-to-end reproduction script for the CS 550 project.
#
# Runs every stage required for grading:
#   1. Train and evaluate all seven recommenders.
#   2. Run the LightGCN matched-budget rerun.
#   3. Run the fairness (Coverage / Gini / long-tail) audit (optional task).
#   4. Regenerate LaTeX tables and PDF/PNG plots.
#   5. Compile report.pdf and slides.pdf.
#
# Pre-requisites:
#   pip install numpy pandas scipy scikit-learn torch tqdm matplotlib flask
#   MiKTeX / TeX Live with the acmart class installed.
#
# Usage:
#   bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  # Fall back to `python` on Windows / mixed environments.
  PYTHON="python"
fi

echo "==> (1/6) Train and evaluate every model"
"$PYTHON" -m src.pipeline

echo "==> (2/6) Re-run LightGCN at BPR-matched gradient budget"
"$PYTHON" -m src.rerun_lightgcn

echo "==> (3/6) Fairness audit (Coverage / Gini / long-tail) -- optional task"
"$PYTHON" -m src.fairness

echo "==> (4/6) Build LaTeX tables and plots from the CSVs"
"$PYTHON" -m src.make_artifacts

echo "==> (5/6) Compile report"
cd report
pdflatex -interaction=nonstopmode report.tex >/dev/null
pdflatex -interaction=nonstopmode report.tex >/dev/null   # second pass for refs
cd ..

echo "==> (6/6) Compile slides"
cd slides
pdflatex -interaction=nonstopmode slides.tex >/dev/null
pdflatex -interaction=nonstopmode slides.tex >/dev/null
cd ..

echo ""
echo "Done."
echo "  Report : report/report.pdf"
echo "  Slides : slides/slides.pdf"
echo "  Metrics: results/metrics.csv, results/fairness.csv"
echo ""
echo "To launch the demo:  $PYTHON -m demo.app  (http://localhost:5000)"
echo "To package the zip:  bash make_submission.sh dd1376 jw2070"
