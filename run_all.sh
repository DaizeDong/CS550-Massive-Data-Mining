#!/bin/bash
# End-to-end reproduction script for the CS 550 project.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> (1/4) Train & evaluate every model"
python3 -m src.pipeline

echo "==> (2/4) Build LaTeX table + plots"
python3 -m src.make_artifacts

echo "==> (3/4) Compile report"
cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null
pdflatex -interaction=nonstopmode report.tex >/dev/null   # second pass for refs
cd ..

echo "==> (4/4) Compile slides"
cd slides && pdflatex -interaction=nonstopmode slides.tex >/dev/null
pdflatex -interaction=nonstopmode slides.tex >/dev/null
cd ..

echo "Done. PDFs: report/report.pdf slides/slides.pdf"
echo "To launch the demo: python3 -m demo.app"
