"""Generate LaTeX table + publication-quality plots from results/metrics.csv."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
REP = ROOT / "report"


def latex_table(df: pd.DataFrame) -> str:
    def fmt(x):
        if isinstance(x, float) and np.isnan(x):
            return "---"
        if isinstance(x, float):
            return f"{x:.4f}"
        return str(x)

    # Order models
    order = ["GlobalMean", "BiasOnly", "ItemKNN", "UserKNN", "MF", "NeuMF", "BPR-MF", "LightGCN"]
    df = df.set_index("model").loc[[m for m in order if m in df.model.values] if "model" in df else order]
    cols = ["MAE", "RMSE", "Precision@K", "Recall@K", "F1@K", "NDCG@K", "train_sec"]
    pretty = {
        "Precision@K": "P@10", "Recall@K": "R@10",
        "F1@K": "F$_1$@10", "NDCG@K": "NDCG@10", "train_sec": "t(s)",
    }
    header = " & ".join(["Model"] + [pretty.get(c, c) for c in cols]) + r" \\"
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Test-set results on MovieLens-1M. Lower is better for MAE/RMSE; higher is better for P/R/F$_1$/NDCG. BPR-MF and LightGCN do not predict explicit ratings and are only evaluated on Top-$N$.}",
             r"\label{tab:results}",
             r"\small",
             r"\begin{tabular}{lcccccccr}",
             r"\toprule",
             header, r"\midrule"]
    for m in df.index:
        row = [m] + [fmt(df.loc[m, c]) for c in cols]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def plots(df: pd.DataFrame):
    order = [m for m in ["GlobalMean", "BiasOnly", "ItemKNN", "UserKNN", "MF", "NeuMF", "BPR-MF", "LightGCN"]
             if m in df.model.values]
    d = df.set_index("model").loc[order]

    # Rating metrics (NaN for ranking-only models skipped)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    rating_mods = [m for m in order if not np.isnan(d.loc[m, "MAE"])]
    axes[0].bar(rating_mods, d.loc[rating_mods, "MAE"], color="#1e3a8a")
    axes[0].set_title("MAE (lower is better)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.3)
    axes[1].bar(rating_mods, d.loc[rating_mods, "RMSE"], color="#3b82f6")
    axes[1].set_title("RMSE (lower is better)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "rating_metrics.pdf")
    fig.savefig(RES / "rating_metrics.png", dpi=150)
    plt.close(fig)

    # Top-N metrics
    fig, ax = plt.subplots(figsize=(9, 3.8))
    metrics = ["Precision@K", "Recall@K", "F1@K", "NDCG@K"]
    x = np.arange(len(order))
    width = 0.2
    for k, m in enumerate(metrics):
        ax.bar(x + (k - 1.5) * width, d[m].values, width, label=m.replace("@K", "@10"))
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("metric value")
    ax.set_title("Top-10 recommendation quality")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "topn_metrics.pdf")
    fig.savefig(RES / "topn_metrics.png", dpi=150)
    plt.close(fig)


def fairness_table(df: pd.DataFrame) -> str:
    order = [m for m in ["GlobalMean", "BiasOnly", "ItemKNN", "UserKNN",
                         "MF", "NeuMF", "BPR-MF", "LightGCN"] if m in df.model.values]
    d = df.set_index("model").loc[order]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Exposure-fairness audit on Top-10 recommendations. Higher Coverage and Long-tail Share are fairer; lower Gini is fairer.}",
             r"\label{tab:fairness}",
             r"\small",
             r"\begin{tabular}{lccc}",
             r"\toprule",
             r"Model & Coverage@10 $\uparrow$ & Gini $\downarrow$ & Long-tail Share@10 $\uparrow$ \\",
             r"\midrule"]
    for m in d.index:
        lines.append(f"{m} & {d.loc[m,'Coverage@K']:.4f} & {d.loc[m,'Gini']:.4f} & {d.loc[m,'LongTailShare@K']:.4f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    df = pd.read_csv(RES / "metrics.csv")
    tbl = latex_table(df)
    with open(REP / "results_table.tex", "w") as f:
        f.write(tbl)
    plots(df)
    fair_path = RES / "fairness.csv"
    if fair_path.exists():
        fdf = pd.read_csv(fair_path)
        with open(REP / "fairness_table.tex", "w") as f:
            f.write(fairness_table(fdf))
        print("[artifacts] wrote fairness_table.tex")
    print("[artifacts] wrote results_table.tex and plots to", RES)


if __name__ == "__main__":
    main()
