from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "检验结果"
OUTPUT_DIR = ROOT / "figures"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "MCC"]
COLORS = ["#2F5D62", "#5E8C61", "#C99545", "#B85C50", "#526D82"]


def load_result(layer: str) -> dict:
    path = RESULT_DIR / f"{layer}_检验结果.json"
    return json.loads(path.read_text(encoding="utf-8"))


def add_metric_bars(ax, values: dict, title: str) -> None:
    scores = [values[metric] for metric in METRICS]
    bars = ax.bar(METRICS, scores, color=COLORS, width=0.68, edgecolor="#303030", linewidth=0.45)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=9)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=20)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.018,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def plot_layer(result: dict) -> None:
    layer = result["layer"]
    fig = plt.figure(figsize=(10.4, 6.3), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[4.1, 1.35])
    majority_ax = fig.add_subplot(grid[0, 0])
    unanimous_ax = fig.add_subplot(grid[0, 1])
    table_ax = fig.add_subplot(grid[1, :])

    majority = result["model_vs_majority"]
    unanimous = result["model_vs_unanimous"]
    add_metric_bars(majority_ax, majority, f"Model vs. Human Majority Vote (n = {majority['sample']:,})")
    add_metric_bars(unanimous_ax, unanimous, f"Model vs. Unanimous Human Labels (n = {unanimous['sample']:,})")

    kappa = result["human_pairwise_kappa"]
    rows = [
        ["Annotator 1 vs. Annotator 2", f"{kappa['annotator_1_vs_2']:.3f}"],
        ["Annotator 1 vs. Annotator 3", f"{kappa['annotator_1_vs_3']:.3f}"],
        ["Annotator 2 vs. Annotator 3", f"{kappa['annotator_2_vs_3']:.3f}"],
    ]
    table_ax.axis("off")
    table = table_ax.table(
        cellText=rows,
        colLabels=["Human Annotator Pair", "Cohen's kappa"],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.7, 0.3],
        bbox=[0.15, 0.02, 0.7, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#777777")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#E6ECEA")
            cell.set_text_props(weight="semibold")
        else:
            cell.set_facecolor("#FAFAF7")

    fig.suptitle(f"{layer} Human Validation Results", fontsize=14, fontweight="bold")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = layer.lower().replace("layer", "layer_") + "_validation"
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for layer in ("LayerA", "LayerB"):
        plot_layer(load_result(layer))
    print(f"Validation figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
