"""
Plot distribution of IMS and IPS from the combined panel CSV.
Outputs figures to test/results/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = (
    ROOT
    / "Regression"
    / "data"
    / "input"
    / "combined_greenwashing_auditor_expertise_panel.csv"
)
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARS = ["IMS", "IPS"]


def summarize(series: pd.Series) -> dict:
    return {
        "N": int(series.notna().sum()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "skew": float(series.skew()),
    }


def plot_single(series: pd.Series, name: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Histogram + KDE
    ax = axes[0]
    ax.hist(series, bins=50, density=True, color="#4C78A8", alpha=0.75, edgecolor="white")
    series.plot(kind="kde", ax=ax, color="#E45756", linewidth=1.8, label="KDE")
    ax.axvline(series.mean(), color="#F58518", linestyle="--", linewidth=1.5, label=f"Mean={series.mean():.4f}")
    ax.axvline(series.median(), color="#54A24B", linestyle=":", linewidth=1.5, label=f"Median={series.median():.4f}")
    ax.set_title(f"{name} Density")
    ax.set_xlabel(name)
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=9)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(
        series.dropna(),
        vert=True,
        widths=0.45,
        patch_artist=True,
        showfliers=True,
        flierprops=dict(marker="o", markersize=3, alpha=0.35),
    )
    bp["boxes"][0].set_facecolor("#4C78A8")
    bp["boxes"][0].set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("#E45756")
        median.set_linewidth(2)
    ax.set_title(f"{name} Boxplot")
    ax.set_xticklabels([name])
    ax.set_ylabel(name)

    stats = summarize(series)
    fig.suptitle(
        f"{name} Distribution (N={stats['N']}, mean={stats['mean']:.4f}, "
        f"sd={stats['std']:.4f}, skew={stats['skew']:.3f})",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    colors = {"IMS": "#4C78A8", "IPS": "#72B7B2"}
    for ax, name in zip(axes, VARS):
        s = df[name].dropna()
        ax.hist(s, bins=50, density=True, color=colors[name], alpha=0.75, edgecolor="white")
        s.plot(kind="kde", ax=ax, color="#E45756", linewidth=1.8)
        ax.axvline(s.mean(), color="#F58518", linestyle="--", linewidth=1.4, label="Mean")
        ax.axvline(s.median(), color="#54A24B", linestyle=":", linewidth=1.4, label="Median")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("Density")
    fig.suptitle("IMS vs IPS Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    df = pd.read_csv(DATA_PATH, usecols=VARS)
    print(f"Loaded {len(df):,} rows from {DATA_PATH.name}")

    summary_rows = []
    for name in VARS:
        stats = summarize(df[name])
        summary_rows.append({"variable": name, **stats})
        print(
            f"{name}: N={stats['N']}, mean={stats['mean']:.4f}, "
            f"median={stats['median']:.4f}, sd={stats['std']:.4f}, "
            f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
        )
        plot_single(df[name], name, OUT_DIR / f"{name.lower()}_distribution.png")

    plot_combined(df, OUT_DIR / "ims_ips_distribution_combined.png")

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "ims_ips_distribution_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
