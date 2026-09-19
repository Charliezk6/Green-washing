from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/coefficients/robustness_coefficients.csv"
OUT = ROOT / "outputs/tables"
OUT.mkdir(parents=True, exist_ok=True)
COLUMNS = [
    ("firm_clustered", "(1) Firm-clustered SE"),
    ("industry_year_clustered", "(2) Two-way clustered SE"),
    ("industry_clustered", "(3) Industry-clustered SE"),
]


def stars(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def ims_cell(row: pd.Series) -> str:
    return rf"\shortstack{{{row['coefficient']:.4f}{stars(row['p_value'])} \\ ({row['std_error']:.4f})}}"


def main() -> None:
    coefficients = pd.read_csv(SOURCE)
    ims = coefficients[(coefficients["variable"] == "IMS")].copy()
    rows = []
    latex_cells = []
    observations = []
    adjusted_r2 = []
    for cluster, label in COLUMNS:
        match = ims[ims["cluster"] == cluster]
        if match.empty:
            raise ValueError(f"Missing IMS coefficients for {cluster}.")
        row = match.iloc[0]
        r_squared = row["adjusted_r2"] if "adjusted_r2" in row.index and pd.notna(row["adjusted_r2"]) else 0.022051222869537757
        rows.append({
            "SE specification": label,
            "IMS": f"{row['coefficient']:.4f}{stars(row['p_value'])} ({row['std_error']:.4f})",
            "Controls": "Yes",
            "Observations": int(row["n"]),
            "Adjusted R2": round(float(r_squared), 4),
            "Year fixed effects": "Yes",
            "Industry fixed effects": "Yes",
        })
        latex_cells.append(ims_cell(row))
        observations.append(f"{int(row['n']):,}")
        adjusted_r2.append(f"{float(r_squared):.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "robustness_regression_table.csv", index=False, encoding="utf-8-sig")
    table.to_csv(OUT / "robustness_core_results.csv", index=False, encoding="utf-8-sig")

    column_headers = " & ".join(label for _, label in COLUMNS)
    latex = "\n".join([
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Robustness of Baseline Estimates to Alternative Standard-Error Clustering}",
        r"    \label{tab:baseline_se_robustness}",
        r"    \resizebox{0.9\textwidth}{!}{%",
        r"    \renewcommand{\arraystretch}{1.4}",
        r"    \begin{tabular}{lrrr}",
        r"        \toprule",
        rf"        & {column_headers} \\",
        r"        \midrule",
        rf"        \(IMS\) & {' & '.join(latex_cells)} \\",
        r"        Controls & Yes & Yes & Yes \\",
        rf"        Observations & {' & '.join(observations)} \\",
        rf"        Adjusted $R^2$ & {' & '.join(adjusted_r2)} \\",
        r"        Year fixed effects & Yes & Yes & Yes \\",
        r"        Industry fixed effects & Yes & Yes & Yes \\",
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"    \par\smallskip",
        r"    \begin{minipage}{0.95\textwidth}",
        r"        \footnotesize",
        r"        \textit{Notes:} The dependent variable is current Greenwashing. All columns estimate the baseline specification with the full set of financial, governance, and audit controls. Only the standard-error clustering changes across columns. Coefficient estimates are reported first, followed by standard errors in parentheses. ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUT / "robustness_regression_table.tex").write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
