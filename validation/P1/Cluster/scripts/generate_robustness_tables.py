"""Write validation-only clustering tables from robustness outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COEF = ROOT / "outputs/coefficients/robustness_coefficients.csv"
WCR = ROOT / "outputs/coefficients/wild_cluster_bootstrap.csv"
COUNTS = ROOT / "outputs/diagnostics/cluster_counts.csv"
NONUNIQUE = ROOT / "outputs/diagnostics/ims_nonunique_summary.csv"
OUT = ROOT / "outputs/tables"

SANDWICH_ROWS = [
    ("firm_clustered", "(1) Firm"),
    ("auditor_clustered", "(2) Audit firm"),
    ("auditor_industry_clustered", "(3) Audit-firm–industry"),
    ("firm_auditor_clustered", "(4) Firm + audit firm"),
    ("auditor_industry_twoway", "(5) Audit firm + industry"),
    ("industry_clustered", "(6) Industry"),
]
WCR_ROWS = [
    ("industry_clustered", "Industry"),
    ("auditor_clustered", "Audit firm"),
]


def stars(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def cell(coefficient: float, std_error: float, p_value: float) -> str:
    return f"{coefficient:.4f}{stars(p_value)} ({std_error:.4f})"


def core_row(coefficients: pd.DataFrame, cluster: str, variable: str) -> pd.Series:
    match = coefficients[(coefficients["cluster"] == cluster) & (coefficients["variable"] == variable)]
    if match.empty:
        raise ValueError(f"Missing {variable} for {cluster}.")
    return match.iloc[0]


def n_clusters_display(row: pd.Series, counts: pd.DataFrame, cluster: str) -> str:
    if cluster == "firm_auditor_clustered":
        return f"{int(row['n_clusters_g1'])} × {int(row['n_clusters_g2'])}"
    if cluster == "auditor_industry_twoway":
        return f"{int(row['n_clusters_g1'])} × {int(row['n_clusters_g2'])}"
    mapping = {
        "firm_clustered": "firm",
        "auditor_clustered": "auditor",
        "auditor_industry_clustered": "auditor_industry",
        "industry_clustered": "industry",
    }
    name = mapping.get(cluster)
    if name and not counts.empty:
        hit = counts[counts["cluster"] == name]
        if not hit.empty:
            return str(int(hit.iloc[0]["n_clusters"]))
    return str(int(row["n_clusters"])) if pd.notna(row.get("n_clusters")) else ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    coefficients = pd.read_csv(COEF)
    counts = pd.read_csv(COUNTS) if COUNTS.exists() else pd.DataFrame()
    sandwich_rows = []
    latex_body = []
    for cluster, label in SANDWICH_ROWS:
        ims = core_row(coefficients, cluster, "IMS")
        imsd = core_row(coefficients, cluster, "IMS_D")
        sandwich_rows.append({
            "SE specification": label,
            "cluster": cluster,
            "IMS": cell(ims["coefficient"], ims["std_error"], ims["p_value"]),
            "IMS_D": cell(imsd["coefficient"], imsd["std_error"], imsd["p_value"]),
            "IMS_se": ims["std_error"],
            "IMS_p": ims["p_value"],
            "IMS_D_se": imsd["std_error"],
            "IMS_D_p": imsd["p_value"],
            "Clusters": n_clusters_display(ims, counts, cluster),
            "Observations": int(ims["n"]),
            "Adjusted R2_IMS": round(float(ims["adjusted_r2"]), 4),
            "Year fixed effects": "Yes",
            "Industry fixed effects": "Yes",
        })
        latex_body.append(
            f"        {label} & {cell(ims['coefficient'], ims['std_error'], ims['p_value'])} & "
            f"{cell(imsd['coefficient'], imsd['std_error'], imsd['p_value'])} & "
            f"{n_clusters_display(ims, counts, cluster)} & {int(ims['n']):,} \\\\"
        )

    sandwich = pd.DataFrame(sandwich_rows)
    sandwich.to_csv(OUT / "treatment_structure_clustering.csv", index=False, encoding="utf-8-sig")
    sandwich.to_csv(OUT / "robustness_core_results.csv", index=False, encoding="utf-8-sig")
    sandwich.to_csv(OUT / "robustness_regression_table.csv", index=False, encoding="utf-8-sig")

    sandwich_tex = "\n".join([
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Baseline Estimates under Treatment-Structure Clustering}",
        r"    \label{tab:treatment_structure_clustering}",
        r"    \resizebox{0.98\textwidth}{!}{%",
        r"    \begin{tabular}{lcccc}",
        r"        \toprule",
        r"        Clustering & \(IMS\) & \(IMS\_D\) & Clusters & N \\",
        r"        \midrule",
        *latex_body,
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"    \par\smallskip",
        r"    \begin{minipage}{0.98\textwidth}",
        r"        \footnotesize",
        r"        \textit{Notes:} Same baseline specification with year and industry fixed effects. Only the standard-error clustering changes. Two-way rows report Cameron--Gelbach--Miller multiway clustering. Coefficient estimates are followed by standard errors in parentheses. ***, **, and * denote two-tailed significance at 1, 5, and 10 percent.",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUT / "treatment_structure_clustering.tex").write_text(sandwich_tex, encoding="utf-8")
    (OUT / "robustness_regression_table.tex").write_text(sandwich_tex, encoding="utf-8")

    wcr = pd.read_csv(WCR)
    wcr_out = []
    wcr_tex_rows = []
    for cluster, label in WCR_ROWS:
        ims = wcr[(wcr["cluster"] == cluster) & (wcr["variable"] == "IMS")].iloc[0]
        imsd = wcr[(wcr["cluster"] == cluster) & (wcr["variable"] == "IMS_D")].iloc[0]
        wcr_out.append({
            "cluster_label": label,
            "cluster": cluster,
            "n_clusters": int(ims["n_clusters"]),
            "n_boot": int(ims["n_boot"]),
            "IMS": f"{ims['coefficient']:.4f}",
            "IMS_sandwich_p": ims["sandwich_p"],
            "IMS_wcr_p": ims["wcr_p"],
            "IMS_CI": f"[{ims['ci_low']:.4f}, {ims['ci_high']:.4f}]",
            "IMS_D": f"{imsd['coefficient']:.4f}",
            "IMS_D_sandwich_p": imsd["sandwich_p"],
            "IMS_D_wcr_p": imsd["wcr_p"],
            "IMS_D_CI": f"[{imsd['ci_low']:.4f}, {imsd['ci_high']:.4f}]",
        })
        wcr_tex_rows.append(
            f"        {label} & {int(ims['n_clusters'])} & {ims['coefficient']:.4f} & "
            f"{ims['sandwich_p']:.4f} & {ims['wcr_p']:.4f} & "
            f"[{ims['ci_low']:.4f}, {ims['ci_high']:.4f}] & "
            f"{imsd['coefficient']:.4f} & {imsd['sandwich_p']:.4f} & {imsd['wcr_p']:.4f} & "
            f"[{imsd['ci_low']:.4f}, {imsd['ci_high']:.4f}] \\\\"
        )
    wcr_table = pd.DataFrame(wcr_out)
    wcr_table.to_csv(OUT / "wild_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    wcr_tex = "\n".join([
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Wild Cluster Bootstrap for Few-Cluster Inference}",
        r"    \label{tab:wild_cluster_bootstrap}",
        r"    \resizebox{0.98\textwidth}{!}{%",
        r"    \begin{tabular}{lccccccccc}",
        r"        \toprule",
        r"        Cluster & G & \(IMS\) & Sandwich \(p\) & WCR \(p\) & \(IMS\) 95\% CI & \(IMS\_D\) & Sandwich \(p\) & WCR \(p\) & \(IMS\_D\) 95\% CI \\",
        r"        \midrule",
        *wcr_tex_rows,
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"    \par\smallskip",
        r"    \begin{minipage}{0.98\textwidth}",
        r"        \footnotesize",
        r"        \textit{Notes:} Restricted wild cluster bootstrap with Webb six-point weights, 999 replications, and the plus-one $p$-value. Confidence intervals are unrestricted wild-cluster percentile intervals. The regression is the baseline specification with year and industry fixed effects.",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUT / "wild_cluster_bootstrap.tex").write_text(wcr_tex, encoding="utf-8")

    if NONUNIQUE.exists():
        nonunique = pd.read_csv(NONUNIQUE)
        nonunique.to_csv(OUT / "ims_assignment_nonunique_summary.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
