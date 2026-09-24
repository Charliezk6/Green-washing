"""Write validation-only clustering tables for SPEC from robustness outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COEF = ROOT / "outputs/coefficients/robustness_coefficients.csv"
WCR = ROOT / "outputs/coefficients/wild_cluster_bootstrap.csv"
COUNTS = ROOT / "outputs/diagnostics/cluster_counts.csv"
OUT = ROOT / "outputs/tables"
CORE = "SPEC"

SANDWICH_ROWS = [
    ("firm_clustered", "(1) Firm"),
    ("auditor_clustered", "(2) Audit firm"),
    ("auditor_industry_clustered", "(3) Audit-firm–industry"),
    ("firm_auditor_clustered", "(4) Firm + audit firm"),
    ("auditor_industry_twoway", "(5) Audit firm + industry"),
    ("industry_clustered", "(6) Industry"),
    ("industry_year_clustered", "(7) Industry + year"),
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
    if pd.notna(row.get("n_clusters_g1")) and pd.notna(row.get("n_clusters_g2")):
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
    sandwich_rows, latex_body = [], []
    for cluster, label in SANDWICH_ROWS:
        spec = core_row(coefficients, cluster, CORE)
        sandwich_rows.append({
            "SE specification": label,
            "cluster": cluster,
            "SPEC": cell(spec["coefficient"], spec["std_error"], spec["p_value"]),
            "SPEC_coef": spec["coefficient"],
            "SPEC_se": spec["std_error"],
            "SPEC_p": spec["p_value"],
            "Clusters": n_clusters_display(spec, counts, cluster),
            "Observations": int(spec["n"]),
            "Adjusted R2": round(float(spec["adjusted_r2"]), 4),
            "Year fixed effects": "Yes",
            "Industry fixed effects": "Yes",
        })
        latex_body.append(
            f"        {label} & {cell(spec['coefficient'], spec['std_error'], spec['p_value'])} & "
            f"{n_clusters_display(spec, counts, cluster)} & {int(spec['n']):,} \\\\"
        )
    sandwich = pd.DataFrame(sandwich_rows)
    sandwich.to_csv(OUT / "treatment_structure_clustering_spec.csv", index=False, encoding="utf-8-sig")
    sandwich_tex = "\n".join([
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Baseline SPEC Estimates under Treatment-Structure Clustering}",
        r"    \label{tab:treatment_structure_clustering_spec}",
        r"    \begin{tabular}{lccc}",
        r"        \toprule",
        r"        Clustering & \(SPEC\) & Clusters & N \\",
        r"        \midrule",
        *latex_body,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \par\smallskip",
        r"    \begin{minipage}{0.95\textwidth}",
        r"        \footnotesize",
        r"        \textit{Notes:} Same baseline specification with all controls and year and industry fixed effects. Only the standard-error clustering changes. Two-way rows report Cameron--Gelbach--Miller multiway clustering. ***, **, and * denote two-tailed significance at 1, 5, and 10 percent.",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUT / "treatment_structure_clustering_spec.tex").write_text(sandwich_tex, encoding="utf-8")

    wcr = pd.read_csv(WCR)
    wcr_out, wcr_tex_rows = [], []
    for cluster, label in WCR_ROWS:
        spec = wcr[(wcr["cluster"] == cluster) & (wcr["variable"] == CORE)].iloc[0]
        wcr_out.append({
            "cluster_label": label,
            "cluster": cluster,
            "n_clusters": int(spec["n_clusters"]),
            "n_boot": int(spec["n_boot"]),
            "SPEC": round(float(spec["coefficient"]), 4),
            "SPEC_sandwich_p": spec["sandwich_p"],
            "SPEC_wcr_p": spec["wcr_p"],
            "SPEC_CI": f"[{spec['ci_low']:.4f}, {spec['ci_high']:.4f}]",
        })
        wcr_tex_rows.append(
            f"        {label} & {int(spec['n_clusters'])} & {spec['coefficient']:.4f} & "
            f"{spec['sandwich_p']:.4f} & {spec['wcr_p']:.4f} & [{spec['ci_low']:.4f}, {spec['ci_high']:.4f}] \\\\"
        )
    pd.DataFrame(wcr_out).to_csv(OUT / "wild_cluster_bootstrap_spec.csv", index=False, encoding="utf-8-sig")
    wcr_tex = "\n".join([
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Wild Cluster Bootstrap for SPEC}",
        r"    \label{tab:wild_cluster_bootstrap_spec}",
        r"    \begin{tabular}{lccccc}",
        r"        \toprule",
        r"        Cluster & G & \(SPEC\) & Sandwich \(p\) & WCR \(p\) & 95\% CI \\",
        r"        \midrule",
        *wcr_tex_rows,
        r"        \bottomrule",
        r"    \end{tabular}",
        r"    \par\smallskip",
        r"    \begin{minipage}{0.95\textwidth}",
        r"        \footnotesize",
        r"        \textit{Notes:} Restricted wild cluster bootstrap with Webb six-point weights, 999 replications, and the plus-one $p$-value. Confidence intervals are unrestricted wild-cluster percentile intervals.",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUT / "wild_cluster_bootstrap_spec.tex").write_text(wcr_tex, encoding="utf-8")
    print(sandwich[["SE specification", "SPEC", "Clusters", "Observations"]].to_string(index=False))
    print(pd.DataFrame(wcr_out).to_string(index=False))


if __name__ == "__main__":
    main()
