from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
EB_SOURCE = PROJECT / "validation/P1/EB/results2/binary_eb_regressions.csv"
PSM_SOURCE = PROJECT / "validation/P1/PSM/results/tables/outcome_regressions.csv"
OUT = ROOT / "results/tables"
OUT.mkdir(parents=True, exist_ok=True)
EMPTY = "---"


def stars(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def coef_cell(coefficient: float, p_value: float, std_error: float) -> str:
    return rf"\shortstack{{{coefficient:.4f}{stars(p_value)} \\ ({std_error:.4f})}}"


def load_eb() -> pd.Series:
    data = pd.read_csv(EB_SOURCE, encoding="utf-8-sig")
    match = data[
        (data["第一阶段规格"] == "完整共同原因加审计选择")
        & (data["模型"] == "B2_二元EB_连续IMS_完整滞后控制")
    ]
    if match.empty:
        raise ValueError("Missing entropy-balancing B2 specification.")
    return match.iloc[0]


def load_psm() -> pd.Series:
    data = pd.read_csv(PSM_SOURCE, encoding="utf-8-sig")
    match = data[
        (data["规格"] == "PS-Full")
        & (data["模型"] == "M2_完整滞后协变量+固定效应")
    ]
    if match.empty:
        raise ValueError("Missing PS-Full M2 specification.")
    return match.iloc[0]


def main() -> None:
    eb = load_eb()
    psm = load_psm()
    eb_n = int(eb["样本数"])
    psm_n = int(psm["样本数"])
    eb_firms = int(eb["企业数"])
    psm_firms = int(psm["企业数"])
    eb_ess = int(round(float(eb["有效样本量ESS"])))
    psm_pairs = psm_n // 2

    summary = pd.DataFrame([
        {
            "specification": "(1) Entropy Balancing",
            "variable": eb["变量"],
            "coefficient": float(eb["系数"]),
            "std_error": float(eb["标准误"]),
            "p_value": float(eb["p值"]),
            "observations": eb_n,
            "firms": eb_firms,
            "adjusted_r2": float(eb["调整R方"]),
            "ess": eb_ess,
        },
        {
            "specification": "(2) Propensity-Score Matching",
            "variable": psm["解释变量"],
            "coefficient": float(psm["系数"]),
            "std_error": float(psm["标准误"]),
            "p_value": float(psm["p值"]),
            "observations": psm_n,
            "firms": psm_firms,
            "adjusted_r2": float(psm["调整R方"]),
            "ess": "",
        },
    ])
    summary.to_csv(OUT / "gps_binary_matching_table.csv", index=False, encoding="utf-8-sig")

    latex = "\n".join([
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Binary Complements to GPS--IPW}",
        r"    \label{tab:gps-binary-matching}",
        r"    \resizebox{0.9\textwidth}{!}{%",
        r"    \renewcommand{\arraystretch}{1.4}",
        r"    \begin{tabular}{lrr}",
        r"        \toprule",
        r"        & (1) Entropy Balancing & (2) Propensity-Score Matching \\",
        r"        \midrule",
        rf"        \(IMS\) & {coef_cell(float(eb['系数']), float(eb['p值']), float(eb['标准误']))} & {EMPTY} \\",
        rf"        \(IMS_D\) & {EMPTY} & {coef_cell(float(psm['系数']), float(psm['p值']), float(psm['标准误']))} \\",
        r"        Controls & Yes & Yes \\",
        rf"        Observations & {eb_n:,} & {psm_n:,} \\",
        rf"        Adjusted $R^2$ & {float(eb['调整R方']):.4f} & {float(psm['调整R方']):.4f} \\",
        r"        Year fixed effects & Yes & Yes \\",
        r"        Industry fixed effects & Yes & Yes \\",
        r"        \bottomrule",
        r"    \end{tabular}%",
        r"    }",
        r"    \par\smallskip",
        r"    \begin{minipage}{0.95\textwidth}",
        r"        \footnotesize",
        rf"        \textit{{Notes:}} The dependent variable is current Greenwashing. Column (1) reports the entropy-balancing specification and column (2) reports the propensity-score matching specification. Coefficient estimates are reported first, followed by standard errors in parentheses. All models include year and industry fixed effects and use firm-clustered standard errors. Column (1) uses {eb_n:,} firm-years from {eb_firms:,} firms; the effective sample size is {eb_ess:,}. Column (2) uses {psm_n:,} matched firm-years from {psm_firms:,} firms ({psm_pairs:,} pairs). ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.",
        r"    \end{minipage}",
        r"\end{table}",
        "",
    ])
    (OUT / "gps_binary_matching_table.tex").write_text(latex, encoding="utf-8")


if __name__ == "__main__":
    main()
