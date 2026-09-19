"""Write paper-ready CSV and LaTeX tables for the scope-condition tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = Path(__file__).resolve().parents[1] / "results/tables"
COEF = TABLES / "heterogeneity_coefficients.csv"


def stars(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def cell(df: pd.DataFrame, model: str, variable: str) -> str:
    match = df[(df["model"] == model) & (df["variable"] == variable)]
    if match.empty:
        return ""
    row = match.iloc[0]
    return rf"\shortstack{{{row['coefficient']:.4f}{stars(row['p_value'])} \\ ({row['std_error']:.4f})}}"


def n_cell(df: pd.DataFrame, model: str) -> str:
    match = df[df["model"] == model]
    return f"{int(match.iloc[0]['n']):,}" if not match.empty else ""


def r2_cell(df: pd.DataFrame, model: str) -> str:
    match = df[df["model"] == model]
    return f"{match.iloc[0]['adjusted_r2']:.4f}" if not match.empty else ""


def main() -> None:
    df = pd.read_csv(COEF)
    main_models = [
        ("IMS x AuditIntersect_Share", "IMS:AuditIntersect_Share", r"\(AuditIntersect\_Share\)"),
        ("IMS x HighIntersect", "IMS:HighIntersect", r"\(HighIntersect\)"),
        ("IMS x KeyPollMonUnit", "IMS:KeyPollMonUnit", r"\(KeyPollMonUnit\)"),
        ("IMS x Penalty_D", "IMS:Penalty_D", r"\(Penalty\_D\)"),
        ("IMS x Soft_Share placebo", "IMS:Soft_Share", r"\(Soft\_Share\)"),
    ]
    headers = " & ".join(f"({i})" for i in range(1, 6))
    ims_cells = [cell(df, model, "IMS") for model, _, _ in main_models]
    v_cells = [
        cell(df, "IMS x AuditIntersect_Share", "AuditIntersect_Share"),
        cell(df, "IMS x HighIntersect", "HighIntersect"),
        cell(df, "IMS x KeyPollMonUnit", "KeyPollMonUnit"),
        cell(df, "IMS x Penalty_D", "Penalty_D"),
        cell(df, "IMS x Soft_Share placebo", "Soft_Share"),
    ]
    inter_cells = [cell(df, model, inter) for model, inter, _ in main_models]
    v_names = " / ".join(name for _, _, name in main_models)
    latex = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"    \centering",
            r"    \caption{Scope Conditions Consistent with an Other-Information Channel}",
            r"    \label{tab:scope-conditions}",
            r"    \resizebox{\textwidth}{!}{%",
            r"    \renewcommand{\arraystretch}{1.35}",
            r"    \begin{tabular}{lccccc}",
            r"        \toprule",
            rf"        & {headers} \\",
            r"        \midrule",
            rf"        \(IMS\) & {' & '.join(ims_cells)} \\",
            rf"        Moderator \(V\) & {' & '.join(v_cells)} \\",
            rf"        \(IMS \times V\) & {' & '.join(inter_cells)} \\",
            r"        Controls & Yes & Yes & Yes & Yes & Yes \\",
            rf"        Observations & {' & '.join(n_cell(df, m) for m, _, _ in main_models)} \\",
            rf"        Adjusted $R^2$ & {' & '.join(r2_cell(df, m) for m, _, _ in main_models)} \\",
            r"        Year fixed effects & Yes & Yes & Yes & Yes & Yes \\",
            r"        Industry fixed effects & Yes & Yes & Yes & Yes & Yes \\",
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"    \par\smallskip",
            r"    \begin{minipage}{0.98\textwidth}",
            r"        \footnotesize",
            rf"        \textit{{Notes:}} The dependent variable is continuous Greenwashing. The moderator \(V\) is {v_names} in columns (1)--(5). Column (5) is a placebo based on symbolic environmental language. All specifications include the baseline financial, governance, and audit controls, year and industry fixed effects, and firm-clustered standard errors. Coefficient estimates are reported first, followed by standard errors in parentheses. ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.",
            r"    \end{minipage}",
            r"\end{table}",
        ]
    )
    (TABLES / "scope_conditions_table.tex").write_text(latex, encoding="utf-8")

    sub_models = [
        ("IMS subsample HighIntersect=1", "High \(AuditIntersect\)"),
        ("IMS subsample HighIntersect=0", "Low \(AuditIntersect\)"),
        ("IMS subsample KeyPollMonUnit=1", "Key pollution unit"),
        ("IMS subsample KeyPollMonUnit=0", "Not key pollution unit"),
    ]
    sub_latex = "\n".join(
        [
            r"\begin{table}[htbp]",
            r"    \centering",
            r"    \caption{Split-Sample Associations by Auditable Environmental Context}",
            r"    \label{tab:scope-splits}",
            r"    \renewcommand{\arraystretch}{1.35}",
            r"    \begin{tabular}{lcccc}",
            r"        \toprule",
            rf"        & {' & '.join(f'({i})' for i in range(1, 5))} \\",
            r"        \midrule",
            rf"        Sample & {' & '.join(label for _, label in sub_models)} \\",
            rf"        \(IMS\) & {' & '.join(cell(df, model, 'IMS') for model, _ in sub_models)} \\",
            r"        Controls & Yes & Yes & Yes & Yes \\",
            rf"        Observations & {' & '.join(n_cell(df, model) for model, _ in sub_models)} \\",
            rf"        Adjusted $R^2$ & {' & '.join(r2_cell(df, model) for model, _ in sub_models)} \\",
            r"        Year fixed effects & Yes & Yes & Yes & Yes \\",
            r"        Industry fixed effects & Yes & Yes & Yes & Yes \\",
            r"        \bottomrule",
            r"    \end{tabular}",
            r"    \par\smallskip",
            r"    \begin{minipage}{0.95\textwidth}",
            r"        \footnotesize",
            r"        \textit{Notes:} Each column re-estimates the baseline model in the indicated subsample. High \(AuditIntersect\) equals one when the firm-year share of environmental statements with auditable anchors exceeds the industry-year median. Key pollution unit equals one when the firm discloses status as a key pollution-monitoring unit. These splits are for display; inference on differences uses the interaction specifications. Firm-clustered standard errors are in parentheses. ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.",
            r"    \end{minipage}",
            r"\end{table}",
        ]
    )
    (TABLES / "scope_splits_table.tex").write_text(sub_latex, encoding="utf-8")

    df.to_csv(TABLES / "heterogeneity_coefficients.csv", index=False, encoding="utf-8-sig")
    print("wrote", TABLES / "scope_conditions_table.tex")
    print("wrote", TABLES / "scope_splits_table.tex")


if __name__ == "__main__":
    main()
