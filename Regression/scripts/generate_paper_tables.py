from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_tables"
COEF = ROOT / "outputs/baseline/基准回归完整系数.csv"
EXPECTED = int(json.loads((ROOT / "config/regression_config.json").read_text(encoding="utf-8"))["expected_common_sample"])
DESC = OUT / f"descriptive_statistics_{EXPECTED}.csv"
CORR = OUT / "correlation_coefficients.csv"
PVAL = OUT / "correlation_pvalues.csv"

VARIABLES = ["IMS", "IMS_D", "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE", "Dual", "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm", "Intercept"]
LABELS = {**{x: x for x in VARIABLES}, "FirmAge": "FirmAge", "IndependentDirectorRatio": "Indep", "LnAuditFee": "AuditFee", "AuditorTenure": "Tenure", "IsChangeFirm": "AuditorChange"}


def clean_sig(value: object) -> str:
    return value if isinstance(value, str) and value in {"*", "**", "***"} else ""


def cell(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return rf"\shortstack{{{row['系数']:.4f}{clean_sig(row['显著性'])}\\({row['标准误']:.4f})}}"


def regression_table(data: pd.DataFrame, models: list[str], title: str, core: str, label: str) -> str:
    model_headers = "&" + "&".join(f"({index})" for index in range(1, len(models) + 1))
    lines = [
        r"\begin{table}[htbp]",
        rf"\centering\caption{{Audit Firm Industry Specialization and Corporate Greenwashing: {title}}}\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\resizebox{0.87\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}l{'c' * len(models)}@{{}}}}\hline",
        model_headers + r"\\[1pt]\hline",
    ]
    for variable in VARIABLES:
        if variable in {"IMS", "IMS_D"} and variable != core:
            continue
        values = []
        for model in models:
            found = data[(data["模型"] == model) & (data["变量"] == variable)]
            values.append(cell(found.iloc[0] if not found.empty else None))
        lines.append(LABELS[variable].replace("_", "\\_") + " & " + " & ".join(values) + r"\\[4pt]")
    for label_text, field in [("Observations", "样本数"), (r"Adjusted \(R^2\)", "调整R方")]:
        values = []
        for model in models:
            value = data[data["模型"] == model].iloc[0][field]
            values.append(f"{int(value):,}" if field == "样本数" else f"{value:.4f}")
        lines.append(label_text + " & " + " & ".join(values) + r"\\")
    for label_text, field in [("Year fixed effects", "年度固定效应"), ("Industry fixed effects", "行业固定效应")]:
        values = ["Yes" if data[data["模型"] == model].iloc[0][field] == "是" else "No" for model in models]
        lines.append(label_text + " & " + " & ".join(values) + r"\\")
    lines += [
        r"\hline\end{tabular}}",
        r"\vspace{0.4em}",
        r"\begin{minipage}{0.92\textwidth}\footnotesize\textit{Notes:} The dependent variable is continuous Greenwashing. All specifications include year and industry fixed effects and use firm-clustered standard errors. Coefficients are reported above standard errors in parentheses. ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.\end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines)


DISPLAY_NAMES = {
    "Greenwashing": "Greenwashing",
    "ListingAge": "FirmAge",
    "IndependentDirectorRatio": "Indep",
    "IndDirector": "Indep",
    "AuditFee": "AuditFee",
    "Tenure": "Tenure",
    "AuditorChange": "AuditorChange",
}


def correlation_rows(coeff: pd.DataFrame, pvals: pd.DataFrame) -> list[str]:
    names = list(coeff.columns)
    lines = []
    for i, row in enumerate(names):
        cells = []
        for j, col in enumerate(names):
            if i < j:
                cells.append("")
            elif i == j:
                cells.append("1.000")
            else:
                value = f"{coeff.loc[row, col]:.3f}"
                cells.append(rf"\textbf{{{value}}}" if pvals.loc[row, col] < .05 else value)
        display_name = DISPLAY_NAMES.get(row, row).replace("_", r"\_")
        lines.append(f"({i + 1}) {display_name}" + " & " + " & ".join(cells) + r"\\")
    return lines


def main() -> None:
    data = pd.read_csv(COEF, encoding="utf-8-sig")
    models = data["模型"].drop_duplicates().tolist()
    ims = [model for model in models if model.startswith("IMS (")]
    imsd = [model for model in models if model.startswith("IMS_D")]
    (OUT / "baseline_regression_table_ims.tex").write_text(regression_table(data, ims, "IMS", "IMS", "tab:baseline-regressions-ims"), encoding="utf-8")
    (OUT / "baseline_regression_table_ims_d.tex").write_text(regression_table(data, imsd, r"$IMS_D$", "IMS_D", "tab:baseline-regressions-imsd"), encoding="utf-8")
    desc = pd.read_csv(DESC, encoding="utf-8-sig", index_col=0)
    common = pd.read_csv(OUT / f"common_sample_{EXPECTED}.csv", encoding="utf-8-sig")
    lines = []
    source_names = {"Greenwashing": "Greenwashing连续_基准", "ListingAge": "FirmAge", "FirmAge": "FirmAge", "IndDirector": "IndependentDirectorRatio", "Indep": "IndependentDirectorRatio", "AuditFee": "LnAuditFee", "Tenure": "AuditorTenure", "AuditorChange": "IsChangeFirm"}
    display_names = {"Greenwashing": "Greenwashing", "ListingAge": "FirmAge", "FirmAge": "FirmAge", "IndDirector": "Indep", "Indep": "Indep", "AuditFee": "AuditFee", "Tenure": "Tenure", "AuditorChange": "AuditorChange"}
    for variable, row in desc.iterrows():
        source_variable = source_names.get(variable, variable)
        values = pd.to_numeric(common[source_variable], errors="coerce")
        minimum = values.min()
        maximum = values.max()
        display_variable = display_names.get(variable, str(variable)).replace("_", r"\_")
        lines.append(display_variable + f"&{int(row['count']):,}&{row['mean']:.4f}&{row['std']:.4f}&{minimum:.4f}&{row['25%']:.4f}&{row['50%']:.4f}&{row['75%']:.4f}&{maximum:.4f}\\\\")
    (OUT / "descriptive_rows.tex").write_text("\n".join(lines), encoding="utf-8")
    coeff = pd.read_csv(CORR, encoding="utf-8-sig", index_col=0)
    pvals = pd.read_csv(PVAL, encoding="utf-8-sig", index_col=0)
    (OUT / "correlation_rows.tex").write_text("\n".join(correlation_rows(coeff, pvals)), encoding="utf-8")


if __name__ == "__main__":
    main()
