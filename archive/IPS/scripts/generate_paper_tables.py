from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COEF = ROOT / "outputs/baseline/ips_regression_coefficients.csv"
OUT = ROOT / "outputs/paper_tables"
VARIABLES = ["IPS", "IPS_D", "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE", "Dual", "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm", "Intercept"]
LABELS = {**{variable: variable for variable in VARIABLES}, "IndependentDirectorRatio": "Indep", "LnAuditFee": "AuditFee", "AuditorTenure": "Tenure", "IsChangeFirm": "AuditorChange"}


def clean_sig(value: object) -> str:
    return value if isinstance(value, str) and value in {"*", "**", "***"} else ""


def cell(row: pd.Series | None) -> str:
    if row is None:
        return ""
    return rf"\shortstack{{{row['系数']:.4f}{clean_sig(row['显著性'])} \\ ({row['标准误']:.4f})}}"


def regression_table(data: pd.DataFrame, models: list[str], title: str, core: str, label: str) -> str:
    column_count = len(models) + 1
    model_headers = "&" + "&".join(f"({index})" for index in range(1, len(models) + 1))
    lines = [
        r"\begin{table}[htbp]",
        rf"\centering\caption{{Audit Firm Industry Specialization and Corporate Greenwashing: {title}}}\label{{{label}}}\scriptsize",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{'r' * len(models)}}}\hline",
        model_headers + r"\\\hline",
        rf"\multicolumn{{{column_count}}}{{l}}{{\textit{{{title}}}}}\\",
    ]
    for variable in VARIABLES:
        if variable in {"IPS", "IPS_D"} and variable != core:
            continue
        values = []
        for model in models:
            found = data[(data["模型"] == model) & (data["变量"] == variable)]
            values.append(cell(found.iloc[0] if not found.empty else None))
        lines.append(LABELS[variable].replace("_", "\\_") + " & " + " & ".join(values) + r"\\[8pt]")
    for label_text, field in [("Observations", "样本数"), (r"Adjusted \(R^2\)", "调整R方")]:
        values = []
        for model in models:
            value = data[data["模型"] == model].iloc[0][field]
            values.append(f"{int(value):,}" if field == "样本数" else f"{value:.4f}")
        lines.append(label_text + " & " + " & ".join(values) + r"\\")
    for label_text, field in [("Year fixed effects", "年度固定效应"), ("Industry fixed effects", "行业固定效应")]:
        values = ["Yes" if data[data["模型"] == model].iloc[0][field] == "是" else "No" for model in models]
        lines.append(label_text + " & " + " & ".join(values) + r"\\")
    lines += [r"\hline\end{tabular}}", r"\begin{minipage}{0.98\textwidth}\footnotesize\textit{Notes:} The dependent variable is continuous Greenwashing. All specifications include year and industry fixed effects and use firm-clustered standard errors. Coefficients are reported above standard errors in parentheses. ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.\end{minipage}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(COEF, encoding="utf-8-sig")
    models = data["模型"].drop_duplicates().tolist()
    ips = [model for model in models if model.startswith("IPS (")]
    ips_d = [model for model in models if model.startswith("IPS_D")]
    (OUT / "ips_regression_table.tex").write_text(regression_table(data, ips, "Panel A: IPS", "IPS", "tab:ips-regressions"), encoding="utf-8")
    (OUT / "ips_d_regression_table.tex").write_text(regression_table(data, ips_d, r"Panel B: $IPS_D$", "IPS_D", "tab:ipsd-regressions"), encoding="utf-8")


if __name__ == "__main__":
    main()
