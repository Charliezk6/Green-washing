from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "paper_tables"
coef = pd.read_csv(OUT / "correlation_coefficients.csv", index_col=0)
pvals = pd.read_csv(OUT / "correlation_pvalues.csv", index_col=0)
names = list(coef.columns)
nums = {name: i + 1 for i, name in enumerate(names)}

lines = [r"\begin{landscape}", r"\begin{table}[p]", r"\centering", r"\caption{Correlation Matrix}", r"\label{tab:correlations}", r"\scriptsize", r"\textit{Panel A: Full-sample pairwise correlations}\\[3pt]", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "r" * len(names) + r"}\hline"]
header_cells = []
for c in names:
    escaped = c.replace("_", r"\_")
    header_cells.append(rf"\rotatebox{{65}}{{({nums[c]}) {escaped}}}")
header = "Variables & " + " & ".join(header_cells) + r"\\\hline"
lines.append(header)
for i, row in enumerate(names):
    cells = []
    for j, col in enumerate(names):
        if i < j:
            cells.append("")
        else:
            value = f"{coef.loc[row, col]:.3f}"
            if i != j and pvals.loc[row, col] < 0.05:
                value = rf"\textbf{{{value}}}"
            cells.append(value)
    label = row.replace("_", r"\_")
    lines.append(f"{label} & " + " & ".join(cells) + r"\\")
lines += [r"\hline\end{tabular}}", r"\begin{minipage}{0.98\textwidth}\footnotesize\textit{Notes:} The table follows the lower-triangular presentation of Reichelt and Wang (2010). Pearson correlations are reported when both variables are continuous; Spearman correlations are reported when either variable is binary. All coefficients use 8,129 observations. Coefficients in bold are significant at the 5 percent level (two-tailed).\end{minipage}", r"\end{table}", r"\end{landscape}"]
(OUT / "correlation_table.tex").write_text("\n".join(lines), encoding="utf-8")

raw = pd.read_csv(ROOT / "data" / "final_reference" / "基准回归完整系数.csv")
models = [m for m in raw["模型"].drop_duplicates() if "HC3" not in m]
ims_cols = [m for m in models if m.startswith("IMS (")]
d_cols = [m for m in models if m.startswith("IMS_D")]
variables = ["IMS", "IMS_D", "Size", "Lev", "ROA", "CFO", "Growth", "FirmAgeLog", "SOE", "Dual", "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm", "Intercept"]
row_labels = {"Intercept": "Intercept", "FirmAgeLog": "FirmAgeLog", "IndependentDirectorRatio": "IndependentDirectorRatio", "LnAuditFee": "LnAuditFee", **{v: v for v in variables}}

def panel(columns, title, core):
    out = [rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}}\\"]
    for variable in variables:
        if variable in {"IMS", "IMS_D"} and variable != core:
            continue
        vals = []
        for model in columns:
            match = raw[(raw["模型"] == model) & (raw["变量"] == variable)]
            if match.empty:
                vals.append("")
            else:
                row = match.iloc[0]
                sig = "" if pd.isna(row["显著性"]) else str(row["显著性"])
                vals.append(rf"\shortstack{{{row['系数']:.4f}{sig} \\ ({row['标准误']:.4f})}}")
        out.append(row_labels[variable].replace("_", r"\_") + " & " + " & ".join(vals) + r"\\[8pt]")
    for label, field in [("Observations", "样本数"), (r"Adjusted \(R^2\)", "调整R方")]:
        vals = []
        for model in columns:
            subset = raw[raw["模型"] == model]
            value = subset.iloc[0][field]
            vals.append(f"{int(value):,}" if field == "样本数" else f"{value:.4f}")
        out.append(label + " & " + " & ".join(vals) + r"\\")
    out += ["Year fixed effects & " + " & ".join("Yes" if raw[raw['模型'] == m].iloc[0]['年度固定效应'] == "是" else "No" for m in columns) + r"\\", "Industry fixed effects & " + " & ".join("Yes" if raw[raw['模型'] == m].iloc[0]['行业固定效应'] == "是" else "No" for m in columns) + r"\\"]
    return out

def regression_table(columns, title, core, label):
    lines = [r"\begin{table}[htbp]", rf"\centering\caption{{Audit Firm Industry Specialization and Corporate Greenwashing: {title}}}\label{{{label}}}\scriptsize", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}\hline", r"&(1)&(2)&(3)&(4)&(5)&(6)\\\hline"]
    lines += panel(columns, title, core)
    lines += [r"\hline\end{tabular}}", r"\begin{minipage}{0.98\textwidth}\footnotesize\textit{Notes:} The dependent variable is continuous Greenwashing. Columns (1)--(5) use firm-clustered standard errors; column (6) uses industry-clustered standard errors. Coefficients are reported above standard errors in parentheses. ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.\end{minipage}", r"\end{table}"]
    return lines

lines = regression_table(ims_cols, "Panel A: IMS", "IMS", "tab:baseline-regressions-ims") + [""] + regression_table(d_cols, r"Panel B: $IMS_D$", "IMS_D", "tab:baseline-regressions-imsd")
(OUT / "baseline_regression_table.tex").write_text("\n".join(lines), encoding="utf-8")
