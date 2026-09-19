import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/paper_tables"
EXPECTED = int(json.loads((ROOT / "config/regression_config.json").read_text(encoding="utf-8"))["expected_common_sample"])
coef = pd.read_csv(OUT / "correlation_coefficients.csv", index_col=0)
pvals = pd.read_csv(OUT / "correlation_pvalues.csv", index_col=0)
names = list(coef.columns)
lines = [r"\begin{table}[p]", r"\centering", r"\caption{Correlation Matrix: Combined Sample}", r"\scriptsize", r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l" + "r" * len(names) + r"}\hline"]
header_names = [f"({i + 1}) {name.replace('_', chr(92) + '_')}" for i, name in enumerate(names)]
lines.append("Variables & " + " & ".join(header_names) + r"\\\hline")
for i, row in enumerate(names):
    cells = []
    for j, col in enumerate(names):
        if i < j: cells.append("")
        else:
            value = f"{coef.loc[row, col]:.3f}"
            if i != j and pvals.loc[row, col] < .05: value = rf"\textbf{{{value}}}"
            cells.append(value)
    lines.append(row.replace("_", r"\_") + " & " + " & ".join(cells) + r"\\")
lines += [r"\hline\end{tabular}}", rf"\begin{{minipage}}{{0.98\textwidth}}\footnotesize Notes: Pearson correlations are reported for two continuous variables and Spearman correlations when either variable is binary. The combined common sample has {EXPECTED:,} observations. Bold coefficients are significant at 5 percent.\end{{minipage}}", r"\end{table}"]
(OUT / "correlation_table.tex").write_text("\n".join(lines), encoding="utf-8")

raw = pd.read_csv(ROOT / "outputs/baseline/基准回归完整系数.csv", encoding="utf-8-sig")
models = [m for m in raw["模型"].drop_duplicates() if "HC3" not in m]
variables = ["IMS", "IMS_D", "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE", "Dual", "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm", "Intercept"]
def table(core: str, path: Path):
    cols = [m for m in models if m.startswith(core + " ")]
    text = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{Combined-sample baseline regression: {core}}}", r"\scriptsize", r"\begin{tabular}{l" + "r" * len(cols) + r"}\hline"]
    text.append(" & " + " & ".join(f"({i + 1})" for i in range(len(cols))) + r"\\\hline")
    for var in variables:
        if var in {"IMS", "IMS_D"} and var != core: continue
        vals=[]
        for model in cols:
            z=raw[(raw["模型"]==model)&(raw["变量"]==var)]
            vals.append("" if z.empty else f"{z.iloc[0]['系数']:.4f}{z.iloc[0]['显著性']} ({z.iloc[0]['标准误']:.4f})")
        text.append(var.replace("_",r"\_")+" & "+" & ".join(vals)+r"\\")
    text += [r"\hline\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(text),encoding="utf-8")
table("IMS", OUT / "baseline_regression_table_ims.tex")
table("IMS_D", OUT / "baseline_regression_table_ims_d.tex")
