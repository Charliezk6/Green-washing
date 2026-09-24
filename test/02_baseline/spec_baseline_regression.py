from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec_common import CORE, FIRM, INDUSTRY, YEAR, common_sample, rr  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
rr.LABELS["SPEC"] = "SPEC"
VARIABLES = ["SPEC", "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE", "Dual", "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm", "Intercept"]
DISPLAY = {"IndependentDirectorRatio": "Indep", "LnAuditFee": "AuditFee", "AuditorTenure": "Tenure", "IsChangeFirm": "AuditorChange"}


def cell(row: pd.Series | None) -> str:
    if row is None:
        return ""
    sig = row["显著性"] if isinstance(row["显著性"], str) else ""
    return rf"\shortstack{{{row['系数']:.4f}{sig}\\({row['标准误']:.4f})}}"


def latex_table(data: pd.DataFrame, models: list[str]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering\caption{Audit Firm Industry Specialization and Corporate Greenwashing: SPEC}\label{tab:baseline-regressions-spec}",
        r"\footnotesize", r"\setlength{\tabcolsep}{6pt}", r"\renewcommand{\arraystretch}{1.12}", r"\resizebox{0.87\textwidth}{!}{%",
        rf"\begin{{tabular}}{{@{{}}l{'c' * len(models)}@{{}}}}\hline",
        "&" + "&".join(f"({i})" for i in range(1, len(models) + 1)) + r"\\[1pt]\hline",
    ]
    for variable in VARIABLES:
        values = []
        for model in models:
            found = data[(data["模型"] == model) & (data["变量"] == variable)]
            values.append(cell(found.iloc[0] if not found.empty else None))
        lines.append(DISPLAY.get(variable, variable) + " & " + " & ".join(values) + r"\\[4pt]")
    first = [data[data["模型"] == m].iloc[0] for m in models]
    lines.append("Observations & " + " & ".join(f"{int(r['样本数']):,}" for r in first) + r"\\")
    lines.append(r"Adjusted \(R^2\) & " + " & ".join(f"{r['调整R方']:.4f}" for r in first) + r"\\")
    lines.append("Year fixed effects & " + " & ".join("Yes" for _ in first) + r"\\")
    lines.append("Industry fixed effects & " + " & ".join("Yes" for _ in first) + r"\\")
    lines += [
        r"\hline\end{tabular}}", r"\vspace{0.4em}",
        r"\begin{minipage}{0.92\textwidth}\footnotesize\textit{Notes:} The dependent variable is continuous Greenwashing. SPEC equals 1 if the client's audit firm has the largest audit-fee market share in the client's industry-year. All specifications include year and industry fixed effects and use firm-clustered standard errors. Coefficients are reported above standard errors in parentheses. ***, **, and * indicate two-tailed significance at 1, 5, and 10 percent.\end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common, winsorization = common_sample()
    winsorization.to_csv(OUT / "连续控制变量缩尾边界.csv", index=False, encoding="utf-8-sig")
    specs = [("(1) 一元回归", []), ("(2) 财务控制", rr.FINANCIAL), ("(3) 财务与治理控制", rr.FINANCIAL + rr.GOVERNANCE), ("(4) 全部控制", rr.ALL_CONTROLS)]
    tables, samples = [], []
    for name, controls in specs:
        result, sample = rr.fit(common, CORE, controls, True, "cluster")
        full_name = f"{CORE} {name}"
        tables.append(rr.tidy(result, full_name, CORE, "公司聚类", True))
        samples.append({"模型": full_name, "核心变量": CORE, "样本数": len(sample), "公司数": sample[FIRM].nunique(), "年度数": sample[YEAR].nunique(), "行业数": sample[INDUSTRY].nunique(), "SPEC=1观测数": int(sample[CORE].sum())})
    coefficients = pd.concat(tables, ignore_index=True)
    coefficients.to_csv(OUT / "SPEC基准回归完整系数.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(OUT / "SPEC基准回归样本信息.csv", index=False, encoding="utf-8-sig")
    models = coefficients["模型"].drop_duplicates().tolist()
    (OUT / "baseline_regression_table_spec.tex").write_text(latex_table(coefficients, models), encoding="utf-8")
    print(coefficients[coefficients["变量"] == CORE][["模型", "系数", "标准误", "p值", "显著性", "样本数", "调整R方"]].to_string())


if __name__ == "__main__":
    main()
