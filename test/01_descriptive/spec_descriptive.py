from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spec_common import CORE, FIRM, INDUSTRY, YEAR, common_sample, rr  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
DEP = rr.DEPENDENT


def describe(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return {
        "N": int(values.size), "Mean": values.mean(), "SD": values.std(), "Min": values.min(),
        "P25": values.quantile(.25), "Median": values.median(), "P75": values.quantile(.75), "Max": values.max(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common, _ = common_sample()
    common.to_csv(OUT / "common_sample_15381_with_spec.csv", index=False, encoding="utf-8-sig")

    overall = pd.DataFrame([
        {"Variable": "SPEC", **describe(common[CORE])},
        {"Variable": "SPEC_LOO", **describe(common["SPEC_LOO"])},
        {"Variable": "FeeShare", **describe(common["FeeShare"])},
        {"Variable": "IMS_D", **describe(common["IMS_D"])},
        {"Variable": "IMS", **describe(common["IMS"])},
    ])
    overall["SPEC=1 count"] = [int(common[CORE].sum()), int(common["SPEC_LOO"].sum()), None, int(common["IMS_D"].sum()), None]
    overall.to_csv(OUT / "spec_descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    by_year = common.groupby(YEAR).agg(N=(CORE, "size"), SPEC_mean=(CORE, "mean"), SPEC_count=(CORE, "sum"), IMS_D_mean=("IMS_D", "mean")).reset_index()
    by_year.to_csv(OUT / "spec_by_year.csv", index=False, encoding="utf-8-sig")
    by_industry = common.groupby(INDUSTRY).agg(N=(CORE, "size"), SPEC_mean=(CORE, "mean"), SPEC_count=(CORE, "sum"), IMS_D_mean=("IMS_D", "mean")).reset_index()
    by_industry.to_csv(OUT / "spec_by_industry.csv", index=False, encoding="utf-8-sig")

    crosstab = pd.crosstab(common[CORE].astype(int).rename("SPEC"), common["IMS_D"].astype(int).rename("IMS_D"), margins=True, margins_name="Total")
    crosstab.to_csv(OUT / "spec_vs_ims_d_crosstab.csv", encoding="utf-8-sig")

    corr_vars = [CORE, "FeeShare", "IMS", "IMS_D", DEP]
    pearson = common[corr_vars].corr()
    spearman = common[corr_vars].corr(method="spearman")
    pearson.to_csv(OUT / "spec_correlation_pearson.csv", encoding="utf-8-sig")
    spearman.to_csv(OUT / "spec_correlation_spearman.csv", encoding="utf-8-sig")

    treated = common.loc[common[CORE].eq(1), DEP]
    control = common.loc[common[CORE].eq(0), DEP]
    welch = stats.ttest_ind(treated, control, equal_var=False)
    mann = stats.mannwhitneyu(treated, control)
    rows = [{
        "Variable": DEP, "SPEC=1 N": len(treated), "SPEC=1 mean": treated.mean(), "SPEC=0 N": len(control), "SPEC=0 mean": control.mean(),
        "Difference": treated.mean() - control.mean(), "Welch t": welch.statistic, "Welch p": welch.pvalue,
        "SPEC=1 median": treated.median(), "SPEC=0 median": control.median(), "Mann-Whitney p": mann.pvalue,
    }]
    for control_var in rr.ALL_CONTROLS:
        a = common.loc[common[CORE].eq(1), control_var]
        b = common.loc[common[CORE].eq(0), control_var]
        test = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"Variable": control_var, "SPEC=1 N": len(a), "SPEC=1 mean": a.mean(), "SPEC=0 N": len(b), "SPEC=0 mean": b.mean(), "Difference": a.mean() - b.mean(), "Welch t": test.statistic, "Welch p": test.pvalue})
    pd.DataFrame(rows).to_csv(OUT / "spec_group_mean_difference.csv", index=False, encoding="utf-8-sig")

    auditors = common[common[CORE].eq(1)].groupby(["SPEC_AuditorID"]).agg(
        AccountingFirmName=("AccountingFirmName", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        SPEC_observations=(CORE, "size"), industries=(INDUSTRY, "nunique"), years=(YEAR, "nunique"), firms=(FIRM, "nunique"),
    ).sort_values("SPEC_observations", ascending=False).reset_index()
    auditors.to_csv(OUT / "spec_leader_auditors.csv", index=False, encoding="utf-8-sig")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering\caption{Descriptive Statistics of SPEC}\label{tab:spec-descriptive}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}\hline",
        r"Variable & N & Mean & SD & Min & P25 & Median & P75 & Max\\\hline",
    ]
    for _, row in overall.iterrows():
        name = row["Variable"].replace("_", r"\_")
        lines.append(f"{name}&{int(row['N']):,}&{row['Mean']:.4f}&{row['SD']:.4f}&{row['Min']:.4f}&{row['P25']:.4f}&{row['Median']:.4f}&{row['P75']:.4f}&{row['Max']:.4f}\\\\")
    lines += [r"\hline", r"\multicolumn{9}{@{}l}{\textit{SPEC by year}}\\"]
    for _, row in by_year.iterrows():
        lines.append(f"{int(row[YEAR])}&{int(row['N']):,}&{row['SPEC_mean']:.4f}&\\multicolumn{{6}}{{l}}{{SPEC=1: {int(row['SPEC_count']):,}}}\\\\")
    lines += [
        r"\hline\end{tabular}",
        r"\vspace{0.4em}",
        r"\begin{minipage}{0.92\textwidth}\footnotesize\textit{Notes:} SPEC equals 1 if the client's audit firm has the largest audit-fee market share in the client's industry-year (full A-share market, 2021--2024; manufacturing uses two-digit CSRC codes, other sectors one-letter codes). Sample is the 15,381 firm-year common sample.\end{minipage}",
        r"\end{table}",
    ]
    (OUT / "spec_descriptive_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print(overall.to_string())
    print(by_year.to_string())
    print(crosstab.to_string())
    print(pd.DataFrame(rows).head(1).T.to_string())


if __name__ == "__main__":
    main()
