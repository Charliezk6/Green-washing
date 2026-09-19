from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "greenwashing_auditor_regression_panel_2021_2024.csv"
RESULTS = ROOT / "结果"
RESULTS.mkdir(parents=True, exist_ok=True)

ID = "证券代码"
YEAR = "时间"
INDUSTRY = "证监会行业新"
Y_CONT = "Greenwashing连续_基准"
Y_BIN = "Greenwashing"


def winsorize(series, lower=0.01, upper=0.99):
    valid = series.dropna()
    if valid.empty:
        return series
    return series.clip(valid.quantile(lower), valid.quantile(upper))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fit_ols(data, dependent, independent, model_name):
    needed = [dependent, ID, YEAR, INDUSTRY] + independent
    sample = data.dropna(subset=needed).copy()
    formula = f'Q("{dependent}") ~ ' + " + ".join(independent) + f' + C(Q("{YEAR}")) + C(Q("{INDUSTRY}"))'
    result = smf.ols(formula, data=sample).fit(
        cov_type="cluster", cov_kwds={"groups": sample[ID]}
    )
    return result, sample, formula, model_name


def tidy_model(model_tuple):
    result, sample, formula, name = model_tuple
    conf = result.conf_int()
    rows = []
    for term in result.params.index:
        if term.startswith("C(") or term == "Intercept":
            continue
        rows.append({
            "模型": name,
            "变量": term,
            "系数": result.params[term],
            "公司聚类标准误": result.bse[term],
            "t值": result.tvalues[term],
            "p值": result.pvalues[term],
            "置信区间下限": conf.loc[term, 0],
            "置信区间上限": conf.loc[term, 1],
            "显著性": stars(result.pvalues[term]),
            "样本量": int(result.nobs),
            "调整R2": result.rsquared_adj,
            "公司数": sample[ID].nunique(),
            "年份固定效应": "是",
            "行业固定效应": "是",
            "标准误": "公司层面聚类",
            "公式": formula,
        })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT, dtype={ID: str}, encoding="utf-8-sig")
    df[ID] = df[ID].str.extract(r"(\d+)", expand=False).str.zfill(6)

    numeric = [
        YEAR, Y_CONT, Y_BIN, "TotalAssets", "TotalLiabilities", "NetProfit",
        "OperatingNetCashFlow", "S0304a", "SOE", "Dual",
        "IndependentDirectorRatio", "Big4", "FirmAge", "TotalAuditFee"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Auditor industry expertise: client assets audited by an accounting firm as a
    # share of all client assets in the same industry-year. A 10% market-share
    # threshold defines the specialist indicator.
    expertise_base = df.dropna(subset=["AccountingFirmName", INDUSTRY, YEAR, "TotalAssets"]).copy()
    expertise_base = expertise_base[expertise_base["TotalAssets"] > 0]
    firm_assets = expertise_base.groupby(
        [YEAR, INDUSTRY, "AccountingFirmName"], as_index=False
    )["TotalAssets"].sum().rename(columns={"TotalAssets": "AuditorIndustryAssets"})
    industry_assets = expertise_base.groupby(
        [YEAR, INDUSTRY], as_index=False
    )["TotalAssets"].sum().rename(columns={"TotalAssets": "IndustryAssets"})
    expertise = firm_assets.merge(industry_assets, on=[YEAR, INDUSTRY], validate="many_to_one")
    expertise["ExpertiseShare"] = expertise["AuditorIndustryAssets"] / expertise["IndustryAssets"]
    expertise["AuditExpert"] = (expertise["ExpertiseShare"] >= 0.10).astype(int)
    expertise["ExpertiseRank"] = expertise.groupby([YEAR, INDUSTRY])["ExpertiseShare"].rank(
        method="min", ascending=False
    )
    expertise["AuditExpertTop1"] = (expertise["ExpertiseRank"] == 1).astype(int)
    df = df.merge(
        expertise[[YEAR, INDUSTRY, "AccountingFirmName", "ExpertiseShare", "AuditExpert", "AuditExpertTop1"]],
        on=[YEAR, INDUSTRY, "AccountingFirmName"], how="left", validate="many_to_one"
    )

    df["Size"] = np.log(df["TotalAssets"].where(df["TotalAssets"] > 0))
    df["Lev"] = df["TotalLiabilities"] / df["TotalAssets"]
    df["ROA"] = df["NetProfit"] / df["TotalAssets"]
    df["CFO"] = df["OperatingNetCashFlow"] / df["TotalAssets"]
    df["Top1"] = df["S0304a"] / 100
    df["Age"] = np.log1p(df["FirmAge"].where(df["FirmAge"] >= 0))
    df["AuditFee"] = np.log(df["TotalAuditFee"].where(df["TotalAuditFee"] > 0))

    continuous = [Y_CONT, "ExpertiseShare", "Size", "Lev", "ROA", "CFO", "Top1", "IndependentDirectorRatio", "Age", "AuditFee"]
    for col in continuous:
        df[f"{col}_w"] = winsorize(df[col])

    controls = ["Size_w", "Lev_w", "ROA_w", "CFO_w", "Top1_w", "SOE", "Dual", "IndependentDirectorRatio_w", "Age_w", "Big4"]
    models = [
        fit_ols(df, f"{Y_CONT}_w", ["AuditExpert"], "M1 基准：行业专长虚拟变量"),
        fit_ols(df, f"{Y_CONT}_w", ["AuditExpert"] + controls, "M2 加入控制变量"),
        fit_ols(df, f"{Y_CONT}_w", ["ExpertiseShare_w"] + controls, "M3 连续市场份额"),
        fit_ols(df, f"{Y_CONT}_w", ["AuditExpertTop1"] + controls, "M4 行业排名第一"),
        fit_ols(df, Y_BIN, ["AuditExpert"] + controls, "M5 二元漂绿LPM"),
    ]

    tidy = pd.concat([tidy_model(model) for model in models], ignore_index=True)
    tidy.to_csv(RESULTS / "01_first_round_regression_results.csv", index=False, encoding="utf-8-sig")

    focus_terms = ["AuditExpert", "ExpertiseShare_w", "AuditExpertTop1"]
    focus = tidy[tidy["变量"].isin(focus_terms)].copy()
    focus.to_csv(RESULTS / "02_core_results_summary.csv", index=False, encoding="utf-8-sig")

    variables = [f"{Y_CONT}_w", Y_BIN, "AuditExpert", "ExpertiseShare_w"] + controls
    labels = {
        f"{Y_CONT}_w": "连续漂绿", Y_BIN: "二元漂绿", "AuditExpert": "审计行业专长",
        "ExpertiseShare_w": "事务所行业市场份额", "Size_w": "企业规模", "Lev_w": "资产负债率",
        "ROA_w": "总资产收益率", "CFO_w": "经营现金流", "Top1_w": "第一大股东持股",
        "SOE": "国有企业", "Dual": "两职合一", "IndependentDirectorRatio_w": "独董比例",
        "Age_w": "企业年龄", "Big4": "国际四大"
    }
    desc = df[variables].describe(percentiles=[0.25, 0.5, 0.75]).T.reset_index().rename(columns={"index": "变量"})
    desc.insert(1, "中文名称", desc["变量"].map(labels))
    desc.to_csv(RESULTS / "03_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    df[variables].corr().rename(index=labels, columns=labels).to_csv(
        RESULTS / "04_pearson_correlations.csv", encoding="utf-8-sig"
    )

    model_stats = []
    text_parts = []
    for result, sample, formula, name in models:
        model_stats.append({
            "模型": name, "样本量": int(result.nobs), "公司数": sample[ID].nunique(),
            "R2": result.rsquared, "调整R2": result.rsquared_adj,
            "年份固定效应": "是", "行业固定效应": "是", "标准误": "公司层面聚类"
        })
        text_parts.append(f"\n{'=' * 90}\n{name}\n公式: {formula}\n{'=' * 90}\n{result.summary().as_text()}\n")
    pd.DataFrame(model_stats).to_csv(RESULTS / "05_model_statistics.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "06_full_regression_output.txt").write_text("".join(text_parts), encoding="utf-8")

    df.to_csv(RESULTS / "07_analysis_dataset_with_expertise.csv", index=False, encoding="utf-8-sig")
    expertise.to_csv(RESULTS / "08_auditor_industry_expertise_market_share.csv", index=False, encoding="utf-8-sig")

    missing = pd.DataFrame({
        "变量": [Y_CONT, "AuditExpert"] + controls,
        "非缺失数": [df[col].notna().sum() for col in [Y_CONT, "AuditExpert"] + controls],
        "缺失数": [df[col].isna().sum() for col in [Y_CONT, "AuditExpert"] + controls],
    })
    missing["缺失率"] = missing["缺失数"] / len(df)
    missing.to_csv(RESULTS / "09_missingness_report.csv", index=False, encoding="utf-8-sig")

    print(f"Input rows: {len(df):,}; firms: {df[ID].nunique():,}")
    print(f"Expert observations: {int(df['AuditExpert'].sum()):,}; share: {df['AuditExpert'].mean():.2%}")
    print(focus[["模型", "变量", "系数", "公司聚类标准误", "p值", "显著性", "样本量", "调整R2"]].to_string(index=False))
    print(f"Results saved to: {RESULTS}")


if __name__ == "__main__":
    main()
