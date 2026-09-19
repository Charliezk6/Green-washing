from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "pre_regression_test" / "greenwashing_auditor_expertise_panel.csv"
FINANCIAL_FILE = (
    PROJECT_DIR
    / "appendix"
    / "上市公司财务指标数据表(年)150744996(仅供南京审计大学使用)"
    / "AIQ_LCFinIndexY.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "回归结果" / "基准回归_8129共同样本"
DEPENDENT = "Greenwashing连续_基准"
CORE_VARIABLES = ["IMS", "IMS_D"]

FINANCIAL_CONTROLS = ["Size", "Lev", "ROA", "CFO", "Growth", "FirmAgeLog"]
GOVERNANCE_CONTROLS = ["SOE", "Dual", "BoardSize", "IndependentDirectorRatio"]
AUDIT_CONTROLS = ["Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm"]
ALL_CONTROLS = FINANCIAL_CONTROLS + GOVERNANCE_CONTROLS + AUDIT_CONTROLS

LABELS = {
    "IMS": "IMS",
    "IMS_D": "IMS_D",
    "Size": "公司规模",
    "Lev": "资产负债率",
    "ROA": "总资产收益率",
    "CFO": "经营现金流",
    "Growth": "营业收入增长率",
    "FirmAgeLog": "公司年龄",
    "SOE": "产权性质",
    "Dual": "两职合一",
    "BoardSize": "董事会规模",
    "IndependentDirectorRatio": "独立董事比例",
    "Big4": "国际四大",
    "DomesticTop10Flag": "国内十大",
    "LnAuditFee": "审计收费",
    "AuditorTenure": "审计任期",
    "IsChangeFirm": "事务所变更",
    "Intercept": "常数项",
}


def significance(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def load_full_market_growth() -> pd.DataFrame:
    financial = pd.read_csv(
        FINANCIAL_FILE, dtype={"Symbol": str}, encoding="utf-8-sig"
    )
    financial["时间"] = pd.to_datetime(
        financial["EndDate"], errors="coerce"
    ).dt.year
    financial["OperatingRevenue_FullMarket"] = pd.to_numeric(
        financial["OperatingRevenue"], errors="coerce"
    )
    financial = financial.loc[
        financial["时间"].between(2020, 2024),
        ["Symbol", "时间", "OperatingRevenue_FullMarket"],
    ].drop_duplicates(["Symbol", "时间"], keep="first")
    financial = financial.sort_values(["Symbol", "时间"])
    financial["LagRevenue_FullMarket"] = financial.groupby("Symbol")[
        "OperatingRevenue_FullMarket"
    ].shift(1)
    previous_year = financial.groupby("Symbol")["时间"].shift(1)
    financial.loc[
        financial["时间"].sub(previous_year).ne(1), "LagRevenue_FullMarket"
    ] = np.nan
    financial["Growth"] = (
        financial["OperatingRevenue_FullMarket"]
        / financial["LagRevenue_FullMarket"]
        - 1
    )
    financial.loc[
        financial["LagRevenue_FullMarket"].le(0), "Growth"
    ] = np.nan
    financial.replace([np.inf, -np.inf], np.nan, inplace=True)
    return financial.rename(columns={"Symbol": "证券代码"})


def construct_variables(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    numeric_columns = [
        DEPENDENT, "IMS", "IMS_D", "TotalAssets", "TotalLiabilities",
        "OperatingRevenue", "NetProfit", "OperatingNetCashFlow", "FirmAge",
        "SOE", "Dual", "DirectorNumber", "IndependentDirectorRatio", "Big4",
        "DomesticTop10Flag", "TotalAuditFee", "AuditorTenure", "IsChangeFirm",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Size"] = np.log(data["TotalAssets"].where(data["TotalAssets"].gt(0)))
    data["Lev"] = data["TotalLiabilities"] / data["TotalAssets"]
    data["ROA"] = data["NetProfit"] / data["TotalAssets"]
    data["CFO"] = data["OperatingNetCashFlow"] / data["TotalAssets"]
    data["FirmAgeLog"] = np.log1p(data["FirmAge"].where(data["FirmAge"].ge(0)))
    data["BoardSize"] = np.log(data["DirectorNumber"].where(data["DirectorNumber"].gt(0)))
    data["LnAuditFee"] = np.log(data["TotalAuditFee"].where(data["TotalAuditFee"].gt(0)))
    growth = load_full_market_growth()
    data = data.merge(
        growth[
            [
                "证券代码",
                "时间",
                "OperatingRevenue_FullMarket",
                "LagRevenue_FullMarket",
                "Growth",
            ]
        ],
        on=["证券代码", "时间"],
        how="left",
        validate="many_to_one",
    )
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.sort_values(["证券代码", "时间"])


def formula(core: str, controls: list[str], fixed_effects: bool) -> str:
    terms = [core] + controls
    if fixed_effects:
        terms += ["C(时间)", "C(ExpertiseIndustryCode)"]
    return f"Q('{DEPENDENT}') ~ " + " + ".join(terms)


def fit_model(data: pd.DataFrame, core: str, controls: list[str], fixed_effects: bool,
              covariance: str):
    required = [DEPENDENT, core, "证券代码"] + controls
    if fixed_effects:
        required += ["时间", "ExpertiseIndustryCode"]
    sample = data.dropna(subset=required).copy()
    model = smf.ols(formula(core, controls, fixed_effects), data=sample)
    if covariance == "cluster":
        result = model.fit(cov_type="cluster", cov_kwds={"groups": sample["证券代码"]})
    elif covariance == "industry_cluster":
        result = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": sample["ExpertiseIndustryCode"]},
        )
    elif covariance == "HC3":
        result = model.fit(cov_type="HC3")
    else:
        result = model.fit()
    return result, sample


def tidy_result(result, model_name: str, core: str, covariance: str,
                fixed_effects: bool) -> pd.DataFrame:
    rows = []
    for variable in ["Intercept", core] + ALL_CONTROLS:
        if variable not in result.params.index:
            continue
        rows.append({
            "模型": model_name,
            "核心变量": core,
            "变量": variable,
            "变量名称": LABELS.get(variable, variable),
            "系数": result.params[variable],
            "标准误": result.bse[variable],
            "t值": result.tvalues[variable],
            "p值": result.pvalues[variable],
            "显著性": significance(result.pvalues[variable]),
            "置信区间下限": result.conf_int().loc[variable, 0],
            "置信区间上限": result.conf_int().loc[variable, 1],
            "标准误类型": covariance,
            "年度固定效应": "是" if fixed_effects else "否",
            "行业固定效应": "是" if fixed_effects else "否",
            "样本数": int(result.nobs),
            "调整R方": result.rsquared_adj,
        })
    return pd.DataFrame(rows)


def publication_table(results: list[dict]) -> pd.DataFrame:
    row_order = ["IMS", "IMS_D"] + ALL_CONTROLS + ["Intercept"]
    table = pd.DataFrame(index=[LABELS[item] for item in row_order] + [
        "样本数", "调整R方", "年度固定效应", "行业固定效应", "标准误"
    ])
    for item in results:
        result = item["result"]
        column = item["name"]
        values = {}
        for variable in row_order:
            if variable in result.params.index:
                stars = significance(result.pvalues[variable])
                values[LABELS[variable]] = f"{result.params[variable]:.4f}{stars}\n({result.bse[variable]:.4f})"
        values.update({
            "样本数": f"{int(result.nobs)}",
            "调整R方": f"{result.rsquared_adj:.4f}",
            "年度固定效应": "是" if item["fe"] else "否",
            "行业固定效应": "是" if item["fe"] else "否",
            "标准误": {
                "cluster": "公司聚类",
                "industry_cluster": "行业聚类",
                "HC3": "HC3稳健",
            }[item["cov"]],
        })
        table[column] = pd.Series(values)
    return table.fillna("")


def main() -> None:
    raw = pd.read_csv(INPUT_FILE, dtype={"证券代码": str}, encoding="utf-8-sig")
    data = construct_variables(raw)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    specifications = [
        ("(1) 一元回归", [], False),
        ("(2) 财务控制", FINANCIAL_CONTROLS, False),
        ("(3) 财务与治理控制", FINANCIAL_CONTROLS + GOVERNANCE_CONTROLS, False),
        ("(4) 全部控制", ALL_CONTROLS, False),
        ("(5) 全部控制+双向固定效应", ALL_CONTROLS, True),
    ]
    common_required = [
        DEPENDENT,
        *CORE_VARIABLES,
        *ALL_CONTROLS,
        "证券代码",
        "时间",
        "ExpertiseIndustryCode",
    ]
    common_sample = data.dropna(subset=common_required).copy()

    fitted = []
    tidy_tables = []
    sample_rows = []
    for core in CORE_VARIABLES:
        for model_name, controls, fixed_effects in specifications:
            result, sample = fit_model(
                common_sample, core, controls, fixed_effects, "cluster"
            )
            display_name = f"{core} {model_name}"
            fitted.append({"name": display_name, "result": result, "core": core,
                           "fe": fixed_effects, "cov": "cluster"})
            tidy_tables.append(tidy_result(result, display_name, core, "公司聚类", fixed_effects))
            sample_rows.append({
                "模型": display_name, "核心变量": core, "样本数": len(sample),
                "公司数": sample["证券代码"].nunique(), "年度数": sample["时间"].nunique(),
                "行业数": sample["ExpertiseIndustryCode"].nunique(),
            })

        hc3_result, _ = fit_model(
            common_sample, core, ALL_CONTROLS, True, "HC3"
        )
        hc3_name = f"{core} (6) 全部控制+双向固定效应_HC3"
        fitted.append({"name": hc3_name, "result": hc3_result, "core": core,
                       "fe": True, "cov": "HC3"})
        tidy_tables.append(tidy_result(hc3_result, hc3_name, core, "HC3稳健", True))

        industry_result, industry_sample = fit_model(
            common_sample, core, ALL_CONTROLS, True, "industry_cluster"
        )
        industry_name = f"{core} (7) 全部控制+双向固定效应_行业聚类"
        fitted.append({
            "name": industry_name,
            "result": industry_result,
            "core": core,
            "fe": True,
            "cov": "industry_cluster",
        })
        tidy_tables.append(
            tidy_result(
                industry_result, industry_name, core, "行业聚类", True
            )
        )
        sample_rows.append({
            "模型": industry_name,
            "核心变量": core,
            "样本数": len(industry_sample),
            "公司数": industry_sample["证券代码"].nunique(),
            "年度数": industry_sample["时间"].nunique(),
            "行业数": industry_sample["ExpertiseIndustryCode"].nunique(),
        })

    tidy = pd.concat(tidy_tables, ignore_index=True)
    samples = pd.DataFrame(sample_rows)
    pub = publication_table(fitted)
    variable_diagnostics = data[[DEPENDENT] + CORE_VARIABLES + ALL_CONTROLS].describe().T
    missing = data[[DEPENDENT] + CORE_VARIABLES + ALL_CONTROLS].isna().sum().rename("缺失数").to_frame()
    diagnostics = variable_diagnostics.join(missing)

    tidy.to_csv(OUTPUT_DIR / "基准回归完整系数.csv", index=False, encoding="utf-8-sig")
    pub.to_csv(OUTPUT_DIR / "基准回归论文格式表.csv", encoding="utf-8-sig")
    samples.to_csv(OUTPUT_DIR / "基准回归样本信息.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(OUTPUT_DIR / "回归变量诊断.csv", encoding="utf-8-sig")
    data.to_csv(OUTPUT_DIR / "已构造基准回归变量的面板.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_DIR / "基准回归结果.xlsx", engine="openpyxl") as writer:
        pub.to_excel(writer, sheet_name="论文格式")
        tidy.to_excel(writer, sheet_name="完整系数", index=False)
        samples.to_excel(writer, sheet_name="样本信息", index=False)
        diagnostics.to_excel(writer, sheet_name="变量诊断")

    print(f"原始观测数：{len(raw)}")
    print(f"公司数：{raw['证券代码'].nunique()}")
    for core in CORE_VARIABLES:
        final = next(item for item in fitted if item['core'] == core and item['name'].endswith('(5) 全部控制+双向固定效应'))
        result = final["result"]
        print(f"{core}最终模型：coef={result.params[core]:.6f}, p={result.pvalues[core]:.6g}, N={int(result.nobs)}")
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
