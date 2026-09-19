from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
PANEL_FILE = BASE_DIR / "greenwashing_auditor_regression_panel_2021_2024.csv"
INDUSTRY_FILE = (
    PROJECT_DIR
    / "appendix"
    / "上市公司基本信息年度表094521305(仅供南京审计大学使用)"
    / "STK_LISTEDCOINFOANL.csv"
)
AUDIT_FILE = (
    PROJECT_DIR
    / "appendix"
    / "上市公司审计机构列表153957251(仅供南京审计大学使用)"
    / "AR_LISTCOMPAUDIT.csv"
)
FINANCIAL_FILE = (
    PROJECT_DIR
    / "appendix"
    / "上市公司财务指标数据表(年)150744996(仅供南京审计大学使用)"
    / "AIQ_LCFinIndexY.csv"
)
FULL_MARKET_FILE = BASE_DIR / "full_a_share_audit_market_2021_2024.csv"
EXPERTISE_FILE = BASE_DIR / "auditor_industry_expertise_2021_2024.csv"
PANEL_WITH_EXPERTISE_FILE = Path(__file__).with_name(
    "greenwashing_panel_with_auditor_expertise_2021_2024.csv"
)
OUTPUT_DIR = Path(__file__).with_name("结果")
EXPERTISE_THRESHOLD = 0.10

CONTINUOUS_VARIABLES = [
    "ClientCount",
    "FirmIndustryRevenue",
    "IndustryYearRevenue",
    "FirmYearRevenue",
    "IMS",
    "IPS",
]
DUMMY_VARIABLES = ["IMS_D", "IPS_D"]
STATISTICS_VARIABLES = CONTINUOUS_VARIABLES + DUMMY_VARIABLES
PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def require_columns(data: pd.DataFrame, columns: set[str], source: str) -> None:
    missing_columns = sorted(columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"{source}缺少字段：{', '.join(missing_columns)}")


def calculate_expertise(panel: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "时间",
        "InstitutionID",
        "IndustryCode",
        "IndustryName",
        "AccountingFirmName",
        "OperatingRevenue",
    }
    require_columns(panel, required_columns, "面板数据")

    data = panel.copy()
    data["OperatingRevenue"] = pd.to_numeric(
        data["OperatingRevenue"], errors="coerce"
    )
    valid = (
        data["时间"].notna()
        & data["InstitutionID"].notna()
        & data["IndustryCode"].notna()
        & data["IndustryName"].notna()
        & data["AccountingFirmName"].notna()
        & data["OperatingRevenue"].gt(0)
    )
    data = data.loc[valid].copy()
    data["ExpertiseIndustryCode"] = data["IndustryCode"].astype("string")
    non_manufacturing = ~data["ExpertiseIndustryCode"].str.startswith(
        "C", na=False
    )
    data.loc[non_manufacturing, "ExpertiseIndustryCode"] = data.loc[
        non_manufacturing, "ExpertiseIndustryCode"
    ].str[:1]
    industry_year_counts = data.groupby(
        ["时间", "ExpertiseIndustryCode"]
    )["证券代码"].transform("nunique")
    data = data.loc[industry_year_counts.gt(1)].copy()

    group_columns = ["时间", "InstitutionID", "ExpertiseIndustryCode"]
    expertise = (
        data.groupby(group_columns, as_index=False)
        .agg(
            AccountingFirmName=("AccountingFirmName", "first"),
            IndustryName=("IndustryName", "first"),
            ClientCount=("OperatingRevenue", "size"),
            FirmIndustryRevenue=("OperatingRevenue", "sum"),
        )
    )

    industry_totals = (
        data.groupby(["时间", "ExpertiseIndustryCode"], as_index=False)[
            "OperatingRevenue"
        ]
        .sum()
        .rename(columns={"OperatingRevenue": "IndustryYearRevenue"})
    )
    firm_totals = (
        data.groupby(["时间", "InstitutionID"], as_index=False)[
            "OperatingRevenue"
        ]
        .sum()
        .rename(columns={"OperatingRevenue": "FirmYearRevenue"})
    )

    expertise = expertise.merge(
        industry_totals, on=["时间", "ExpertiseIndustryCode"], how="left"
    ).merge(firm_totals, on=["时间", "InstitutionID"], how="left")
    expertise["IMS"] = (
        expertise["FirmIndustryRevenue"] / expertise["IndustryYearRevenue"]
    )
    expertise["IMS_D"] = expertise["IMS"].ge(EXPERTISE_THRESHOLD).astype(int)
    expertise["IPS"] = (
        expertise["FirmIndustryRevenue"] / expertise["FirmYearRevenue"]
    )
    expertise["IPS_D"] = expertise["IPS"].ge(EXPERTISE_THRESHOLD).astype(int)

    output_columns = [
        "时间",
        "InstitutionID",
        "ExpertiseIndustryCode",
        "AccountingFirmName",
        "IndustryName",
        "ClientCount",
        "FirmIndustryRevenue",
        "IndustryYearRevenue",
        "FirmYearRevenue",
        "IMS",
        "IMS_D",
        "IPS",
        "IPS_D",
    ]
    return expertise[output_columns].sort_values(group_columns)


def load_full_a_share_market() -> pd.DataFrame:
    financial = pd.read_csv(
        FINANCIAL_FILE, dtype={"Symbol": str}, encoding="utf-8-sig"
    )
    financial["时间"] = pd.to_datetime(
        financial["EndDate"], errors="coerce"
    ).dt.year
    financial = financial.loc[financial["时间"].between(2021, 2024)].copy()

    audit = pd.read_csv(
        AUDIT_FILE, dtype={"Symbol": str, "InstitutionID": str},
        encoding="utf-8-sig",
    )
    audit["时间"] = pd.to_datetime(audit["EndDate"], errors="coerce").dt.year
    domestic = pd.to_numeric(
        audit["TerritoryForeignIdentity"], errors="coerce"
    ).eq(1)
    annual_audit = pd.to_numeric(audit["AuditItems"], errors="coerce").eq(1)
    audit = audit.loc[
        audit["时间"].between(2021, 2024) & annual_audit & domestic
    ].copy()
    audit = audit.sort_values(["Symbol", "时间", "Source"]).drop_duplicates(
        ["Symbol", "时间"], keep="first"
    )

    industry = pd.read_csv(
        INDUSTRY_FILE, dtype={"Symbol": str}, encoding="utf-8-sig"
    )
    industry = industry[
        ["Symbol", "IndustryName", "IndustryCode"]
    ].drop_duplicates("Symbol", keep="first")

    market = financial.merge(
        audit[["Symbol", "时间", "AccountingFirmName", "InstitutionID"]],
        on=["Symbol", "时间"], how="left", validate="one_to_one",
    ).merge(industry, on="Symbol", how="left", validate="many_to_one")
    market = market.rename(columns={"Symbol": "证券代码"})
    return market


def load_and_calculate_expertise() -> tuple[pd.DataFrame, int, int]:
    market = load_full_a_share_market()
    market.to_csv(FULL_MARKET_FILE, index=False, encoding="utf-8-sig")
    expertise = calculate_expertise(market)
    valid = (
        pd.to_numeric(market["OperatingRevenue"], errors="coerce").gt(0)
        & market["时间"].notna()
        & market["InstitutionID"].notna()
        & market["IndustryCode"].notna()
        & market["IndustryName"].notna()
        & market["AccountingFirmName"].notna()
    )
    excluded_count = len(market) - int(valid.sum())
    return expertise, len(market), excluded_count


def merge_expertise_to_panel(
    panel: pd.DataFrame, expertise: pd.DataFrame
) -> pd.DataFrame:
    panel = panel.copy()
    expertise = expertise.copy()
    panel["InstitutionID"] = panel["InstitutionID"].astype("string")
    expertise["InstitutionID"] = expertise["InstitutionID"].astype("string")
    panel["ExpertiseIndustryCode"] = panel["IndustryCode"].astype("string")
    non_manufacturing = ~panel["ExpertiseIndustryCode"].str.startswith(
        "C", na=False
    )
    panel.loc[non_manufacturing, "ExpertiseIndustryCode"] = panel.loc[
        non_manufacturing, "ExpertiseIndustryCode"
    ].str[:1]
    expertise["ExpertiseIndustryCode"] = expertise[
        "ExpertiseIndustryCode"
    ].astype("string")
    merge_keys = ["时间", "InstitutionID", "ExpertiseIndustryCode"]
    expertise_columns = merge_keys + ["IMS", "IMS_D", "IPS", "IPS_D"]
    return panel.merge(
        expertise[expertise_columns], on=merge_keys, how="left", validate="many_to_one"
    )


def calculate_overall_statistics(data: pd.DataFrame) -> pd.DataFrame:
    statistics = data[STATISTICS_VARIABLES].describe(
        percentiles=PERCENTILES
    ).T
    statistics.index.name = "变量"
    return statistics


def calculate_company_overall_statistics(data: pd.DataFrame) -> pd.DataFrame:
    statistics = data[["IMS", "IPS", "IMS_D", "IPS_D"]].describe(
        percentiles=PERCENTILES
    ).T
    statistics.index.name = "变量"
    return statistics


def calculate_annual_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("时间")[STATISTICS_VARIABLES].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )


def calculate_company_annual_statistics(data: pd.DataFrame) -> pd.DataFrame:
    return data.groupby("时间")[["IMS", "IPS", "IMS_D", "IPS_D"]].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )


def calculate_industry_statistics(data: pd.DataFrame) -> pd.DataFrame:
    statistics = data.groupby("IndustryName")[
        ["IMS", "IPS", "IMS_D", "IPS_D"]
    ].agg(["count", "mean", "std", "median", "min", "max"])
    return statistics.sort_values(("IMS", "count"), ascending=False)


def calculate_company_industry_statistics(data: pd.DataFrame) -> pd.DataFrame:
    statistics = data.groupby("IndustryName")[
        ["IMS", "IPS", "IMS_D", "IPS_D"]
    ].agg(["count", "mean", "std", "median", "min", "max"])
    return statistics.sort_values(("IMS", "count"), ascending=False)


def calculate_industry_year_diagnostics(
    market: pd.DataFrame, expertise: pd.DataFrame
) -> pd.DataFrame:
    data = market.copy()
    data["OperatingRevenue"] = pd.to_numeric(
        data["OperatingRevenue"], errors="coerce"
    )
    data["ExpertiseIndustryCode"] = data["IndustryCode"].astype("string")
    non_manufacturing = ~data["ExpertiseIndustryCode"].str.startswith(
        "C", na=False
    )
    data.loc[non_manufacturing, "ExpertiseIndustryCode"] = data.loc[
        non_manufacturing, "ExpertiseIndustryCode"
    ].str[:1]
    valid = (
        data["OperatingRevenue"].gt(0)
        & data["InstitutionID"].notna()
        & data["ExpertiseIndustryCode"].notna()
    )
    data = data.loc[valid].copy()

    keys = ["时间", "ExpertiseIndustryCode"]
    counts = data.groupby(keys, as_index=False).agg(
        CompanyCount=("证券代码", "nunique"),
        AccountingFirmCount=("InstitutionID", "nunique"),
        IndustryNames=("IndustryName", lambda x: "；".join(sorted(set(x.dropna())))),
    )
    eligible = counts.loc[counts["CompanyCount"].gt(1), keys]
    data = data.merge(eligible, on=keys, how="inner")

    detail = data.merge(
        expertise[["时间", "InstitutionID", "ExpertiseIndustryCode", "IMS"]],
        on=["时间", "InstitutionID", "ExpertiseIndustryCode"],
        how="inner",
        validate="many_to_one",
    )
    extrema = detail.groupby(keys)["IMS"].agg(IMS_Min="min", IMS_Max="max").reset_index()
    detail = detail.merge(extrema, on=keys, how="left")

    def join_subjects(group: pd.DataFrame, column: str) -> str:
        selected = group.loc[group["IMS"].eq(group[column])]
        values = (
            selected["证券代码"].astype(str)
            + " " + selected["ShortName"].fillna("").astype(str)
            + " | " + selected["AccountingFirmName"].fillna("").astype(str)
            + " | IMS=" + selected["IMS"].map(lambda value: f"{value:.10f}")
        )
        return "；".join(sorted(set(values)))

    subjects = detail.groupby(keys).apply(
        lambda group: pd.Series(
            {
                "IMS_Min_CompaniesAndFirms": join_subjects(group, "IMS_Min"),
                "IMS_Max_CompaniesAndFirms": join_subjects(group, "IMS_Max"),
            }
        ),
        include_groups=False,
    ).reset_index()
    return (
        counts.loc[counts["CompanyCount"].gt(1)]
        .merge(extrema, on=keys, how="left")
        .merge(subjects, on=keys, how="left")
        .sort_values(keys)
    )


def main() -> None:
    expertise, panel_count, excluded_count = load_and_calculate_expertise()
    expertise.to_csv(EXPERTISE_FILE, index=False, encoding="utf-8-sig")
    print(f"已生成：{EXPERTISE_FILE}")

    panel = pd.read_csv(
        PANEL_FILE, dtype={"证券代码": str}, encoding="utf-8-sig"
    )
    company_panel = merge_expertise_to_panel(panel, expertise)
    company_panel = company_panel.loc[company_panel["IMS"].notna()].copy()
    company_panel.to_csv(PANEL_WITH_EXPERTISE_FILE, index=False, encoding="utf-8-sig")
    print(f"已生成：{PANEL_WITH_EXPERTISE_FILE}")

    full_market = load_full_a_share_market()
    diagnostics = calculate_industry_year_diagnostics(full_market, expertise)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "事务所行业年度层面_总体描述性统计.csv": calculate_overall_statistics(
            expertise
        ),
        "事务所行业年度层面_分年度描述性统计.csv": calculate_annual_statistics(
            expertise
        ),
        "事务所行业年度层面_分行业描述性统计.csv": calculate_industry_statistics(
            expertise
        ),
        "公司年度层面_混合行业口径_剔除单公司行业_总体描述性统计.csv": calculate_company_overall_statistics(
            company_panel
        ),
        "公司年度层面_混合行业口径_剔除单公司行业_分年度描述性统计.csv": calculate_company_annual_statistics(
            company_panel
        ),
        "公司年度层面_混合行业口径_剔除单公司行业_分行业描述性统计.csv": calculate_company_industry_statistics(
            company_panel
        ),
        "年度分行业公司事务所数量及IMS极值明细.csv": diagnostics,
    }

    for filename, statistics in outputs.items():
        output_file = OUTPUT_DIR / filename
        statistics.to_csv(output_file, encoding="utf-8-sig")
        print(f"已生成：{output_file}")

    matched_count = int(company_panel["IMS"].notna().sum())
    research_panel_count = len(company_panel)
    print(f"完整A股市场公司—年度观测数：{panel_count}")
    print(f"完整市场中因行业、事务所或营业收入无效而排除：{excluded_count}")
    print(f"事务所—行业—年度观测数：{len(expertise)}")
    print(f"绿色漂洗研究面板观测数：{research_panel_count}")
    print(f"成功匹配行业专长的公司—年度观测数：{matched_count}")
    print(
        "未匹配行业专长的公司—年度观测数："
        f"{research_panel_count - matched_count}"
    )
    print(f"行业数量：{expertise['IndustryName'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
