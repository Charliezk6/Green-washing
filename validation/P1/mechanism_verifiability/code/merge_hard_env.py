"""Merge external hard environmental information onto the 15,381 common sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "results/derived"
OUT.mkdir(parents=True, exist_ok=True)

CERT = ROOT / "appendix/上市公司环境监管与认证披露情况表155713542(仅供南京审计大学使用)/ENV_SupCertDisclose.csv"
INVEST = ROOT / "appendix/上市公司环境投资统计表155903146(仅供南京审计大学使用)/ENV_InvestStats.csv"
INVEST_XLSX = ROOT / "appendix/上市公司环保投资额（1999-2024年）.xlsx"
PENALTY = ROOT / "appendix/上市公司环保处罚总表.xlsx"
INTERSECT = OUT / "audit_intersect_firm_year.csv"
COMMON = ROOT / "Regression/outputs/paper_tables/common_sample_15381.csv"


def pad_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.endswith(".0"):
        text = text[:-2]
    try:
        if text.isdigit() or (text.replace(".", "", 1).isdigit() and "." not in text):
            return str(int(float(text))).zfill(6)
        return str(int(float(text))).zfill(6)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(6) if digits else text.zfill(6)


def year_from_enddate(value: object) -> float:
    text = str(value).strip()
    if not text or text == "nan":
        return np.nan
    if len(text) >= 4 and text[:4].isdigit():
        return float(text[:4])
    try:
        return float(int(float(text)))
    except (TypeError, ValueError):
        return np.nan


def load_key_pollution() -> pd.DataFrame:
    data = pd.read_csv(CERT, dtype=str, encoding="utf-8-sig")
    data["证券代码"] = data["Symbol"].map(pad_code)
    data["时间"] = data["EndDate"].map(year_from_enddate)
    data["KeyPollMonUnit"] = pd.to_numeric(data["KeyPollMonUnit"], errors="coerce")
    data["IsPassISO14001"] = pd.to_numeric(data["IsPassISO14001"], errors="coerce")
    return (
        data.dropna(subset=["证券代码", "时间"])
        .sort_values(["证券代码", "时间"])
        .drop_duplicates(["证券代码", "时间"], keep="last")[["证券代码", "时间", "KeyPollMonUnit", "IsPassISO14001"]]
    )


def load_investment() -> pd.DataFrame:
    csmar = pd.read_csv(INVEST, dtype=str, encoding="utf-8-sig")
    csmar["证券代码"] = csmar["Symbol"].map(pad_code)
    csmar["时间"] = csmar["EndDate"].map(year_from_enddate)
    csmar["EPInvest"] = pd.to_numeric(csmar["EPInvest"], errors="coerce")
    csmar = (
        csmar.dropna(subset=["证券代码", "时间"])
        .sort_values(["证券代码", "时间"])
        .drop_duplicates(["证券代码", "时间"], keep="last")[["证券代码", "时间", "EPInvest"]]
    )
    xlsx = pd.read_excel(INVEST_XLSX, dtype=str)
    year_col = "年份" if "年份" in xlsx.columns else xlsx.columns[0]
    code_col = "股票代码" if "股票代码" in xlsx.columns else "证券代码"
    amount_col = "环保投资金额_元" if "环保投资金额_元" in xlsx.columns else xlsx.columns[-1]
    extra = pd.DataFrame(
        {
            "证券代码": xlsx[code_col].map(pad_code),
            "时间": pd.to_numeric(xlsx[year_col], errors="coerce"),
            "EnvInvestYuan": pd.to_numeric(xlsx[amount_col], errors="coerce"),
        }
    ).dropna(subset=["证券代码", "时间"])
    extra = extra.sort_values(["证券代码", "时间"]).drop_duplicates(["证券代码", "时间"], keep="last")
    return csmar.merge(extra, on=["证券代码", "时间"], how="outer")


def load_penalty() -> pd.DataFrame:
    raw = pd.read_excel(PENALTY, dtype=str)
    code_col = "证券代码" if "证券代码" in raw.columns else raw.columns[0]
    date_col = "处罚日期" if "处罚日期" in raw.columns else "处罚时间"
    amount_col = "处罚金额（万元）" if "处罚金额（万元）" in raw.columns else None
    data = pd.DataFrame(
        {
            "证券代码": raw[code_col].map(pad_code),
            "时间": raw[date_col].map(year_from_enddate),
            "PenaltyAmt": pd.to_numeric(raw[amount_col], errors="coerce") if amount_col else np.nan,
        }
    )
    data = data[(data["时间"] >= 2021) & (data["时间"] <= 2024)]
    grouped = data.groupby(["证券代码", "时间"], as_index=False).agg(
        Penalty_Count=("证券代码", "size"),
        PenaltyAmt=("PenaltyAmt", "sum"),
    )
    grouped["Penalty_D"] = 1
    grouped["LnPenaltyAmt"] = np.log1p(grouped["PenaltyAmt"].clip(lower=0))
    return grouped


def main() -> None:
    common = pd.read_csv(COMMON, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    common["证券代码"] = common["证券代码"].map(pad_code)
    common["时间"] = pd.to_numeric(common["时间"], errors="coerce")
    intersect = pd.read_csv(INTERSECT, dtype={"证券代码": str}, encoding="utf-8-sig")
    intersect["证券代码"] = intersect["证券代码"].map(pad_code)
    intersect["时间"] = pd.to_numeric(intersect["时间"], errors="coerce")

    merged = common.merge(intersect, on=["证券代码", "时间"], how="left", suffixes=("", "_intersect"))
    if "GI_intersect" in merged.columns:
        merged = merged.drop(columns=["GI_intersect"])
    if "ExpertiseIndustryCode_intersect" in merged.columns:
        merged = merged.drop(columns=["ExpertiseIndustryCode_intersect"])

    merged = merged.merge(load_key_pollution(), on=["证券代码", "时间"], how="left")
    merged = merged.merge(load_investment(), on=["证券代码", "时间"], how="left")
    merged = merged.merge(load_penalty(), on=["证券代码", "时间"], how="left")

    merged["Penalty_D"] = merged["Penalty_D"].fillna(0)
    merged["Penalty_Count"] = merged["Penalty_Count"].fillna(0)
    merged["LnPenaltyAmt"] = merged["LnPenaltyAmt"].fillna(0)
    merged["EnvInvest_D"] = (merged["EPInvest"].notna() | merged["EnvInvestYuan"].notna()).astype(float)
    invest = merged["EnvInvestYuan"].where(merged["EnvInvestYuan"].notna(), merged["EPInvest"])
    merged["LnEnvInvest"] = np.log1p(pd.to_numeric(invest, errors="coerce").clip(lower=0))

    merged.to_csv(OUT / "mechanism_firm_year_panel.csv", index=False, encoding="utf-8-sig")
    coverage = pd.DataFrame(
        [
            {
                "n": len(merged),
                "AuditIntersect_Share_nonmissing": int(merged["AuditIntersect_Share"].notna().sum()),
                "KeyPollMonUnit_nonmissing": int(merged["KeyPollMonUnit"].notna().sum()),
                "KeyPollMonUnit_1": int(merged["KeyPollMonUnit"].eq(1).sum()),
                "Penalty_D_1": int(merged["Penalty_D"].eq(1).sum()),
                "EnvInvest_D_1": int(merged["EnvInvest_D"].eq(1).sum()),
            }
        ]
    )
    coverage.to_csv(OUT / "hard_env_coverage.csv", index=False, encoding="utf-8-sig")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
