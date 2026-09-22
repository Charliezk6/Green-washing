"""Build partner-level industry market share from the full A-share audit market.

Each signing CPA receives the client's operating revenue. The partner's industry
share divides that total by the industry-year revenue of all listed clients.
Company-year measures keep both the maximum and the mean across signing partners,
plus leave-one-out versions that drop the focal client's revenue.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parents[5]
CONFIG = json.loads((ROOT / "config" / "partner_ims_config.json").read_text(encoding="utf-8"))
MARKET = PROJECT / "X_reproduction" / "data" / "input" / "full_a_share_audit_market_2021_2024.csv"
FEE_CANDIDATES = list(PROJECT.glob("appendix/**/AIQ_FinAuditOpinY.csv"))
DATA = ROOT / "data"
DIAG = ROOT / "outputs" / "diagnostics"
FIRM = "证券代码"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"


def expertise_code(industry_code: object) -> str:
    code = str(industry_code).strip()
    return code if code.startswith("C") else code[:1]


def split_names(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).replace("，", ",").replace("、", ",").replace(";", ",")
    names = []
    for part in text.split(","):
        name = part.strip()
        if name and name not in {"-", "无", "None", "nan"}:
            names.append(name)
    return names


def load_market() -> pd.DataFrame:
    if not FEE_CANDIDATES:
        raise FileNotFoundError("AIQ_FinAuditOpinY.csv was not found under appendix/.")
    opinions = pd.read_csv(FEE_CANDIDATES[0], dtype={"Symbol": str}, encoding="utf-8-sig", low_memory=False)
    opinions[FIRM] = opinions["Symbol"].astype(str).str.zfill(6)
    opinions[YEAR] = pd.to_numeric(opinions["EndDate"].astype(str).str[:4], errors="coerce")
    opinions = opinions.dropna(subset=[FIRM, YEAR, "Auditor"]).drop_duplicates([FIRM, YEAR], keep="first")

    market = pd.read_csv(MARKET, dtype={FIRM: str, "InstitutionID": str}, encoding="utf-8-sig", low_memory=False)
    market[FIRM] = market[FIRM].astype(str).str.zfill(6)
    market[YEAR] = pd.to_numeric(market[YEAR], errors="coerce")
    market["OperatingRevenue"] = pd.to_numeric(market["OperatingRevenue"], errors="coerce")
    market[INDUSTRY] = market["IndustryCode"].map(expertise_code)
    market = market[market[YEAR].isin(CONFIG["years"])]
    market = market[market[INDUSTRY].ne("O")]
    market = market.dropna(subset=[FIRM, YEAR, INDUSTRY])
    market = market.drop_duplicates([FIRM, YEAR], keep="first")
    merged = market.merge(opinions[[FIRM, YEAR, "Auditor"]], on=[FIRM, YEAR], how="left")
    return merged


def partner_long(market: pd.DataFrame) -> pd.DataFrame:
    usable = market[market["OperatingRevenue"].gt(0)].copy()
    usable["partner"] = usable["Auditor"].map(split_names)
    long = usable.explode("partner").dropna(subset=["partner"])
    return long[[FIRM, YEAR, INDUSTRY, "partner", "OperatingRevenue"]].copy()


def build_company_measures(market: pd.DataFrame, long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    industry_total = (
        market.loc[market["OperatingRevenue"].gt(0)]
        .groupby([INDUSTRY, YEAR])["OperatingRevenue"]
        .sum()
        .rename("industry_revenue")
    )
    partner_total = (
        long.groupby(["partner", INDUSTRY, YEAR])["OperatingRevenue"]
        .sum()
        .rename("partner_revenue")
    )
    partner_share = partner_total.reset_index().merge(
        industry_total.reset_index(), on=[INDUSTRY, YEAR], how="left"
    )
    partner_share["PIMS"] = partner_share["partner_revenue"] / partner_share["industry_revenue"]
    # Ties at the maximum share are all leaders.
    maxima = partner_share.groupby([INDUSTRY, YEAR])["PIMS"].transform("max")
    partner_share["is_leader"] = partner_share["PIMS"].eq(maxima)
    threshold = partner_share.groupby([INDUSTRY, YEAR])["PIMS"].transform(lambda s: s.quantile(0.90))
    partner_share["is_p90"] = partner_share["PIMS"].ge(threshold)

    linked = long.merge(
        partner_share[["partner", INDUSTRY, YEAR, "PIMS", "partner_revenue", "is_leader", "is_p90"]],
        on=["partner", INDUSTRY, YEAR],
        how="left",
    )
    linked = linked.merge(industry_total.reset_index(), on=[INDUSTRY, YEAR], how="left")
    numerator = linked["partner_revenue"] - linked["OperatingRevenue"]
    denominator = linked["industry_revenue"] - linked["OperatingRevenue"]
    valid = numerator.ge(0) & denominator.gt(0)
    linked["PIMS_LOO"] = (numerator / denominator).where(valid)

    company = linked.groupby([FIRM, YEAR], as_index=False).agg(
        PIMS_max=("PIMS", "max"),
        PIMS_mean=("PIMS", "mean"),
        PIMS_LOO_max=("PIMS_LOO", "max"),
        PIMS_LOO_mean=("PIMS_LOO", "mean"),
        PIMS_D_leader=("is_leader", "max"),
        PIMS_D_p90=("is_p90", "max"),
        n_partners=("partner", "nunique"),
        partners=("partner", lambda s: "|".join(sorted(set(s)))),
    )
    for column in ("PIMS_D_leader", "PIMS_D_p90"):
        company[column] = company[column].astype(float)
    return company, partner_share


def distribution(partner_share: pd.DataFrame, company: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, series in (
        ("合伙人-行业-年度份额", partner_share["PIMS"]),
        ("公司-年度PIMS_max", company["PIMS_max"]),
        ("公司-年度PIMS_mean", company["PIMS_mean"]),
        ("公司-年度PIMS_LOO_max", company["PIMS_LOO_max"]),
    ):
        desc = series.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        rows.append({
            "变量": name,
            "观测数": int(series.notna().sum()),
            "均值": float(desc["mean"]),
            "标准差": float(desc["std"]),
            "p50": float(desc["50%"]),
            "p75": float(desc["75%"]),
            "p90": float(desc["90%"]),
            "p95": float(desc["95%"]),
            "p99": float(desc["99%"]),
            "最大值": float(desc["max"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    market = load_market()
    long = partner_long(market)
    company, partner_share = build_company_measures(market, long)
    company.to_csv(DATA / "partner_ims_company_year.csv", index=False, encoding="utf-8-sig")
    distribution(partner_share, company).to_csv(
        DIAG / "partner_share_distribution.csv", index=False, encoding="utf-8-sig"
    )
    positive = market["OperatingRevenue"].gt(0)
    summary = pd.DataFrame([{
        "市场公司年": int(len(market)),
        "正营收公司年": int(positive.sum()),
        "匹配到签字会计师": int(market["Auditor"].notna().sum()),
        "正营收且匹配签字会计师": int((positive & market["Auditor"].notna()).sum()),
        "合伙人-公司-年": int(len(long)),
        "独立合伙人姓名数": int(long["partner"].nunique()),
        "合伙人-行业-年": int(len(partner_share)),
        "公司-年输出": int(len(company)),
        "领导者公司年占比": float(company["PIMS_D_leader"].mean()),
        "P90公司年占比": float(company["PIMS_D_p90"].mean()),
        "平均签字人数": float(company["n_partners"].mean()),
    }])
    summary.to_csv(DIAG / "build_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(distribution(partner_share, company).to_string(index=False))


if __name__ == "__main__":
    main()
