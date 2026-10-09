"""Check IMS construction on the full A-share audit market.

IMS is the audit firm's operating-revenue share of an expertise industry-year.
The regression panel only stores the matched share, so client counts, industry
revenue, and participating firms are counted on the market file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
MARKET_PATH = ROOT / "X_reproduction" / "data" / "input" / "full_a_share_audit_market_2021_2024.csv"
PANEL_PATH = ROOT / "Regression" / "data" / "input" / "combined_greenwashing_auditor_expertise_panel.csv"
OUTPUT = HERE / "outputs"
YEARS = {2021, 2022, 2023, 2024}
RELATIVE_TOLERANCE = 1e-8
FIRM = "证券代码"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"
AUDITOR_ID = "InstitutionID"
AUDITOR_NAME = "AccountingFirmName"
REVENUE = "OperatingRevenue"


def expertise_code(industry_code: object) -> str:
    code = "" if pd.isna(industry_code) else str(industry_code).strip()
    if not code or code.lower() == "nan":
        return ""
    return code if code.startswith("C") else code[:1]


def identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def firm_code(value: object) -> str | None:
    text = identifier(value)
    if text is None:
        return None
    return text.zfill(6)


def load_market() -> pd.DataFrame:
    market = pd.read_csv(MARKET_PATH, encoding="utf-8-sig", low_memory=False)
    market[FIRM] = market[FIRM].map(firm_code)
    market[YEAR] = pd.to_numeric(market[YEAR], errors="coerce")
    market[REVENUE] = pd.to_numeric(market[REVENUE], errors="coerce")
    market[AUDITOR_ID] = market[AUDITOR_ID].map(identifier)
    market[INDUSTRY] = market["IndustryCode"].map(expertise_code)
    market[AUDITOR_NAME] = market[AUDITOR_NAME].astype("string").str.strip()
    valid = market[YEAR].isin(YEARS) & market[REVENUE].gt(0) & market[INDUSTRY].ne("") & market[INDUSTRY].ne("O")
    valid &= market[FIRM].notna() & market[AUDITOR_ID].notna()
    return market.loc[valid].copy()


def load_panel_keys() -> pd.DataFrame:
    panel = pd.read_csv(
        PANEL_PATH,
        usecols=[FIRM, YEAR, INDUSTRY, AUDITOR_ID, "IMS"],
        encoding="utf-8-sig",
        low_memory=False,
    )
    panel[FIRM] = panel[FIRM].map(firm_code)
    panel[YEAR] = pd.to_numeric(panel[YEAR], errors="coerce")
    panel[AUDITOR_ID] = panel[AUDITOR_ID].map(identifier)
    panel[INDUSTRY] = panel[INDUSTRY].map(lambda value: "" if pd.isna(value) else str(value).strip())
    panel["IMS"] = pd.to_numeric(panel["IMS"], errors="coerce")
    panel = panel.dropna(subset=[FIRM, YEAR]).copy()
    panel[YEAR] = panel[YEAR].astype(int)
    return panel


def build_tables(market: pd.DataFrame, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel_keys = set(zip(panel[FIRM], panel[YEAR].astype(int)))
    company = market.copy()
    company[YEAR] = company[YEAR].astype(int)
    company["in_regression_panel"] = [
        int((firm, year) in panel_keys) for firm, year in zip(company[FIRM], company[YEAR])
    ]
    industry_revenue = company.groupby([INDUSTRY, YEAR])[REVENUE].sum().rename("industry_year_revenue")
    company = company.merge(industry_revenue.reset_index(), on=[INDUSTRY, YEAR], how="left")
    company["revenue_share"] = company[REVENUE] / company["industry_year_revenue"]

    names = (
        company.sort_values([YEAR, INDUSTRY, AUDITOR_ID, FIRM])
        .groupby([YEAR, AUDITOR_ID, INDUSTRY], as_index=False)[AUDITOR_NAME]
        .first()
    )
    cells = company.groupby([YEAR, AUDITOR_ID, INDUSTRY], as_index=False).agg(
        n_companies=(FIRM, "size"),
        n_companies_in_panel=("in_regression_panel", "sum"),
        firm_industry_revenue=(REVENUE, "sum"),
    )
    cells = cells.merge(names, on=[YEAR, AUDITOR_ID, INDUSTRY], how="left")
    cells = cells.merge(industry_revenue.reset_index(), on=[INDUSTRY, YEAR], how="left")
    cells["IMS"] = cells["firm_industry_revenue"] / cells["industry_year_revenue"]
    cells = cells[
        [
            YEAR,
            AUDITOR_ID,
            AUDITOR_NAME,
            INDUSTRY,
            "n_companies",
            "n_companies_in_panel",
            "firm_industry_revenue",
            "industry_year_revenue",
            "IMS",
        ]
    ].sort_values([YEAR, INDUSTRY, AUDITOR_ID])

    industry = company.groupby([YEAR, INDUSTRY], as_index=False).agg(
        n_companies=(FIRM, "size"),
        n_audit_firms=(AUDITOR_ID, "nunique"),
        industry_revenue=(REVENUE, "sum"),
        n_companies_in_panel=("in_regression_panel", "sum"),
    ).sort_values([YEAR, INDUSTRY])

    company_out = company[
        [
            FIRM,
            YEAR,
            INDUSTRY,
            AUDITOR_ID,
            AUDITOR_NAME,
            REVENUE,
            "industry_year_revenue",
            "revenue_share",
            "in_regression_panel",
        ]
    ].sort_values([YEAR, INDUSTRY, FIRM])
    return cells, company_out, industry, company


def share_identity(company: pd.DataFrame) -> pd.Series:
    totals = company.groupby([YEAR, INDUSTRY])["revenue_share"].sum()
    return (totals - 1).abs()


def cell_identity(company: pd.DataFrame, cells: pd.DataFrame) -> pd.Series:
    client_share = company.groupby([YEAR, AUDITOR_ID, INDUSTRY])["revenue_share"].sum().rename("client_share")
    compared = cells.merge(client_share.reset_index(), on=[YEAR, AUDITOR_ID, INDUSTRY], how="left")
    return (compared["client_share"] - compared["IMS"]).abs()


def panel_ims_gap(panel: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    market_ims = cells[[YEAR, AUDITOR_ID, INDUSTRY, "IMS"]].rename(columns={"IMS": "IMS_market"})
    return panel.rename(columns={"IMS": "IMS_panel"}).merge(
        market_ims, on=[YEAR, AUDITOR_ID, INDUSTRY], how="left"
    )


def summary_rows(
    market_rows: int,
    panel: pd.DataFrame,
    cells: pd.DataFrame,
    company: pd.DataFrame,
    industry: pd.DataFrame,
    company_detail: pd.DataFrame,
) -> list[dict[str, object]]:
    share_gap = share_identity(company_detail)
    cell_gap = cell_identity(company_detail, cells)
    compared = panel_ims_gap(panel, cells)
    matched = compared["IMS_market"].notna() & compared["IMS_panel"].notna()
    usable = compared.loc[matched].copy()
    relative = (usable["IMS_panel"] - usable["IMS_market"]).abs() / usable["IMS_market"].abs()
    mismatches = usable.loc[relative.gt(RELATIVE_TOLERANCE), [FIRM, YEAR, AUDITOR_ID, INDUSTRY]]
    mismatch_key = ";".join(
        f"{firm}|{year}|{auditor}|{industry}"
        for firm, year, auditor, industry in mismatches.itertuples(index=False, name=None)
    )
    panel_firms = set(zip(panel[FIRM], panel[YEAR].astype(int)))
    market_firms = set(zip(company_detail[FIRM], company_detail[YEAR].astype(int)))
    return [
        {"item": "market_rows_valid", "value": market_rows},
        {"item": "firm_industry_year_rows", "value": len(cells)},
        {"item": "company_year_rows", "value": len(company)},
        {"item": "industry_year_rows", "value": len(industry)},
        {"item": "panel_company_years", "value": len(panel_firms)},
        {"item": "panel_company_years_in_valid_market", "value": len(panel_firms & market_firms)},
        {"item": "panel_company_years_not_in_valid_market", "value": len(panel_firms - market_firms)},
        {"item": "industry_share_sum_max_abs_gap", "value": float(share_gap.max())},
        {"item": "industry_share_sum_within_1e-8_share", "value": float(share_gap.le(RELATIVE_TOLERANCE).mean())},
        {"item": "cell_share_sum_max_abs_gap", "value": float(cell_gap.max())},
        {"item": "cell_share_sum_equals_ims_within_1e-8_share", "value": float(cell_gap.le(RELATIVE_TOLERANCE).mean())},
        {"item": "panel_rows_cell_and_ims_matched", "value": int(matched.sum())},
        {"item": "panel_ims_relative_error_within_1e-8_share", "value": float(relative.le(RELATIVE_TOLERANCE).mean())},
        {"item": "panel_ims_mismatch_rows", "value": int(len(mismatches))},
        {"item": "panel_ims_mismatch_keys", "value": mismatch_key},
        {"item": "panel_ims_max_abs_gap", "value": float((usable["IMS_panel"] - usable["IMS_market"]).abs().max())},
        {"item": "panel_ims_max_relative_error", "value": float(relative.max())},
    ]


def main() -> None:
    market = load_market()
    panel = load_panel_keys()
    cells, company, industry, company_detail = build_tables(market, panel)
    rows = summary_rows(len(market), panel, cells, company, industry, company_detail)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUTPUT / "firm_industry_year_clients.csv", index=False, encoding="utf-8-sig")
    company.to_csv(OUTPUT / "company_industry_revenue_share.csv", index=False, encoding="utf-8-sig")
    industry.to_csv(OUTPUT / "industry_year_structure.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUTPUT / "construction_summary.csv", index=False, encoding="utf-8-sig")
    for row in rows:
        print(f"{row['item']}={row['value']}")


if __name__ == "__main__":
    main()
