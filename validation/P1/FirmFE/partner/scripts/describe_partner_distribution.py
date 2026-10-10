"""Describe the signing-partner share distribution.

Reuses the market merge and PIMS formulas in build_partner_ims.py.
A partner-industry-year is one signing name in one expertise industry-year.
Company-year rows summarize that year's signing partners; they do not repeat
one partner's share once per client. Does not change the share definition
or refit regressions.
"""

from __future__ import annotations

import pandas as pd

from build_partner_ims import (
    DIAG,
    FIRM,
    INDUSTRY,
    PROJECT,
    YEAR,
    build_company_measures,
    load_market,
    partner_long,
)

PANEL_PATH = PROJECT / "Regression" / "data" / "input" / "combined_greenwashing_auditor_expertise_panel.csv"
REVENUE = "OperatingRevenue"
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def quantile_rows(layer: str, variable: str, series: pd.Series) -> list[dict[str, object]]:
    rows = [
        {"layer": layer, "variable": variable, "statistic": "n", "value": int(series.notna().sum())},
        {"layer": layer, "variable": variable, "statistic": "mean", "value": float(series.mean())},
        {"layer": layer, "variable": variable, "statistic": "min", "value": float(series.min())},
        {"layer": layer, "variable": variable, "statistic": "max", "value": float(series.max())},
    ]
    for probability in QUANTILES:
        label = f"p{int(probability * 100)}"
        rows.append({
            "layer": layer,
            "variable": variable,
            "statistic": label,
            "value": float(series.quantile(probability)),
        })
    return rows


def share_row(layer: str, variable: str, statistic: str, mask: pd.Series) -> dict[str, object]:
    return {"layer": layer, "variable": variable, "statistic": statistic, "value": float(mask.mean())}


def count_row(layer: str, variable: str, statistic: str, mask: pd.Series) -> dict[str, object]:
    return {"layer": layer, "variable": variable, "statistic": statistic, "value": int(mask.sum())}


def frequency_rows(layer: str, variable: str, series: pd.Series) -> list[dict[str, object]]:
    rows = []
    for count, frequency in series.value_counts().sort_index().items():
        rows.append({
            "layer": layer,
            "variable": variable,
            "statistic": str(int(count)),
            "value": int(frequency),
        })
    return rows


def load_panel_keys() -> set[tuple[str, int]]:
    panel = pd.read_csv(PANEL_PATH, usecols=[FIRM, YEAR], dtype={FIRM: str}, encoding="utf-8-sig")
    panel[FIRM] = panel[FIRM].astype(str).str.strip().str.zfill(6)
    panel[YEAR] = pd.to_numeric(panel[YEAR], errors="coerce")
    panel = panel.dropna(subset=[FIRM, YEAR])
    return set(zip(panel[FIRM], panel[YEAR].astype(int)))


def assemble() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    market = load_market()
    long = partner_long(market)
    company, partner_share = build_company_measures(market, long)
    n_clients = (
        long.groupby(["partner", INDUSTRY, YEAR])[FIRM]
        .nunique()
        .rename("n_companies")
        .reset_index()
    )
    partner_share = partner_share.merge(n_clients, on=["partner", INDUSTRY, YEAR], how="left")

    positive = market.loc[market[REVENUE].gt(0), [FIRM, YEAR, INDUSTRY, REVENUE]].copy()
    industry_revenue = positive.groupby([INDUSTRY, YEAR])[REVENUE].sum().rename("industry_revenue")
    positive = positive.merge(industry_revenue.reset_index(), on=[INDUSTRY, YEAR], how="left")
    positive["revenue_share"] = positive[REVENUE] / positive["industry_revenue"]
    company = company.merge(
        positive[[FIRM, YEAR, INDUSTRY, "revenue_share"]],
        on=[FIRM, YEAR],
        how="left",
        validate="one_to_one",
    )
    panel_keys = load_panel_keys()
    company["in_regression_panel"] = [
        int((firm, int(year)) in panel_keys) for firm, year in zip(company[FIRM], company[YEAR])
    ]

    industry_companies = positive.groupby([INDUSTRY, YEAR]).size().rename("n_companies")
    industry = partner_share.groupby([INDUSTRY, YEAR]).agg(
        n_partners=("partner", "nunique"),
        max_PIMS=("PIMS", "max"),
        sum_PIMS=("PIMS", "sum"),
        n_leaders=("is_leader", "sum"),
    )
    industry = industry.join(partner_share.groupby([INDUSTRY, YEAR])["PIMS"].quantile(0.90).rename("p90_threshold"))
    industry = industry.join(industry_companies)
    industry = industry.reset_index()
    return market, long, partner_share, company, industry


def describe(
    market: pd.DataFrame,
    long: pd.DataFrame,
    partner_share: pd.DataFrame,
    company: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    positive = market[REVENUE].gt(0)
    matched = market["Auditor"].notna()
    rows.append({"layer": "market", "variable": "company_year", "statistic": "n", "value": int(len(market))})
    rows.append({"layer": "market", "variable": "positive_revenue", "statistic": "n", "value": int(positive.sum())})
    rows.append({"layer": "market", "variable": "matched_auditor", "statistic": "n", "value": int(matched.sum())})
    rows.append({
        "layer": "market",
        "variable": "positive_and_matched",
        "statistic": "n",
        "value": int((positive & matched).sum()),
    })
    rows.append({"layer": "market", "variable": "partner_company_year", "statistic": "n", "value": int(len(long))})
    rows.append({
        "layer": "market",
        "variable": "partner_name",
        "statistic": "nunique",
        "value": int(long["partner"].nunique()),
    })
    rows.append({
        "layer": "market",
        "variable": "partner_industry_year",
        "statistic": "n",
        "value": int(len(partner_share)),
    })
    rows.append({"layer": "market", "variable": "company_year_output", "statistic": "n", "value": int(len(company))})
    rows.append({"layer": "market", "variable": "industry_year", "statistic": "n", "value": int(len(industry))})
    rows.append({
        "layer": "market",
        "variable": INDUSTRY,
        "statistic": "nunique",
        "value": int(industry[INDUSTRY].nunique()),
    })
    rows.append({"layer": "market", "variable": YEAR, "statistic": "nunique", "value": int(industry[YEAR].nunique())})

    rows.append({
        "layer": "partners",
        "variable": "partner_industry_year",
        "statistic": "n",
        "value": len(partner_share),
    })
    rows.extend(quantile_rows("partners", "n_companies", partner_share["n_companies"]))
    rows.append(share_row("partners", "n_companies", "share_eq_1", partner_share["n_companies"].eq(1)))
    rows.append(share_row("partners", "n_companies", "share_le_2", partner_share["n_companies"].le(2)))
    rows.append(share_row("partners", "n_companies", "share_ge_5", partner_share["n_companies"].ge(5)))
    rows.append(share_row("partners", "n_companies", "share_ge_10", partner_share["n_companies"].ge(10)))
    rows.extend(frequency_rows("partners", "n_companies_frequency", partner_share["n_companies"]))
    rows.extend(quantile_rows("partners", "PIMS", partner_share["PIMS"]))
    rows.append(share_row("partners", "PIMS", "share_lt_0.01", partner_share["PIMS"].lt(0.01)))
    rows.append(share_row("partners", "PIMS", "share_lt_0.05", partner_share["PIMS"].lt(0.05)))
    rows.append(share_row("partners", "PIMS", "share_ge_0.05", partner_share["PIMS"].ge(0.05)))
    rows.append(count_row("partners", "PIMS", "count_ge_0.05", partner_share["PIMS"].ge(0.05)))
    rows.append(share_row("partners", "PIMS", "share_ge_0.10", partner_share["PIMS"].ge(0.10)))
    rows.append(count_row("partners", "PIMS", "count_ge_0.10", partner_share["PIMS"].ge(0.10)))
    rows.append(share_row("partners", "PIMS", "share_ge_0.20", partner_share["PIMS"].ge(0.20)))
    rows.append(count_row("partners", "PIMS", "count_ge_0.20", partner_share["PIMS"].ge(0.20)))

    rows.append({"layer": "companies", "variable": "company_year", "statistic": "n", "value": len(company)})
    rows.extend(quantile_rows("companies", "n_partners", company["n_partners"]))
    rows.append(share_row("companies", "n_partners", "share_eq_1", company["n_partners"].eq(1)))
    rows.append(share_row("companies", "n_partners", "share_eq_2", company["n_partners"].eq(2)))
    rows.append(share_row("companies", "n_partners", "share_ge_3", company["n_partners"].ge(3)))
    rows.extend(frequency_rows("companies", "n_partners_frequency", company["n_partners"]))
    rows.extend(quantile_rows("companies", "revenue_share", company["revenue_share"]))
    rows.append(share_row("companies", "revenue_share", "share_gt_0.05", company["revenue_share"].gt(0.05)))
    rows.append(share_row("companies", "revenue_share", "share_gt_0.10", company["revenue_share"].gt(0.10)))
    rows.append(share_row("companies", "revenue_share", "share_gt_0.20", company["revenue_share"].gt(0.20)))
    for variable in ("PIMS_max", "PIMS_mean", "PIMS_LOO_max"):
        rows.extend(quantile_rows("companies", variable, company[variable]))
        rows.append(share_row("companies", variable, "share_ge_0.05", company[variable].ge(0.05)))
        rows.append(count_row("companies", variable, "count_ge_0.05", company[variable].ge(0.05)))
        rows.append(share_row("companies", variable, "share_ge_0.10", company[variable].ge(0.10)))
        rows.append(share_row("companies", variable, "share_ge_0.20", company[variable].ge(0.20)))
        rows.append(count_row("companies", variable, "count_ge_0.20", company[variable].ge(0.20)))
    rows.append(share_row("companies", "PIMS_LOO_max", "share_eq_0", company["PIMS_LOO_max"].eq(0)))
    rows.append(count_row("companies", "PIMS_LOO_max", "count_eq_0", company["PIMS_LOO_max"].eq(0)))
    rows.append(share_row("companies", "PIMS_D_leader", "share", company["PIMS_D_leader"].eq(1)))
    rows.append(count_row("companies", "PIMS_D_leader", "count", company["PIMS_D_leader"].eq(1)))
    rows.append(share_row("companies", "PIMS_D_p90", "share", company["PIMS_D_p90"].eq(1)))
    rows.append(count_row("companies", "PIMS_D_p90", "count", company["PIMS_D_p90"].eq(1)))
    rows.append(share_row("companies", "in_regression_panel", "share", company["in_regression_panel"].eq(1)))
    rows.append(count_row("companies", "in_regression_panel", "count", company["in_regression_panel"].eq(1)))

    panel = company.loc[company["in_regression_panel"].eq(1)]
    rows.append({"layer": "panel_companies", "variable": "company_year", "statistic": "n", "value": len(panel)})
    rows.extend(quantile_rows("panel_companies", "PIMS_max", panel["PIMS_max"]))
    rows.append(share_row("panel_companies", "PIMS_max", "share_ge_0.05", panel["PIMS_max"].ge(0.05)))
    rows.append(share_row("panel_companies", "PIMS_max", "share_ge_0.10", panel["PIMS_max"].ge(0.10)))
    rows.append(share_row("panel_companies", "PIMS_max", "share_ge_0.20", panel["PIMS_max"].ge(0.20)))
    rows.append(count_row("panel_companies", "PIMS_max", "count_ge_0.20", panel["PIMS_max"].ge(0.20)))
    rows.extend(quantile_rows("panel_companies", "PIMS_LOO_max", panel["PIMS_LOO_max"]))
    rows.append(share_row("panel_companies", "PIMS_LOO_max", "share_eq_0", panel["PIMS_LOO_max"].eq(0)))
    rows.append(share_row("panel_companies", "PIMS_D_leader", "share", panel["PIMS_D_leader"].eq(1)))
    rows.append(count_row("panel_companies", "PIMS_D_leader", "count", panel["PIMS_D_leader"].eq(1)))
    rows.append(share_row("panel_companies", "PIMS_D_p90", "share", panel["PIMS_D_p90"].eq(1)))
    rows.append(count_row("panel_companies", "PIMS_D_p90", "count", panel["PIMS_D_p90"].eq(1)))
    rows.extend(quantile_rows("panel_companies", "n_partners", panel["n_partners"]))

    rows.append({"layer": "industries", "variable": "industry_year", "statistic": "n", "value": len(industry)})
    rows.extend(quantile_rows("industries", "n_companies", industry["n_companies"]))
    rows.extend(quantile_rows("industries", "n_partners", industry["n_partners"]))
    rows.extend(quantile_rows("industries", "max_PIMS", industry["max_PIMS"]))
    rows.append(share_row("industries", "max_PIMS", "share_ge_0.10", industry["max_PIMS"].ge(0.10)))
    rows.append(share_row("industries", "max_PIMS", "share_ge_0.20", industry["max_PIMS"].ge(0.20)))
    rows.append(share_row("industries", "max_PIMS", "share_ge_0.50", industry["max_PIMS"].ge(0.50)))
    rows.extend(quantile_rows("industries", "p90_threshold", industry["p90_threshold"]))
    rows.extend(quantile_rows("industries", "sum_PIMS", industry["sum_PIMS"]))
    rows.append(share_row("industries", "sum_PIMS", "share_gt_1", industry["sum_PIMS"].gt(1)))
    rows.extend(quantile_rows("industries", "n_leaders", industry["n_leaders"]))
    rows.append(share_row("industries", "n_leaders", "share_eq_1", industry["n_leaders"].eq(1)))
    return pd.DataFrame(rows)


def main() -> None:
    summary = describe(*assemble())
    destination = DIAG / "partner_distribution_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False, encoding="utf-8-sig")
    print(f"wrote {len(summary)} rows to {destination}")


if __name__ == "__main__":
    main()
