"""Describe the IMS distribution from the construction tables.

Reads the three market tables already written by check_ims_construction.py.
A cell is one audit firm in one expertise industry-year. Company-year rows
repeat that cell's IMS once per client.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
OUTPUT = HERE / "outputs"
CELLS_PATH = OUTPUT / "firm_industry_year_clients.csv"
COMPANY_PATH = OUTPUT / "company_industry_revenue_share.csv"
INDUSTRY_PATH = OUTPUT / "industry_year_structure.csv"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"
AUDITOR_ID = "InstitutionID"
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
        rows.append({"layer": layer, "variable": variable, "statistic": label, "value": float(series.quantile(probability))})
    return rows


def share_row(layer: str, variable: str, statistic: str, mask: pd.Series) -> dict[str, object]:
    return {"layer": layer, "variable": variable, "statistic": statistic, "value": float(mask.mean())}


def count_row(layer: str, variable: str, statistic: str, mask: pd.Series) -> dict[str, object]:
    return {"layer": layer, "variable": variable, "statistic": statistic, "value": int(mask.sum())}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(CELLS_PATH, encoding="utf-8-sig")
    company = pd.read_csv(COMPANY_PATH, encoding="utf-8-sig")
    industry = pd.read_csv(INDUSTRY_PATH, encoding="utf-8-sig")
    company = company.merge(
        cells[[YEAR, AUDITOR_ID, INDUSTRY, "IMS", "n_companies"]],
        on=[YEAR, AUDITOR_ID, INDUSTRY],
        how="left",
        validate="many_to_one",
    )
    return cells, company, industry


def describe(cells: pd.DataFrame, company: pd.DataFrame, industry: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append({"layer": "cells", "variable": "firm_industry_year", "statistic": "n", "value": len(cells)})
    rows.extend(quantile_rows("cells", "n_companies", cells["n_companies"]))
    rows.append(share_row("cells", "n_companies", "share_eq_1", cells["n_companies"].eq(1)))
    rows.append(share_row("cells", "n_companies", "share_le_2", cells["n_companies"].le(2)))
    rows.append(share_row("cells", "n_companies", "share_ge_5", cells["n_companies"].ge(5)))
    rows.append(share_row("cells", "n_companies", "share_ge_10", cells["n_companies"].ge(10)))
    for count, frequency in cells["n_companies"].value_counts().sort_index().items():
        rows.append({"layer": "cells", "variable": "n_companies_frequency", "statistic": str(int(count)), "value": int(frequency)})
    rows.extend(quantile_rows("cells", "IMS", cells["IMS"]))
    rows.append(share_row("cells", "IMS", "share_lt_0.05", cells["IMS"].lt(0.05)))
    rows.append(share_row("cells", "IMS", "share_ge_0.10", cells["IMS"].ge(0.10)))
    rows.append(share_row("cells", "IMS", "share_ge_0.20", cells["IMS"].ge(0.20)))
    rows.append(count_row("cells", "IMS", "count_ge_0.20", cells["IMS"].ge(0.20)))

    rows.append({"layer": "companies", "variable": "company_year", "statistic": "n", "value": len(company)})
    rows.append(share_row("companies", "in_regression_panel", "share", company["in_regression_panel"].eq(1)))
    rows.append(count_row("companies", "in_regression_panel", "count", company["in_regression_panel"].eq(1)))
    rows.extend(quantile_rows("companies", "revenue_share", company["revenue_share"]))
    rows.append(share_row("companies", "revenue_share", "share_gt_0.05", company["revenue_share"].gt(0.05)))
    rows.append(share_row("companies", "revenue_share", "share_gt_0.10", company["revenue_share"].gt(0.10)))
    rows.append(share_row("companies", "revenue_share", "share_gt_0.20", company["revenue_share"].gt(0.20)))
    rows.extend(quantile_rows("companies", "IMS", company["IMS"]))
    rows.append(share_row("companies", "IMS", "share_ge_0.10", company["IMS"].ge(0.10)))
    rows.append(share_row("companies", "IMS", "share_ge_0.20", company["IMS"].ge(0.20)))
    rows.append(count_row("companies", "IMS", "count_ge_0.20", company["IMS"].ge(0.20)))
    rows.append(share_row("companies", "n_companies", "share_eq_1", company["n_companies"].eq(1)))
    rows.extend(quantile_rows("companies", "n_companies", company["n_companies"]))

    panel = company.loc[company["in_regression_panel"].eq(1)]
    rows.append({"layer": "panel_companies", "variable": "company_year", "statistic": "n", "value": len(panel)})
    rows.extend(quantile_rows("panel_companies", "IMS", panel["IMS"]))
    rows.append(share_row("panel_companies", "IMS", "share_ge_0.10", panel["IMS"].ge(0.10)))
    rows.append(share_row("panel_companies", "IMS", "share_ge_0.20", panel["IMS"].ge(0.20)))
    rows.append(count_row("panel_companies", "IMS", "count_ge_0.20", panel["IMS"].ge(0.20)))
    rows.append(share_row("panel_companies", "n_companies", "share_eq_1", panel["n_companies"].eq(1)))
    rows.extend(quantile_rows("panel_companies", "n_companies", panel["n_companies"]))

    rows.append({"layer": "industries", "variable": "industry_year", "statistic": "n", "value": len(industry)})
    rows.append({"layer": "industries", "variable": "ExpertiseIndustryCode", "statistic": "nunique", "value": int(industry[INDUSTRY].nunique())})
    rows.append({"layer": "industries", "variable": YEAR, "statistic": "nunique", "value": int(industry[YEAR].nunique())})
    rows.extend(quantile_rows("industries", "n_companies", industry["n_companies"]))
    rows.extend(quantile_rows("industries", "n_audit_firms", industry["n_audit_firms"]))
    coverage = industry["n_companies_in_panel"] / industry["n_companies"]
    rows.extend(quantile_rows("industries", "panel_coverage", coverage))
    maximum = cells.groupby([YEAR, INDUSTRY])["IMS"].max()
    rows.extend(quantile_rows("industries", "max_IMS", maximum))
    rows.append(share_row("industries", "max_IMS", "share_ge_0.20", maximum.ge(0.20)))
    rows.append(share_row("industries", "max_IMS", "share_ge_0.50", maximum.ge(0.50)))
    hhi = cells.groupby([YEAR, INDUSTRY])["IMS"].apply(lambda shares: float((shares ** 2).sum()))
    rows.extend(quantile_rows("industries", "HHI", hhi))
    return pd.DataFrame(rows)


def main() -> None:
    summary = describe(*load())
    summary.to_csv(OUTPUT / "distribution_summary.csv", index=False, encoding="utf-8-sig")
    print(f"wrote {len(summary)} rows to {OUTPUT / 'distribution_summary.csv'}")


if __name__ == "__main__":
    main()
