"""Construct the audit-fee industry-leader specialist dummy SPEC.

FeeShare_{a,k,t} = sum_{i in (a,k,t)} Fee_i / sum_{i in (k,t)} Fee_i
SPEC_{i,t}       = 1{FeeShare_{a(i),k(i),t} = max_b FeeShare_{b,k(i),t}}  (ties all coded 1)
SPEC_LOO_{i,t}   = 1{A_{a,k,t} - Fee_i > 0 and A_{a,k,t} - Fee_i >= max_{b != a} A_{b,k,t}}

The market is the full A-share audit market 2021-2024 (same universe and industry
rule as IMS). The client's auditor is taken from the full-market file by
security code and year.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEST = Path(__file__).resolve().parents[1]
PROJECT = TEST.parent
MARKET = PROJECT / "X_reproduction/data/input/full_a_share_audit_market_2021_2024.csv"
FEE_FILE = next(PROJECT.glob("appendix/**/AIQ_FinAuditOpinY.csv"))
PANELS = {
    "combined_greenwashing_auditor_expertise_panel_spec.csv": PROJECT / "Regression/data/input/combined_greenwashing_auditor_expertise_panel.csv",
    "combined_regression_panel_15381_spec.csv": PROJECT / "validation/P1/data/input/combined_regression_panel_15381.csv",
    "common_sample_15381_soe_imputed_spec.csv": PROJECT / "validation/panel_version_audit/outputs/soe_impute/common_sample_15381_soe_imputed.csv",
}
DATA = TEST / "data"
OUT = Path(__file__).resolve().parent / "outputs"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
YEARS = [2021, 2022, 2023, 2024]
NEW_COLUMNS = ["SPEC_AuditorID", "SPEC_Fee", "FeeShare", "SPEC", "SPEC_LOO"]


def expertise_code(industry_code: object) -> str:
    if pd.isna(industry_code):
        return ""
    code = str(industry_code).strip()
    return code if code.startswith("C") else code[:1]


def normalize_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def read_fees() -> pd.DataFrame:
    fees = pd.read_csv(FEE_FILE, dtype={"Symbol": str}, encoding="utf-8-sig", low_memory=False)
    fees = fees[fees["EndDate"].astype(str).str[5:10].eq("12-31")].copy()
    fees[FIRM] = fees["Symbol"].str.zfill(6)
    fees[YEAR] = pd.to_numeric(fees["EndDate"].astype(str).str[:4], errors="coerce")
    fees["TotalAuditFee"] = pd.to_numeric(fees["TotalAuditFee"], errors="coerce")
    fees = fees.sort_values("TotalAuditFee", ascending=False).drop_duplicates([FIRM, YEAR], keep="first")
    return fees[[FIRM, YEAR, "TotalAuditFee"]]


def load_market() -> pd.DataFrame:
    market = pd.read_csv(MARKET, dtype={FIRM: str, "InstitutionID": str}, encoding="utf-8-sig", low_memory=False)
    market[FIRM] = market[FIRM].str.zfill(6)
    market[YEAR] = pd.to_numeric(market[YEAR], errors="coerce")
    market["SPEC_AuditorID"] = market["InstitutionID"].map(normalize_id)
    market["MarketIndustry"] = market["IndustryCode"].map(expertise_code)
    market = market[market[YEAR].isin(YEARS)].merge(read_fees(), on=[FIRM, YEAR], how="left", validate="one_to_one")
    market["SPEC_Fee"] = market["TotalAuditFee"].where(market["TotalAuditFee"].gt(0))
    return market


def fee_shares(market: pd.DataFrame) -> pd.DataFrame:
    valid = market.dropna(subset=["SPEC_AuditorID", "SPEC_Fee"])
    valid = valid[valid["MarketIndustry"].ne("O") & valid["MarketIndustry"].ne("")]
    cells = valid.groupby([YEAR, "MarketIndustry", "SPEC_AuditorID"]).agg(
        ClientCount=(FIRM, "size"), AuditorIndustryFee=("SPEC_Fee", "sum")
    ).reset_index()
    group = cells.groupby([YEAR, "MarketIndustry"])["AuditorIndustryFee"]
    cells["IndustryYearFee"] = group.transform("sum")
    cells["FeeShare"] = cells["AuditorIndustryFee"] / cells["IndustryYearFee"]
    cells["MaxFeeShare"] = cells.groupby([YEAR, "MarketIndustry"])["FeeShare"].transform("max")
    cells["SPEC"] = np.isclose(cells["FeeShare"], cells["MaxFeeShare"], rtol=0, atol=1e-12).astype(int)
    ordered = cells.sort_values([YEAR, "MarketIndustry", "AuditorIndustryFee"], ascending=[True, True, False])
    top = ordered.groupby([YEAR, "MarketIndustry"])["AuditorIndustryFee"]
    cells["Top1Fee"] = top.transform("max")
    second = ordered.groupby([YEAR, "MarketIndustry"])["AuditorIndustryFee"].apply(lambda s: s.iloc[1] if len(s) > 1 else 0.0)
    cells["Top2Fee"] = cells.set_index([YEAR, "MarketIndustry"]).index.map(second)
    names = market.dropna(subset=["SPEC_AuditorID"]).drop_duplicates([YEAR, "SPEC_AuditorID"]).set_index([YEAR, "SPEC_AuditorID"])["AccountingFirmName"]
    cells["AccountingFirmName"] = pd.MultiIndex.from_frame(cells[[YEAR, "SPEC_AuditorID"]]).map(names)
    return cells.rename(columns={"MarketIndustry": INDUSTRY})


def firm_year_spec(market: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    firm = market[[FIRM, YEAR, "SPEC_AuditorID", "MarketIndustry", "SPEC_Fee"]].rename(columns={"MarketIndustry": "_MarketIndustry"})
    firm = firm.merge(
        cells[[YEAR, INDUSTRY, "SPEC_AuditorID", "AuditorIndustryFee", "FeeShare", "SPEC", "Top1Fee", "Top2Fee"]],
        left_on=[YEAR, "_MarketIndustry", "SPEC_AuditorID"], right_on=[YEAR, INDUSTRY, "SPEC_AuditorID"], how="left",
    ).drop(columns=[INDUSTRY])
    # Auditors with no positive-fee client in the industry-year hold zero fee share.
    has_industry = firm.set_index([YEAR, "_MarketIndustry"]).index.isin(cells.set_index([YEAR, INDUSTRY]).index)
    no_cell = firm["SPEC"].isna() & firm["SPEC_AuditorID"].notna() & has_industry & firm["_MarketIndustry"].ne("O")
    firm.loc[no_cell, ["AuditorIndustryFee", "FeeShare", "SPEC"]] = 0.0
    industry_top = cells.groupby([YEAR, INDUSTRY]).agg(Top1Fee=("Top1Fee", "first"), Top2Fee=("Top2Fee", "first"))
    keys = pd.MultiIndex.from_frame(firm[[YEAR, "_MarketIndustry"]])
    firm["Top1Fee"] = keys.map(industry_top["Top1Fee"])
    firm["Top2Fee"] = keys.map(industry_top["Top2Fee"])
    own = firm["SPEC_Fee"].fillna(0.0)
    remaining = firm["AuditorIndustryFee"] - own
    # Best rival total: the top fee if the focal auditor is not the leader, otherwise the runner-up.
    is_leader = np.isclose(firm["AuditorIndustryFee"], firm["Top1Fee"], rtol=0, atol=1e-6)
    rival = np.where(is_leader, firm["Top2Fee"], firm["Top1Fee"])
    firm["SPEC_LOO"] = ((remaining > 0) & (remaining >= rival - 1e-6)).astype(float).where(firm["SPEC"].notna())
    return firm[[FIRM, YEAR, "_MarketIndustry", *NEW_COLUMNS]]


def crosscheck_firmfe_ims_fee(cells: pd.DataFrame) -> dict:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(PROJECT / "validation/P1/FirmFE/scripts"))
    from measurement_error_correction import alternative_shares

    reference = alternative_shares()[["InstitutionID", INDUSTRY, YEAR, "IMS_fee"]].dropna(subset=["IMS_fee"])
    reference["ref_leader"] = np.isclose(
        reference["IMS_fee"], reference.groupby([INDUSTRY, YEAR])["IMS_fee"].transform("max"), rtol=0, atol=1e-12
    ).astype(int)
    joined = cells.merge(reference, left_on=["SPEC_AuditorID", INDUSTRY, YEAR], right_on=["InstitutionID", INDUSTRY, YEAR], how="outer", indicator=True)
    both = joined[joined["_merge"].eq("both")]
    return {
        "cells_both": int(len(both)),
        "cells_only_spec": int(joined["_merge"].eq("left_only").sum()),
        "cells_only_firmfe": int(joined["_merge"].eq("right_only").sum()),
        "max_abs_share_difference": float((both["FeeShare"] - both["IMS_fee"]).abs().max()),
        "leader_agreement_rate": float(both["SPEC"].eq(both["ref_leader"]).mean()),
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    market = load_market()
    cells = fee_shares(market)
    cells.drop(columns=["MaxFeeShare"]).to_csv(DATA / "spec_fee_share_auditor_industry_year.csv", index=False, encoding="utf-8-sig")
    firm = firm_year_spec(market, cells)
    firm.to_csv(DATA / "spec_company_year_full_market.csv", index=False, encoding="utf-8-sig")

    checks = {}
    industry_groups = cells.groupby([YEAR, INDUSTRY])
    checks["industry_years"] = int(industry_groups.ngroups)
    checks["industry_years_without_leader"] = int((industry_groups["SPEC"].sum() == 0).sum())
    checks["industry_years_with_tied_leader"] = int((industry_groups["SPEC"].sum() > 1).sum())
    checks["max_abs_share_sum_minus_1"] = float((industry_groups["FeeShare"].sum() - 1).abs().max())
    checks["market_rows"] = int(len(market))
    checks["market_rows_with_positive_fee"] = int(market["SPEC_Fee"].notna().sum())

    for output_name, source in PANELS.items():
        panel = pd.read_csv(source, dtype={FIRM: str}, encoding="utf-8-sig", low_memory=False)
        panel[FIRM] = panel[FIRM].str.zfill(6)
        panel[YEAR] = pd.to_numeric(panel[YEAR], errors="coerce")
        panel = panel.drop(columns=[c for c in NEW_COLUMNS if c in panel.columns])
        merged = panel.merge(firm, on=[FIRM, YEAR], how="left", validate="many_to_one")
        industry_mismatch = merged["_MarketIndustry"].notna() & merged["_MarketIndustry"].ne(merged[INDUSTRY].astype(str))
        merged = merged.drop(columns=["_MarketIndustry"])
        merged.to_csv(DATA / output_name, index=False, encoding="utf-8-sig")
        panel_id = merged["InstitutionID"].map(normalize_id) if "InstitutionID" in merged.columns else pd.Series(dtype=object)
        checks[output_name] = {
            "rows": int(len(merged)),
            "spec_nonmissing": int(merged["SPEC"].notna().sum()),
            "spec_mean": float(merged["SPEC"].mean()),
            "spec_loo_nonmissing": int(merged["SPEC_LOO"].notna().sum()),
            "spec_loo_mean": float(merged["SPEC_LOO"].mean()),
            "industry_code_mismatch_vs_market": int(industry_mismatch.sum()),
            "panel_institution_id_differs_from_market": int((panel_id != merged["SPEC_AuditorID"]).sum()),
        }

    checks["firmfe_ims_fee_crosscheck"] = crosscheck_firmfe_ims_fee(cells)
    (OUT / "build_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
