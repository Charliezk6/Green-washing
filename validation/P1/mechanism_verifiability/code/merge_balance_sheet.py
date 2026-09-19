"""Merge consolidated CIP, provisions, PPE, and cash-flow capex onto the 15,381 sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "results/derived"
OUT.mkdir(parents=True, exist_ok=True)

FS = ROOT / "appendix/资产负债表121834604(仅供南京审计大学使用)/FS_Combas.csv"
CAPEX = ROOT / "appendix/财务指标文件123452422(仅供南京审计大学使用)/CSR_Finidx.csv"
BASE = OUT / "mechanism_firm_year_panel.csv"


def pad_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return str(int(float(text))).zfill(6)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(6) if digits else text.zfill(6)


def load_balance_sheet() -> pd.DataFrame:
    data = pd.read_csv(FS, dtype={"Stkcd": str}, encoding="utf-8-sig", low_memory=False)
    data["证券代码"] = data["Stkcd"].map(pad_code)
    accper = data["Accper"].astype(str)
    data = data[accper.str.endswith("12-31")].copy()
    data = data[data["Typrep"].astype(str).str.upper().eq("A")].copy()
    data["时间"] = accper.str.slice(0, 4).astype(int)
    data["PPE"] = pd.to_numeric(data["A001212000"], errors="coerce")
    data["CIP"] = pd.to_numeric(data["A001213000"], errors="coerce")
    data["Provision"] = pd.to_numeric(data["A002207000"], errors="coerce")
    data = data.sort_values(["证券代码", "时间"]).drop_duplicates(["证券代码", "时间"], keep="last")
    return data[["证券代码", "时间", "PPE", "CIP", "Provision"]]


def load_capex() -> pd.DataFrame:
    data = pd.read_csv(CAPEX, dtype={"Stkcd": str}, encoding="utf-8-sig", low_memory=False)
    data["证券代码"] = data["Stkcd"].map(pad_code)
    accper = data["Accper"].astype(str)
    data = data[accper.str.endswith("12-31")].copy()
    data["时间"] = accper.str.slice(0, 4).astype(int)
    data["Outcap"] = pd.to_numeric(data["Outcap"], errors="coerce")
    data.loc[data["Outcap"].lt(0), "Outcap"] = np.nan
    data = data.sort_values(["证券代码", "时间"]).drop_duplicates(["证券代码", "时间"], keep="last")
    return data[["证券代码", "时间", "Outcap"]]


def add_high_flag(data: pd.DataFrame, ratio: str, flag: str) -> pd.Series:
    median = data.groupby(["ExpertiseIndustryCode", "时间"])[ratio].transform("median")
    return (data[ratio] > median).astype(float)


def main() -> None:
    base = pd.read_csv(BASE, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    base["证券代码"] = base["证券代码"].map(pad_code)
    base["时间"] = pd.to_numeric(base["时间"], errors="coerce").astype(int)
    if len(base) != 15381:
        raise ValueError(f"Expected 15,381 rows in mechanism panel, got {len(base)}")

    sheet = load_balance_sheet()
    capex = load_capex()
    financial = sheet.merge(capex, on=["证券代码", "时间"], how="outer")

    merged = base.merge(financial, on=["证券代码", "时间"], how="left", validate="one_to_one")
    unmatched = merged[["PPE", "CIP", "Provision", "Outcap"]].isna().all(axis=1).sum()
    if unmatched:
        raise ValueError(f"Balance-sheet/capex merge left {unmatched} firm-years unmatched")

    merged["PPE_observed"] = merged["PPE"].notna().astype(int)
    merged["CIP_observed"] = merged["CIP"].notna().astype(int)
    merged["Provision_observed"] = merged["Provision"].notna().astype(int)
    merged["Capex_observed"] = merged["Outcap"].notna().astype(int)

    assets = pd.to_numeric(merged["TotalAssets"], errors="coerce")
    for raw in ["PPE", "CIP", "Provision", "Outcap"]:
        merged[raw] = pd.to_numeric(merged[raw], errors="coerce")

    filled = merged.copy()
    for raw in ["PPE", "CIP", "Provision", "Outcap"]:
        filled[raw] = filled[raw].fillna(0)

    filled["PPE_TA"] = filled["PPE"] / assets
    filled["CIP_TA"] = filled["CIP"] / assets
    filled["Provision_TA"] = filled["Provision"] / assets
    filled["Capex_TA"] = filled["Outcap"] / assets
    filled["Provision_D"] = (filled["Provision"] > 0).astype(float)
    filled["HighCapex"] = add_high_flag(filled, "Capex_TA", "HighCapex")
    filled["HighCIP"] = add_high_flag(filled, "CIP_TA", "HighCIP")

    lagged = financial.copy()
    lagged["时间"] = lagged["时间"] + 1
    lagged = lagged.rename(columns={"PPE": "L_PPE_raw", "CIP": "L_CIP_raw", "Provision": "L_Provision_raw", "Outcap": "L_Outcap_raw"})
    filled = filled.merge(lagged, on=["证券代码", "时间"], how="left")
    filled["L_Capex_observed"] = filled["L_Outcap_raw"].notna().astype(float)
    filled["L_CIP_observed"] = filled["L_CIP_raw"].notna().astype(float)
    for raw, filled_name in [
        ("L_PPE_raw", "L_PPE"),
        ("L_CIP_raw", "L_CIP"),
        ("L_Provision_raw", "L_Provision"),
        ("L_Outcap_raw", "L_Outcap"),
    ]:
        filled[filled_name] = pd.to_numeric(filled[raw], errors="coerce").fillna(0)
        filled.drop(columns=[raw], inplace=True)
    filled["L_PPE_TA"] = filled["L_PPE"] / assets
    filled["L_CIP_TA"] = filled["L_CIP"] / assets
    filled["L_Provision_TA"] = filled["L_Provision"] / assets
    filled["L_Capex_TA"] = filled["L_Outcap"] / assets
    filled["L_Provision_D"] = (filled["L_Provision"] > 0).astype(float)

    keep = [
        "证券代码",
        "时间",
        "PPE",
        "CIP",
        "Provision",
        "Outcap",
        "PPE_observed",
        "CIP_observed",
        "Provision_observed",
        "Capex_observed",
        "PPE_TA",
        "CIP_TA",
        "Provision_TA",
        "Capex_TA",
        "Provision_D",
        "HighCapex",
        "HighCIP",
        "L_PPE_TA",
        "L_CIP_TA",
        "L_Provision_TA",
        "L_Capex_TA",
        "L_Provision_D",
        "L_Capex_observed",
        "L_CIP_observed",
    ]
    filled[keep].to_csv(OUT / "financial_crosscheck_panel.csv", index=False, encoding="utf-8-sig")

    coverage = pd.DataFrame(
        [
            {
                "n": len(filled),
                "unmatched": int(unmatched),
                "PPE_observed": int(merged["PPE_observed"].sum()),
                "CIP_observed": int(merged["CIP_observed"].sum()),
                "CIP_positive_filled": int((filled["CIP"] > 0).sum()),
                "Provision_observed": int(merged["Provision_observed"].sum()),
                "Provision_D": int(filled["Provision_D"].sum()),
                "Capex_observed": int(merged["Capex_observed"].sum()),
                "HighCapex": int(filled["HighCapex"].sum()),
                "HighCIP": int(filled["HighCIP"].sum()),
                "Capex_TA_median": float(filled["Capex_TA"].median()),
                "CIP_TA_median": float(filled["CIP_TA"].median()),
                "PPE_TA_median": float(filled["PPE_TA"].median()),
                "corr_Capex_CIP": float(filled["Capex_TA"].corr(filled["CIP_TA"])),
                "corr_Capex_PPE": float(filled["Capex_TA"].corr(filled["PPE_TA"])),
                "corr_Provision_D_TA": float(filled["Provision_D"].corr(filled["Provision_TA"])),
                "L_Capex_nonmissing_raw": int(filled["L_Capex_observed"].sum()),
            }
        ]
    )
    coverage.to_csv(OUT / "financial_crosscheck_coverage.csv", index=False, encoding="utf-8-sig")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
