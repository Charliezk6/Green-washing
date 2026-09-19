from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT / "config/loo_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
INPUT = (ROOT / CONFIG["input_panel"]).resolve()
MARKET_INPUT = (ROOT / CONFIG["market_input"]).resolve()
OUTPUT = PROJECT / "outputs"
DERIVED = PROJECT / "data/derived"
DEP = CONFIG["dependent"]
FIRM = CONFIG["firm_id"]
YEAR = CONFIG["year"]
INDUSTRY = CONFIG["industry"]
AUDITOR_ID = CONFIG["auditor_id"]
AUDITOR_NAME = CONFIG["auditor_name"]
REVENUE = CONFIG["revenue"]
CORE = ["IMS_LOO", "IMS_LOO_D"]
CONTROLS = CONFIG["controls"]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str, AUDITOR_ID: str, AUDITOR_NAME: str}, encoding="utf-8-sig", low_memory=False)
    required = [FIRM, YEAR, INDUSTRY, DEP, REVENUE, AUDITOR_ID, *CONTROLS]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    for column in [YEAR, REVENUE, DEP, *CONTROLS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[AUDITOR_ID] = data[AUDITOR_ID].map(normalize_identifier).astype("string")
    data[AUDITOR_NAME] = data[AUDITOR_NAME].astype("string")
    return data


def expertise_code(industry_code: object) -> str:
    code = str(industry_code).strip()
    return code if code.startswith("C") else code[:1]


def normalize_identifier(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def load_full_market() -> pd.DataFrame:
    market = pd.read_csv(MARKET_INPUT, dtype={AUDITOR_ID: str}, encoding="utf-8-sig", low_memory=False)
    required = ["证券代码", YEAR, CONFIG["market_industry"], AUDITOR_ID, REVENUE]
    missing = [column for column in required if column not in market.columns]
    if missing:
        raise KeyError(f"Missing market columns: {missing}")
    market[YEAR] = pd.to_numeric(market[YEAR], errors="coerce")
    market[REVENUE] = pd.to_numeric(market[REVENUE], errors="coerce")
    market[AUDITOR_ID] = market[AUDITOR_ID].map(normalize_identifier).astype("string")
    market[INDUSTRY] = market[CONFIG["market_industry"]].map(expertise_code)
    market = market[market[YEAR].isin(CONFIG["years"]) & market[REVENUE].gt(0) & market[INDUSTRY].ne("O")]
    return market.dropna(subset=[AUDITOR_ID, INDUSTRY, YEAR, REVENUE]).copy()


def construct_loo(data: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    market_auditor_total = market.groupby([AUDITOR_ID, INDUSTRY, YEAR])[REVENUE].sum()
    market_industry_total = market.groupby([INDUSTRY, YEAR])[REVENUE].sum()
    auditor_keys = pd.MultiIndex.from_frame(result[[AUDITOR_ID, INDUSTRY, YEAR]])
    industry_keys = pd.MultiIndex.from_frame(result[[INDUSTRY, YEAR]])
    result["_market_auditor_total"] = auditor_keys.map(market_auditor_total)
    result["_market_industry_total"] = industry_keys.map(market_industry_total)
    positive_revenue = result[REVENUE].where(result[REVENUE].gt(0))
    result["IMS_LOO_NUMERATOR"] = result["_market_auditor_total"] - positive_revenue
    result["IMS_LOO_DENOMINATOR"] = result["_market_industry_total"] - positive_revenue
    valid_keys = result[[AUDITOR_ID, INDUSTRY, YEAR, REVENUE]].notna().all(axis=1)
    valid = valid_keys & result["IMS_LOO_NUMERATOR"].ge(0) & result["IMS_LOO_DENOMINATOR"].gt(0)
    result["IMS_LOO"] = (result["IMS_LOO_NUMERATOR"] / result["IMS_LOO_DENOMINATOR"]).where(valid)
    result["IMS_LOO_D"] = (result["IMS_LOO"] >= CONFIG["ims_threshold"]).astype(float).where(result["IMS_LOO"].notna())
    result["LOO_VALID"] = valid
    result["LOO_MISSING_REASON"] = np.select(
        [valid_keys.eq(False), valid_keys & result["_market_auditor_total"].isna(), valid_keys & result["IMS_LOO_DENOMINATOR"].le(0), valid_keys & result["IMS_LOO_NUMERATOR"].lt(0)],
        ["missing_focal_key_or_revenue", "focal_key_not_found_in_full_market", "nonpositive_leave_one_out_denominator", "focal_revenue_exceeds_full_market_auditor_total"],
        default="valid",
    )
    return result.drop(columns=["_market_auditor_total", "_market_industry_total"])


def common_sample(data: pd.DataFrame, x: str) -> pd.DataFrame:
    required = [DEP, x, FIRM, YEAR, INDUSTRY, *CONTROLS]
    return data.dropna(subset=required).copy()


def fit(data: pd.DataFrame, x: str) -> tuple[pd.DataFrame, dict]:
    sample = common_sample(data, x)
    formula = f"Q('{DEP}') ~ Q('{x}') + " + " + ".join(f"Q('{control}')" for control in CONTROLS)
    formula += f" + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"
    result = smf.ols(formula, data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
    rows = []
    for variable in [x, *CONTROLS]:
        parameter = f"Q('{variable}')"
        if parameter not in result.params.index:
            continue
        rows.append({
            "model": f"{x}_industry_year_fe",
            "variable": variable,
            "coefficient": result.params[parameter],
            "std_error": result.bse[parameter],
            "t_value": result.tvalues[parameter],
            "p_value": result.pvalues[parameter],
            "n": int(result.nobs),
            "firms": sample[FIRM].nunique(),
            "industries": sample[INDUSTRY].nunique(),
            "years": sample[YEAR].nunique(),
            "fixed_effects": "industry_year",
            "cluster": "firm",
        })
    return pd.DataFrame(rows), {
        "model": f"{x}_industry_year_fe",
        "n": int(result.nobs),
        "firms": sample[FIRM].nunique(),
        "industries": sample[INDUSTRY].nunique(),
        "years": sample[YEAR].nunique(),
        "fixed_effects": "industry_year",
        "cluster": "firm",
    }


def write_diagnostics(data: pd.DataFrame) -> None:
    diagnostics = OUTPUT / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    duplicate_count = int(data.duplicated([FIRM, YEAR]).sum())
    variable_summary = data[["IMS", "IMS_LOO", "IMS_D", "IMS_LOO_D"]].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    variable_summary["missing"] = data[["IMS", "IMS_LOO", "IMS_D", "IMS_LOO_D"]].isna().sum()
    variable_summary["missing_rate"] = data[["IMS", "IMS_LOO", "IMS_D", "IMS_LOO_D"]].isna().mean()
    variable_summary.to_csv(diagnostics / "variable_summary.csv", encoding="utf-8-sig")
    pd.DataFrame([{
        "raw_rows": len(data),
        "company_year_duplicates": duplicate_count,
        "firms": data[FIRM].nunique(),
        "industries": data[INDUSTRY].nunique(),
        "years": data[YEAR].nunique(),
        "auditor_ids": data[AUDITOR_ID].nunique(),
        "auditor_names": data[AUDITOR_NAME].nunique(),
        "loo_valid_rows": int(data["IMS_LOO"].notna().sum()),
    }]).to_csv(diagnostics / "data_quality.csv", index=False, encoding="utf-8-sig")
    data["LOO_MISSING_REASON"].value_counts(dropna=False).rename_axis("reason").reset_index(name="observations").to_csv(diagnostics / "missing_reasons.csv", index=False, encoding="utf-8-sig")
    data[["IMS", "IMS_LOO", "IMS_D", "IMS_LOO_D"]].corr().to_csv(diagnostics / "correlation.csv", encoding="utf-8-sig")
    within = data.groupby(FIRM)[["IMS", "IMS_LOO", "IMS_D", "IMS_LOO_D"]].agg(["count", "nunique", "std"])
    rows = []
    for variable in ["IMS", "IMS_LOO", "IMS_D", "IMS_LOO_D"]:
        rows.append({
            "variable": variable,
            "firms_with_observations": int(within[(variable, "count")].gt(0).sum()),
            "firms_with_within_variation": int(within[(variable, "nunique")].gt(1).sum()),
            "mean_within_std": within[(variable, "std")].mean(),
        })
    pd.DataFrame(rows).to_csv(diagnostics / "within_firm.csv", index=False, encoding="utf-8-sig")
    # Compare the existing IMS with the same-sample raw market-share formula as a construction audit.
    revenue = data[REVENUE].where(data[REVENUE].gt(0))
    auditor_total = revenue.groupby([data[AUDITOR_ID], data[INDUSTRY], data[YEAR]], dropna=False).transform("sum")
    industry_total = revenue.groupby([data[INDUSTRY], data[YEAR]], dropna=False).transform("sum")
    raw_formula = auditor_total / industry_total
    audit = pd.DataFrame({
        "existing_IMS": data["IMS"],
        "raw_formula_IMS": raw_formula,
        "absolute_difference": (data["IMS"] - raw_formula).abs(),
    })
    pd.DataFrame([{
        "comparable_rows": int(audit.dropna().shape[0]),
        "correlation_existing_vs_raw_formula": audit["existing_IMS"].corr(audit["raw_formula_IMS"]),
        "mean_absolute_difference": audit["absolute_difference"].mean(),
        "share_exact_within_1e-8": float(audit["absolute_difference"].le(1e-8).mean()),
    }]).to_csv(diagnostics / "ims_construction_audit.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    for folder in [OUTPUT / "coefficients", OUTPUT / "diagnostics", OUTPUT / "logs", DERIVED]:
        folder.mkdir(parents=True, exist_ok=True)
    data = construct_loo(load_data(), load_full_market())
    data.to_csv(DERIVED / "ims_loo_panel_corrected.csv", index=False, encoding="utf-8-sig")
    data.to_csv(DERIVED / "ims_loo_panel.csv", index=False, encoding="utf-8-sig")
    write_diagnostics(data)
    rows, samples = [], []
    for x in CORE:
        result, info = fit(data, x)
        rows.append(result)
        samples.append(info)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT / "coefficients/loo_regression_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(OUTPUT / "diagnostics/loo_regression_sample_info.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "logs/run_summary.txt").write_text(
        f"input={INPUT}\nrows={len(data)}\nloo_definition=exclude_focal_firm_from_auditor_industry_year_numerator_and_industry_year_denominator\nauditor_identifier={AUDITOR_ID}\nmodels={len(samples)}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
