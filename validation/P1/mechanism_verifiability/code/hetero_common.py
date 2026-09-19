"""Shared helpers for KeyPoll and financial-side heterogeneity regressions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[4]
REGRESSION = ROOT / "Regression"
sys.path.insert(0, str(REGRESSION / "scripts"))
from reproduce_regression import ALL_CONTROLS, DEPENDENT, WINSORIZED_CONTROLS, load_data, winsorize_controls

OUT = Path(__file__).resolve().parents[1]
DERIVED = OUT / "results/derived"
TABLES = OUT / "results/tables"
TABLES.mkdir(parents=True, exist_ok=True)

CONTROLS = ALL_CONTROLS
RATIO_WINSOR = ["Capex_TA", "CIP_TA", "Provision_TA", "PPE_TA", "L_Capex_TA", "L_CIP_TA"]
Y = f"Q('{DEPENDENT}')"


def stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def pad_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return str(int(float(text))).zfill(6)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(6) if digits else text.zfill(6)


def load_analysis_panel() -> pd.DataFrame:
    baseline = load_data()
    required = [DEPENDENT, "IMS", "IMS_D", "证券代码", "时间", "ExpertiseIndustryCode", *CONTROLS]
    common = baseline.dropna(subset=required).copy()
    common, _ = winsorize_controls(common)
    if len(common) != 15381:
        raise ValueError(f"Expected 15,381 common sample, got {len(common)}")
    common["证券代码"] = common["证券代码"].map(pad_code)
    common["时间"] = pd.to_numeric(common["时间"], errors="coerce")

    mech = pd.read_csv(DERIVED / "mechanism_firm_year_panel.csv", dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    mech["证券代码"] = mech["证券代码"].map(pad_code)
    mech["时间"] = pd.to_numeric(mech["时间"], errors="coerce")
    hard = mech[["证券代码", "时间", "KeyPollMonUnit", "Penalty_D"]].copy()
    hard["KeyPollMonUnit"] = pd.to_numeric(hard["KeyPollMonUnit"], errors="coerce")
    hard["Penalty_D"] = pd.to_numeric(hard["Penalty_D"], errors="coerce")

    fin = pd.read_csv(DERIVED / "financial_crosscheck_panel.csv", dtype={"证券代码": str}, encoding="utf-8-sig")
    fin["证券代码"] = fin["证券代码"].map(pad_code)
    fin["时间"] = pd.to_numeric(fin["时间"], errors="coerce")

    merged = common.merge(hard, on=["证券代码", "时间"], how="left")
    merged = merged.merge(fin, on=["证券代码", "时间"], how="left", validate="one_to_one")

    extra = [column for column in RATIO_WINSOR if column in merged.columns]
    for column in extra:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
        lower, upper = merged[column].quantile([0.01, 0.99])
        merged[column] = merged[column].clip(lower=lower, upper=upper)

    merged = merged.sort_values(["证券代码", "时间"])
    previous_year = merged.groupby("证券代码")["时间"].shift(1)
    consecutive = merged["时间"].sub(previous_year).eq(1)
    merged["L_KeyPollMonUnit"] = merged.groupby("证券代码")["KeyPollMonUnit"].shift(1).where(consecutive)
    return merged.replace([np.inf, -np.inf], np.nan)


def tidy_focus(result, name: str, variables: list[str], sample: pd.DataFrame, notes: str = "") -> pd.DataFrame:
    rows = []
    for variable in variables:
        if variable not in result.params.index:
            continue
        interval = result.conf_int().loc[variable]
        rows.append(
            {
                "model": name,
                "variable": variable,
                "coefficient": float(result.params[variable]),
                "std_error": float(result.bse[variable]),
                "t": float(result.tvalues[variable]),
                "p_value": float(result.pvalues[variable]),
                "stars": stars(result.pvalues[variable]),
                "ci_low": float(interval[0]),
                "ci_high": float(interval[1]),
                "n": int(result.nobs),
                "n_firms": int(sample["证券代码"].nunique()),
                "adjusted_r2": float(result.rsquared_adj),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def fit_cluster(sample: pd.DataFrame, formula: str):
    model = smf.ols(formula, data=sample)
    return model.fit(cov_type="cluster", cov_kwds={"groups": sample["证券代码"]})


def controls_except(moderator: str) -> list[str]:
    return [control for control in CONTROLS if control != moderator]


def rhs_controls(moderator: str | None = None) -> str:
    controls = CONTROLS if moderator is None else controls_except(moderator)
    return " + ".join(controls)


def constrained_formula(core: str, moderator: str, extra_fe: bool = False) -> str:
    formula = f"{Y} ~ {core} + {moderator} + {core}:{moderator} + {rhs_controls(moderator)} + C(时间) + C(ExpertiseIndustryCode)"
    if extra_fe:
        formula += " + C(时间):C(ExpertiseIndustryCode)"
    return formula


def full_interact_formula(core: str, moderator: str, extra_fe: bool = False) -> str:
    terms = [core, moderator, f"{core}:{moderator}"]
    for control in controls_except(moderator):
        terms.extend([control, f"{control}:{moderator}"])
    terms.extend(["C(时间)", "C(ExpertiseIndustryCode)"])
    if extra_fe:
        terms.append("C(时间):C(ExpertiseIndustryCode)")
    return f"{Y} ~ " + " + ".join(terms)


def subsample_formula(core: str) -> str:
    return f"{Y} ~ {core} + {rhs_controls()} + C(时间) + C(ExpertiseIndustryCode)"


def run_three_layer(data: pd.DataFrame, core: str, moderator: str, label: str, notes: str = "") -> pd.DataFrame:
    needed = [DEPENDENT, core, moderator, "证券代码", "时间", "ExpertiseIndustryCode", *CONTROLS]
    sample = data.dropna(subset=needed).copy()
    rows = []
    constrained = fit_cluster(sample, constrained_formula(core, moderator))
    rows.append(tidy_focus(constrained, f"{label} constrained {core} x {moderator}", [core, moderator, f"{core}:{moderator}"], sample, notes or "constrained"))
    full = fit_cluster(sample, full_interact_formula(core, moderator))
    rows.append(tidy_focus(full, f"{label} full {core} x {moderator}", [core, moderator, f"{core}:{moderator}"], sample, "full interact"))
    return pd.concat(rows, ignore_index=True), sample, full


def run_split(data: pd.DataFrame, core: str, splitter: str, label: str) -> pd.DataFrame:
    needed = [DEPENDENT, core, splitter, "证券代码", "时间", "ExpertiseIndustryCode", *CONTROLS]
    sample = data.dropna(subset=needed).copy()
    rows = []
    for value, tag in [(1, "high"), (0, "low")]:
        subset = sample[sample[splitter].eq(value)]
        result = fit_cluster(subset, subsample_formula(core))
        rows.append(tidy_focus(result, f"{label} subsample {splitter}={int(value)} {core}", [core], subset, f"split {tag}"))
    return pd.concat(rows, ignore_index=True)
