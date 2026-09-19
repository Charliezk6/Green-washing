"""Baseline-style interactions of IMS with auditable-claim and hard-info moderators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
REGRESSION = ROOT / "Regression"
sys.path.insert(0, str(REGRESSION / "scripts"))
from reproduce_regression import ALL_CONTROLS, DEPENDENT, WINSORIZED_CONTROLS, load_data, winsorize_controls

OUT = Path(__file__).resolve().parents[1]
DERIVED = OUT / "results/derived"
TABLES = OUT / "results/tables"
TABLES.mkdir(parents=True, exist_ok=True)

MECHANISM = DERIVED / "mechanism_firm_year_panel.csv"
CONTROLS = ALL_CONTROLS
SHARE_VARS = [
    "AuditIntersect_Share",
    "Soft_Share",
    "Equipment_Share",
    "Capex_Share",
    "PenaltyClaim_Share",
    "Remediation_Share",
]


def stars(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def tidy(result, name: str, variables: list[str], extra: dict | None = None) -> pd.DataFrame:
    rows = []
    extra = extra or {}
    for variable in variables:
        if variable not in result.params.index:
            continue
        interval = result.conf_int().loc[variable]
        rows.append(
            {
                "model": name,
                "variable": variable,
                "coefficient": result.params[variable],
                "std_error": result.bse[variable],
                "t": result.tvalues[variable],
                "p_value": result.pvalues[variable],
                "stars": stars(result.pvalues[variable]),
                "ci_low": interval[0],
                "ci_high": interval[1],
                "n": int(result.nobs),
                "adjusted_r2": result.rsquared_adj,
                "n_firms": extra.get("n_firms"),
                "notes": extra.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def fit_ols(sample: pd.DataFrame, formula: str, name: str, focus: list[str], notes: str = "") -> tuple:
    model = smf.ols(formula, data=sample)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": sample["证券代码"]})
    extra = {"n_firms": int(sample["证券代码"].nunique()), "notes": notes}
    return result, tidy(result, name, focus, extra)


def industry_year_fe(formula_rhs: str) -> str:
    return formula_rhs + " + C(时间):C(ExpertiseIndustryCode)"


def main() -> None:
    baseline = load_data()
    required = [DEPENDENT, "IMS", "IMS_D", "证券代码", "时间", "ExpertiseIndustryCode", *CONTROLS]
    common = baseline.dropna(subset=required).copy()
    common, _ = winsorize_controls(common)
    common["证券代码"] = common["证券代码"].astype(str).str.zfill(6)
    if len(common) != 15381:
        raise ValueError(f"Expected 15,381 common sample, got {len(common)}")

    mech = pd.read_csv(MECHANISM, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    mech["证券代码"] = mech["证券代码"].astype(str).str.zfill(6)
    keep = [
        "证券代码",
        "时间",
        "GI",
        "AuditIntersect_Share",
        "Intersect_Count",
        "EnvSentence_Count",
        "LnIntersect_Count",
        "HighIntersect",
        "Soft_Share",
        "Equipment_Share",
        "Capex_Share",
        "PenaltyClaim_Share",
        "Remediation_Share",
        "KeyPollMonUnit",
        "Penalty_D",
        "EnvInvest_D",
        "LnEnvInvest",
        "LnPenaltyAmt",
    ]
    merged = common.merge(mech[keep], on=["证券代码", "时间"], how="left", suffixes=("", "_mech"))
    if "GI_mech" in merged.columns:
        merged["GI"] = merged["GI"].fillna(merged["GI_mech"])
        merged = merged.drop(columns=["GI_mech"])

    merged = merged.sort_values(["证券代码", "时间"])
    previous_year = merged.groupby("证券代码")["时间"].shift(1)
    consecutive = merged["时间"].sub(previous_year).eq(1)
    for column in SHARE_VARS + ["HighIntersect", "KeyPollMonUnit", "Penalty_D", "LnIntersect_Count", "EnvSentence_Count"]:
        merged[f"L_{column}"] = merged.groupby("证券代码")[column].shift(1).where(consecutive)

    merged.to_csv(DERIVED / "regression_analysis_panel.csv", index=False, encoding="utf-8-sig")

    corr = merged[["AuditIntersect_Share", "GI", "Soft_Share", "Greenwashing连续_基准", "IMS"]].corr()
    corr.to_csv(OUT / "results/validation/gi_intersect_correlation.csv", encoding="utf-8-sig")
    gi_corr = float(corr.loc["AuditIntersect_Share", "GI"])

    rhs_controls = " + ".join(CONTROLS)
    y = f"Q('{DEPENDENT}')"
    rows = []
    focus_share = ["IMS", "AuditIntersect_Share", "IMS:AuditIntersect_Share"]
    sample = merged.dropna(subset=[DEPENDENT, "IMS", "AuditIntersect_Share", *CONTROLS, "ExpertiseIndustryCode"]).copy()
    formula = f"{y} ~ IMS + AuditIntersect_Share + IMS:AuditIntersect_Share + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
    result, tidy_row = fit_ols(sample, formula, "IMS x AuditIntersect_Share", focus_share, "main composition")
    rows.append(tidy_row)

    formula_d = f"{y} ~ IMS_D + AuditIntersect_Share + IMS_D:AuditIntersect_Share + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
    _, tidy_row = fit_ols(sample, formula_d, "IMS_D x AuditIntersect_Share", ["IMS_D", "AuditIntersect_Share", "IMS_D:AuditIntersect_Share"])
    rows.append(tidy_row)

    high_sample = sample.dropna(subset=["HighIntersect"])
    formula_h = f"{y} ~ IMS + HighIntersect + IMS:HighIntersect + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
    _, tidy_row = fit_ols(high_sample, formula_h, "IMS x HighIntersect", ["IMS", "HighIntersect", "IMS:HighIntersect"], "display split")
    rows.append(tidy_row)
    formula_hd = f"{y} ~ IMS_D + HighIntersect + IMS_D:HighIntersect + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
    _, tidy_row = fit_ols(high_sample, formula_hd, "IMS_D x HighIntersect", ["IMS_D", "HighIntersect", "IMS_D:HighIntersect"])
    rows.append(tidy_row)

    for label, subset in [("HighIntersect=1", high_sample[high_sample["HighIntersect"].eq(1)]), ("HighIntersect=0", high_sample[high_sample["HighIntersect"].eq(0)])]:
        formula_g = f"{y} ~ IMS + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
        _, tidy_row = fit_ols(subset, formula_g, f"IMS subsample {label}", ["IMS"], "display only")
        rows.append(tidy_row)

    for core, moderator, label in [
        ("IMS", "KeyPollMonUnit", "IMS x KeyPollMonUnit"),
        ("IMS_D", "KeyPollMonUnit", "IMS_D x KeyPollMonUnit"),
        ("IMS", "Penalty_D", "IMS x Penalty_D"),
        ("IMS_D", "Penalty_D", "IMS_D x Penalty_D"),
        ("IMS", "Soft_Share", "IMS x Soft_Share placebo"),
        ("IMS", "LnIntersect_Count", "IMS x ln(1+Intersect_Count)"),
        ("IMS", "Equipment_Share", "IMS x Equipment_Share"),
        ("IMS", "Capex_Share", "IMS x Capex_Share"),
        ("IMS", "PenaltyClaim_Share", "IMS x PenaltyClaim_Share"),
        ("IMS", "Remediation_Share", "IMS x Remediation_Share"),
    ]:
        spec = merged.dropna(subset=[DEPENDENT, core, moderator, *CONTROLS, "ExpertiseIndustryCode"]).copy()
        formula_m = f"{y} ~ {core} + {moderator} + {core}:{moderator} + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
        _, tidy_row = fit_ols(spec, formula_m, label, [core, moderator, f"{core}:{moderator}"])
        rows.append(tidy_row)

    poll = merged.dropna(subset=[DEPENDENT, "IMS", "KeyPollMonUnit", *CONTROLS, "ExpertiseIndustryCode"])
    for label, subset in [("KeyPollMonUnit=1", poll[poll["KeyPollMonUnit"].eq(1)]), ("KeyPollMonUnit=0", poll[poll["KeyPollMonUnit"].eq(0)])]:
        formula_g = f"{y} ~ IMS + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
        _, tidy_row = fit_ols(subset, formula_g, f"IMS subsample {label}", ["IMS"], "display only")
        rows.append(tidy_row)

    formula_kfe = industry_year_fe(f"{y} ~ IMS + KeyPollMonUnit + IMS:KeyPollMonUnit + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)")
    _, tidy_row = fit_ols(poll, formula_kfe, "IMS x KeyPollMonUnit + industry-year FE", ["IMS", "KeyPollMonUnit", "IMS:KeyPollMonUnit"], "industry-year FE")
    rows.append(tidy_row)

    # Share regression controlling for opportunity set size.
    spec = sample.dropna(subset=["EnvSentence_Count"])
    formula_c = f"{y} ~ IMS + AuditIntersect_Share + IMS:AuditIntersect_Share + EnvSentence_Count + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
    _, tidy_row = fit_ols(spec, formula_c, "IMS x Share | EnvSentence_Count", focus_share, "opportunity-set control")
    rows.append(tidy_row)

    lagged = merged.dropna(subset=[DEPENDENT, "IMS", "L_AuditIntersect_Share", *CONTROLS, "ExpertiseIndustryCode"]).copy()
    formula_l = f"{y} ~ IMS + L_AuditIntersect_Share + IMS:L_AuditIntersect_Share + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
    _, tidy_row = fit_ols(lagged, formula_l, "IMS x L_AuditIntersect_Share", ["IMS", "L_AuditIntersect_Share", "IMS:L_AuditIntersect_Share"], "lagged V")
    rows.append(tidy_row)

    formula_ixy = industry_year_fe(f"{y} ~ IMS + AuditIntersect_Share + IMS:AuditIntersect_Share + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)")
    _, tidy_row = fit_ols(sample, formula_ixy, "IMS x Share + industry-year FE", focus_share, "industry-year FE")
    rows.append(tidy_row)

    no_penalty_year = sample[sample["Penalty_D"].fillna(0).eq(0)]
    _, tidy_row = fit_ols(no_penalty_year, formula, "IMS x Share drop penalty years", focus_share, "drop Penalty_D=1")
    rows.append(tidy_row)

    if abs(gi_corr) >= 0.60:
        q1, q3 = sample["GI"].quantile([0.25, 0.75])
        close = sample[sample["GI"].between(q1, q3)]
        _, tidy_row = fit_ols(close, formula, "IMS x Share in GI IQR", focus_share, f"GI-IQR because corr={gi_corr:.3f}")
        rows.append(tidy_row)

    invest = merged.dropna(subset=[DEPENDENT, "IMS", "EnvInvest_D", *CONTROLS, "ExpertiseIndustryCode"])
    formula_inv = f"{y} ~ IMS + EnvInvest_D + IMS:EnvInvest_D + {rhs_controls} + C(时间) + C(ExpertiseIndustryCode)"
    _, tidy_row = fit_ols(invest, formula_inv, "IMS x EnvInvest_D subsample-style", ["IMS", "EnvInvest_D", "IMS:EnvInvest_D"], "investment disclosed")
    rows.append(tidy_row)

    coefficients = pd.concat(rows, ignore_index=True)
    coefficients.to_csv(TABLES / "heterogeneity_coefficients.csv", index=False, encoding="utf-8-sig")
    print(coefficients[coefficients["variable"].str.contains("IMS")].to_string(index=False))
    print("GI vs AuditIntersect_Share correlation:", gi_corr)


if __name__ == "__main__":
    main()
