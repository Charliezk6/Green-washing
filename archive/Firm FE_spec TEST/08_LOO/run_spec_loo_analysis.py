"""Leave-one-out robustness for SPEC.

SPEC_LOO = 1{A_{a,k,t} - Fee_i > 0 and A_{a,k,t} - Fee_i >= max_{b != a} A_{b,k,t}},
i.e. after removing the focal client's own audit fee, its audit firm is still
the audit-fee share leader of the industry-year. SPEC_LOO is built in
test/00_build_spec/build_spec.py from the same full-market data as SPEC.

Regression specification follows validation/P1/LOO/run_loo_analysis.py:
all controls, year and industry FE, firm-clustered SE. The binary parts of
validation/P1/LOO/diagnose_loo_insignificance.py (transition crosstab and the
BOTH/LOST split, disclosure/performance split) are reproduced for SPEC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT = Path(__file__).resolve().parent
TEST_ROOT = PROJECT.parent
REPO_ROOT = TEST_ROOT.parent
CONFIG = json.loads((REPO_ROOT / "validation/P1/LOO/config/loo_config.json").read_text(encoding="utf-8"))
INPUT = TEST_ROOT / "data/combined_regression_panel_15381_spec.csv"
CELLS = TEST_ROOT / "data/spec_fee_share_auditor_industry_year.csv"
OUTPUT = PROJECT / "outputs"
DEP = CONFIG["dependent"]
FIRM = CONFIG["firm_id"]
YEAR = CONFIG["year"]
INDUSTRY = CONFIG["industry"]
CONTROLS = CONFIG["controls"]
GI_Z, ESGE_Z = "GI_z_行业年度", "ESGE_z_行业年度"
CORE = ["SPEC_LOO"]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str, "SPEC_AuditorID": str}, encoding="utf-8-sig", low_memory=False)
    for column in [YEAR, DEP, GI_Z, ESGE_Z, "SPEC", "SPEC_LOO", "SPEC_Fee", "FeeShare", *CONTROLS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    cells = pd.read_csv(CELLS, dtype={"SPEC_AuditorID": str}, encoding="utf-8-sig")
    data = data.merge(
        cells[[YEAR, INDUSTRY, "SPEC_AuditorID", "ClientCount", "AuditorIndustryFee"]],
        on=[YEAR, INDUSTRY, "SPEC_AuditorID"], how="left", validate="many_to_one",
    )
    data["SINGLE_CLIENT"] = data["ClientCount"].eq(1).astype(float).where(data["ClientCount"].notna())
    data["SPEC_BOTH"] = (data["SPEC"].eq(1) & data["SPEC_LOO"].eq(1)).astype(float).where(data["SPEC_LOO"].notna())
    data["SPEC_LOST"] = (data["SPEC"].eq(1) & data["SPEC_LOO"].eq(0)).astype(float).where(data["SPEC_LOO"].notna())
    # Removing the focal fee can only shrink its own auditor's total, so SPEC=0 never becomes SPEC_LOO=1.
    data["LOO_REASON"] = np.select(
        [data["SPEC_LOO"].isna(), data["SPEC_BOTH"].eq(1), data["SPEC_LOST"].eq(1) & data["SINGLE_CLIENT"].eq(1),
         data["SPEC_LOST"].eq(1)],
        ["missing", "leader_both", "lost_single_client_cell", "lost_rival_overtakes"],
        default="non_leader_both",
    )
    return data


def common_sample(data: pd.DataFrame, x: str) -> pd.DataFrame:
    required = [DEP, x, FIRM, YEAR, INDUSTRY, *CONTROLS]
    return data.dropna(subset=required).copy()


def fit(data: pd.DataFrame, xs: list[str], model: str, dep: str = DEP) -> tuple[pd.DataFrame, dict]:
    sample = data.dropna(subset=[dep, *xs, FIRM, YEAR, INDUSTRY, *CONTROLS]).copy()
    formula = f"Q('{dep}') ~ " + " + ".join(f"Q('{x}')" for x in xs) + " + " + " + ".join(f"Q('{control}')" for control in CONTROLS)
    formula += f" + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"
    result = smf.ols(formula, data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
    rows = []
    for variable in [*xs, *CONTROLS]:
        parameter = f"Q('{variable}')"
        if parameter not in result.params.index:
            continue
        rows.append({
            "model": model,
            "dependent": dep,
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
        "model": model, "dependent": dep, "n": int(result.nobs), "firms": sample[FIRM].nunique(),
        "industries": sample[INDUSTRY].nunique(), "years": sample[YEAR].nunique(),
        "fixed_effects": "industry_year", "cluster": "firm",
    }


def write_diagnostics(data: pd.DataFrame) -> None:
    diagnostics = OUTPUT / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    variables = ["SPEC", "SPEC_LOO", "FeeShare"]
    summary = data[variables].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    summary["missing"] = data[variables].isna().sum()
    summary["missing_rate"] = data[variables].isna().mean()
    summary.to_csv(diagnostics / "variable_summary.csv", encoding="utf-8-sig")
    pd.DataFrame([{
        "raw_rows": len(data),
        "company_year_duplicates": int(data.duplicated([FIRM, YEAR]).sum()),
        "firms": data[FIRM].nunique(),
        "industries": data[INDUSTRY].nunique(),
        "years": data[YEAR].nunique(),
        "spec_auditor_ids": data["SPEC_AuditorID"].nunique(),
        "loo_valid_rows": int(data["SPEC_LOO"].notna().sum()),
        "focal_fee_missing_rows": int(data["SPEC_Fee"].isna().sum()),
    }]).to_csv(diagnostics / "data_quality.csv", index=False, encoding="utf-8-sig")
    data["LOO_REASON"].value_counts(dropna=False).rename_axis("reason").reset_index(name="observations").to_csv(
        diagnostics / "loo_transition_reasons.csv", index=False, encoding="utf-8-sig")
    data[variables].corr().to_csv(diagnostics / "correlation.csv", encoding="utf-8-sig")
    within = data.groupby(FIRM)[["SPEC", "SPEC_LOO"]].agg(["count", "nunique", "std"])
    pd.DataFrame([{
        "variable": variable,
        "firms_with_observations": int(within[(variable, "count")].gt(0).sum()),
        "firms_with_within_variation": int(within[(variable, "nunique")].gt(1).sum()),
        "mean_within_std": within[(variable, "std")].mean(),
    } for variable in ["SPEC", "SPEC_LOO"]]).to_csv(diagnostics / "within_firm.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(data["SPEC"], data["SPEC_LOO"], margins=True).to_csv(diagnostics / "binary_transition_crosstab.csv", encoding="utf-8-sig")


def main() -> None:
    for folder in [OUTPUT / "coefficients", OUTPUT / "diagnostics", OUTPUT / "logs", PROJECT / "data/derived"]:
        folder.mkdir(parents=True, exist_ok=True)
    data = load_data()
    data.to_csv(PROJECT / "data/derived/spec_loo_panel.csv", index=False, encoding="utf-8-sig")
    write_diagnostics(data)

    rows, samples = [], []
    for x in CORE:
        result, info = fit(data, [x], f"{x}_industry_year_fe")
        rows.append(result)
        samples.append(info)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT / "coefficients/spec_loo_regression_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(OUTPUT / "diagnostics/spec_loo_regression_sample_info.csv", index=False, encoding="utf-8-sig")

    diagnostic_specs = [
        ("A1", "Baseline SPEC, full sample", ["SPEC"], DEP),
        ("A2", "SPEC_LOO", ["SPEC_LOO"], DEP),
        ("D1", "SPEC transition groups (reference: SPEC = 0)", ["SPEC_BOTH", "SPEC_LOST"], DEP),
        ("E_GI_z_1", "SPEC on disclosure GI_z", ["SPEC"], GI_Z),
        ("E_GI_z_2", "SPEC_LOO on disclosure GI_z", ["SPEC_LOO"], GI_Z),
        ("E_ESGE_z_1", "SPEC on E performance ESGE_z", ["SPEC"], ESGE_Z),
        ("E_ESGE_z_2", "SPEC_LOO on E performance ESGE_z", ["SPEC_LOO"], ESGE_Z),
    ]
    diagnostic_rows = []
    for spec, description, xs, dep in diagnostic_specs:
        result, _ = fit(data, xs, spec, dep)
        result = result[result["variable"].isin(xs)].assign(description=description)
        diagnostic_rows.append(result)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    diagnostics.to_csv(OUTPUT / "diagnostics/binary_diagnostic_regressions.csv", index=False, encoding="utf-8-sig")

    (OUTPUT / "logs/run_summary.txt").write_text(
        f"input={INPUT}\nrows={len(data)}\nloo_definition=exclude_focal_client_fee_from_its_auditor_industry_year_total_then_check_leadership\n"
        f"auditor_identifier=SPEC_AuditorID (full-market file)\nmodels={len(samples)}\n",
        encoding="utf-8",
    )
    print(pd.concat(rows)[lambda f: f["variable"].isin(CORE)].to_string(index=False))
    print(diagnostics[["model", "dependent", "variable", "coefficient", "std_error", "p_value", "n"]].to_string(index=False))
    print(pd.crosstab(data["SPEC"], data["SPEC_LOO"], margins=True))
    print(data["LOO_REASON"].value_counts())


if __name__ == "__main__":
    main()
