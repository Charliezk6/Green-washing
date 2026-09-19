from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/robustness_config.json").read_text(encoding="utf-8"))
INPUT = (ROOT / CONFIG["input_panel"]).resolve()
OUT = ROOT / "outputs"
DEP = CONFIG["dependent"]
FIRM = CONFIG["firm_id"]
YEAR = CONFIG["year"]
INDUSTRY = CONFIG["industry"]
CONTROLS = CONFIG["controls"]
CORE = ["IMS", "IMS_D"]


def load() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig", low_memory=False)
    required = [FIRM, YEAR, INDUSTRY, DEP, *CONTROLS, *CORE]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    for column in required:
        if column not in {FIRM, INDUSTRY}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    threshold = float(CONFIG.get("ims_threshold", 0.20))
    if "IMS" in data.columns and "IMS_D" in data.columns:
        share = data["IMS"]
        data["IMS_D"] = (share >= threshold).astype(float)
        data.loc[share.isna(), "IMS_D"] = float("nan")
    return data


def model_sample(data: pd.DataFrame, x: str) -> pd.DataFrame:
    return data.dropna(subset=[DEP, x, FIRM, YEAR, INDUSTRY, *CONTROLS]).copy()


def fit_with_covariance(model, covariance, x: str, sample: pd.DataFrame, covariance_label: str) -> pd.DataFrame:
    parameters = model.params
    diagonal = covariance.diagonal().copy()
    diagonal[diagonal < 0] = float("nan")
    standard_errors = pd.Series(diagonal ** 0.5, index=parameters.index)
    t_values = parameters / standard_errors
    p_values = pd.Series(2 * norm.sf(t_values.abs()), index=parameters.index)
    rows = []
    for variable in [x, *CONTROLS]:
        parameter_key = f"Q('{variable}')"
        if parameter_key not in parameters.index:
            continue
        rows.append({
            "model": f"{x}_{covariance_label}",
            "variable": variable,
            "coefficient": parameters[parameter_key],
            "std_error": standard_errors[parameter_key],
            "t_value": t_values[parameter_key],
            "p_value": p_values[parameter_key],
            "n": int(model.nobs),
            "firms": sample[FIRM].nunique(),
            "industries": sample[INDUSTRY].nunique(),
            "years": sample[YEAR].nunique(),
            "fixed_effects": "industry_year",
            "cluster": covariance_label,
        })
    return pd.DataFrame(rows)


def fit(data: pd.DataFrame, x: str, covariance_label: str) -> tuple[pd.DataFrame, dict]:
    sample = model_sample(data, x)
    formula = f"Q('{DEP}') ~ Q('{x}') + " + " + ".join(f"Q('{control}')" for control in CONTROLS)
    formula += f" + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"
    result = smf.ols(formula, data=sample).fit()
    if covariance_label == "firm_clustered":
        covariance = cov_cluster(result, sample[FIRM])
    elif covariance_label == "industry_clustered":
        covariance = cov_cluster(result, sample[INDUSTRY])
    elif covariance_label == "industry_year_clustered":
        industry_codes = pd.Categorical(sample[INDUSTRY]).codes
        year_codes = pd.Categorical(sample[YEAR]).codes
        covariance = cov_cluster_2groups(result, industry_codes, year_codes)[0]
    else:
        raise ValueError(f"Unknown covariance: {covariance_label}")
    rows = fit_with_covariance(result, covariance, x, sample, covariance_label)
    info = {
        "model": f"{x}_{covariance_label}",
        "n": int(result.nobs),
        "firms": sample[FIRM].nunique(),
        "industries": sample[INDUSTRY].nunique(),
        "years": sample[YEAR].nunique(),
        "fixed_effects": "industry_year",
        "cluster": covariance_label,
    }
    return rows, info


def save_diagnostics(data: pd.DataFrame) -> None:
    numeric = [*CORE, *CONTROLS]
    summary = data[numeric].describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]).T
    summary["missing"] = data[numeric].isna().sum()
    summary["missing_rate"] = data[numeric].isna().mean()
    summary.to_csv(OUT / "diagnostics/variable_summary.csv", encoding="utf-8-sig")
    duplicate_count = int(data.duplicated([FIRM, YEAR]).sum())
    pd.DataFrame([{
        "raw_rows": len(data),
        "company_year_duplicates": duplicate_count,
        "firms": data[FIRM].nunique(),
        "industries": data[INDUSTRY].nunique(),
        "years": data[YEAR].nunique(),
    }]).to_csv(OUT / "diagnostics/data_quality.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    for folder in [OUT / "coefficients", OUT / "diagnostics", OUT / "tables", OUT / "logs"]:
        folder.mkdir(parents=True, exist_ok=True)
    data = load()
    save_diagnostics(data)
    all_rows, samples = [], []
    covariances = ["firm_clustered", "industry_clustered", "industry_year_clustered"]
    for x in CORE:
        for covariance_label in covariances:
            rows, info = fit(data, x, covariance_label)
            all_rows.append(rows)
            samples.append(info)
    pd.concat(all_rows, ignore_index=True).to_csv(OUT / "coefficients/robustness_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(OUT / "diagnostics/robustness_sample_info.csv", index=False, encoding="utf-8-sig")
    metadata = pd.DataFrame({
        "item": ["input_panel", "raw_rows", "firms", "industries", "years", "fixed_effects", "covariances"],
        "value": [str(INPUT), len(data), data[FIRM].nunique(), data[INDUSTRY].nunique(), data[YEAR].nunique(), "industry + year", ";".join(covariances)],
    })
    metadata.to_csv(OUT / "diagnostics/input_metadata.csv", index=False, encoding="utf-8-sig")
    (OUT / "logs/run_summary.txt").write_text(
        f"input={INPUT}\nrows={len(data)}\nmodels={len(samples)}\nfixed_effects=industry_year\ncovariances={covariances}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
