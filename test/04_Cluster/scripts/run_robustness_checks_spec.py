"""Treatment-structure clustering for baseline SPEC regressions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy.stats import norm
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "config/robustness_config.json").read_text(encoding="utf-8"))
OUT = ROOT / "outputs"
DEP = CONFIG["dependent"]
FIRM = CONFIG["firm_id"]
YEAR = CONFIG["year"]
INDUSTRY = CONFIG["industry"]
AUDITOR = CONFIG["auditor_id"]
AUDITOR_NAME = CONFIG.get("auditor_name", "AccountingFirmName")
CONTROLS = CONFIG["controls"]
CORE = CONFIG.get("core_variables", ["SPEC"])
N_BOOT = int(CONFIG.get("n_boot", 999))
BOOT_SEED = int(CONFIG.get("boot_seed", 20260921))
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])

ONEWAY = [
    ("firm_clustered", FIRM),
    ("auditor_clustered", "_auditor"),
    ("auditor_industry_clustered", "_auditor_industry"),
    ("industry_clustered", INDUSTRY),
]
TWOWAY = [
    ("firm_auditor_clustered", FIRM, "_auditor"),
    ("auditor_industry_twoway", "_auditor", INDUSTRY),
    ("industry_year_clustered", INDUSTRY, YEAR),
]
WCR_CLUSTERS = [
    ("industry_clustered", INDUSTRY),
    ("auditor_clustered", "_auditor"),
]


def resolve_input(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path
    for base in (ROOT, PROJECT_ROOT):
        candidate = (base / path_str).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find input panel: {path_str}")


INPUT = resolve_input(CONFIG["input_panel"])


def clean_auditor(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def add_cluster_keys(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["_auditor"] = clean_auditor(out[AUDITOR])
    out["_auditor_industry"] = out["_auditor"] + "|" + out[INDUSTRY].astype(str)
    out["_auditor_industry_year"] = out["_auditor_industry"] + "|" + out[YEAR].astype(str)
    return out


def load() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str, AUDITOR: str}, encoding="utf-8-sig", low_memory=False)
    required = [FIRM, YEAR, INDUSTRY, AUDITOR, DEP, *CONTROLS, *CORE]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    for column in required:
        if column not in {FIRM, INDUSTRY, AUDITOR}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return add_cluster_keys(data)


def model_sample(data: pd.DataFrame, x: str) -> pd.DataFrame:
    return data.dropna(subset=[DEP, x, FIRM, YEAR, INDUSTRY, AUDITOR, *CONTROLS]).copy()


def formula_for(x: str) -> str:
    terms = [f"Q('{x}')"] + [f"Q('{control}')" for control in CONTROLS]
    return f"Q('{DEP}') ~ " + " + ".join(terms) + f" + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"


def parameter_key(variable: str) -> str:
    return f"Q('{variable}')"


def fit_with_covariance(
    model,
    covariance,
    x: str,
    sample: pd.DataFrame,
    covariance_label: str,
    n_clusters: int,
    n_clusters_g1: int | None = None,
    n_clusters_g2: int | None = None,
) -> pd.DataFrame:
    parameters = model.params
    diagonal = covariance.diagonal().copy()
    diagonal[diagonal < 0] = float("nan")
    standard_errors = pd.Series(diagonal ** 0.5, index=parameters.index)
    t_values = parameters / standard_errors
    p_values = pd.Series(2 * norm.sf(t_values.abs()), index=parameters.index)
    rows = []
    for variable in [x, *CONTROLS]:
        key = parameter_key(variable)
        if key not in parameters.index:
            continue
        rows.append({
            "model": f"{x}_{covariance_label}",
            "variable": variable,
            "coefficient": parameters[key],
            "std_error": standard_errors[key],
            "t_value": t_values[key],
            "p_value": p_values[key],
            "n": int(model.nobs),
            "firms": sample[FIRM].nunique(),
            "auditors": sample["_auditor"].nunique(),
            "industries": sample[INDUSTRY].nunique(),
            "years": sample[YEAR].nunique(),
            "n_clusters": n_clusters,
            "n_clusters_g1": n_clusters_g1,
            "n_clusters_g2": n_clusters_g2,
            "adjusted_r2": float(model.rsquared_adj),
            "fixed_effects": "industry_year",
            "cluster": covariance_label,
        })
    return pd.DataFrame(rows)


def fit_oneway(result, sample: pd.DataFrame, x: str, label: str, column: str) -> pd.DataFrame:
    groups = sample[column]
    covariance = cov_cluster(result, groups)
    return fit_with_covariance(result, covariance, x, sample, label, int(groups.nunique()))


def fit_twoway(result, sample: pd.DataFrame, x: str, label: str, column1: str, column2: str) -> pd.DataFrame:
    codes1 = pd.factorize(sample[column1], sort=False)[0]
    codes2 = pd.factorize(sample[column2], sort=False)[0]
    covariance = cov_cluster_2groups(result, codes1, codes2)[0]
    g1 = int(pd.Series(sample[column1]).nunique())
    g2 = int(pd.Series(sample[column2]).nunique())
    return fit_with_covariance(
        result, covariance, x, sample, label, g1 * g2, n_clusters_g1=g1, n_clusters_g2=g2
    )


def cluster_size_row(name: str, values: pd.Series, n_obs: int) -> dict:
    sizes = values.value_counts()
    return {
        "cluster": name,
        "n_clusters": int(sizes.size),
        "n_obs": n_obs,
        "min_size": int(sizes.min()),
        "p25_size": float(sizes.quantile(0.25)),
        "median_size": float(sizes.median()),
        "p75_size": float(sizes.quantile(0.75)),
        "max_size": int(sizes.max()),
        "mean_size": float(sizes.mean()),
    }


def save_cluster_diagnostics(data: pd.DataFrame) -> None:
    diag = OUT / "diagnostics"
    rows = [
        cluster_size_row("firm", data[FIRM], len(data)),
        cluster_size_row("auditor", data["_auditor"], len(data)),
        cluster_size_row("auditor_industry", data["_auditor_industry"], len(data)),
        cluster_size_row("auditor_industry_year", data["_auditor_industry_year"], len(data)),
        cluster_size_row("industry", data[INDUSTRY], len(data)),
        cluster_size_row("year", data[YEAR], len(data)),
    ]
    pd.DataFrame(rows).to_csv(diag / "cluster_counts.csv", index=False, encoding="utf-8-sig")


def save_variable_diagnostics(data: pd.DataFrame) -> None:
    numeric = [*CORE, *CONTROLS]
    summary = data[numeric].describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]).T
    summary["missing"] = data[numeric].isna().sum()
    summary["missing_rate"] = data[numeric].isna().mean()
    summary.to_csv(OUT / "diagnostics/variable_summary.csv", encoding="utf-8-sig")
    pd.DataFrame([{
        "raw_rows": len(data),
        "company_year_duplicates": int(data.duplicated([FIRM, YEAR]).sum()),
        "firms": data[FIRM].nunique(),
        "auditors": data["_auditor"].nunique(),
        "auditor_industry": data["_auditor_industry"].nunique(),
        "auditor_industry_year": data["_auditor_industry_year"].nunique(),
        "industries": data[INDUSTRY].nunique(),
        "years": data[YEAR].nunique(),
    }]).to_csv(OUT / "diagnostics/data_quality.csv", index=False, encoding="utf-8-sig")


def clustered_vcov(xtx_inv: np.ndarray, scores: np.ndarray, codes: np.ndarray, g: int, n: int, k: int) -> np.ndarray:
    gsum = np.zeros((g, k), dtype=float)
    np.add.at(gsum, codes, scores)
    meat = gsum.T @ gsum
    scale = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    return scale * xtx_inv @ meat @ xtx_inv


def wild_cluster(sample: pd.DataFrame, x: str, cluster_col: str, n_boot: int, seed: int) -> dict:
    formula = formula_for(x)
    y_df, x_df = patsy.dmatrices(formula, sample, return_type="dataframe")
    y = np.asarray(y_df).reshape(-1)
    x_mat = np.asarray(x_df, dtype=float)
    names = list(x_df.columns)
    idx = names.index(parameter_key(x))
    n, k = x_mat.shape
    xtx_inv = np.linalg.inv(x_mat.T @ x_mat)
    codes, uniques = pd.factorize(sample[cluster_col], sort=False)
    g = int(len(uniques))

    beta = xtx_inv @ (x_mat.T @ y)
    resid = y - x_mat @ beta
    vcov = clustered_vcov(xtx_inv, x_mat * resid[:, None], codes, g, n, k)
    se_obs = float(np.sqrt(max(vcov[idx, idx], 0.0)))
    t_obs = float(beta[idx] / se_obs) if se_obs > 0 else np.nan

    keep = [i for i in range(k) if i != idx]
    x_r = x_mat[:, keep]
    xtx_r_inv = np.linalg.inv(x_r.T @ x_r)
    beta_r = xtx_r_inv @ (x_r.T @ y)
    fitted_r = x_r @ beta_r
    resid_r = y - fitted_r
    fitted_u = x_mat @ beta
    resid_u = resid

    rng = np.random.default_rng(seed)
    t_stars = np.empty(n_boot)
    beta_stars = np.empty(n_boot)
    for b in range(n_boot):
        weights = rng.choice(WEBB, size=g)
        w = weights[codes]
        y_r = fitted_r + w * resid_r
        beta_b = xtx_inv @ (x_mat.T @ y_r)
        resid_b = y_r - x_mat @ beta_b
        vcov_b = clustered_vcov(xtx_inv, x_mat * resid_b[:, None], codes, g, n, k)
        se_b = float(np.sqrt(max(vcov_b[idx, idx], 0.0)))
        t_stars[b] = beta_b[idx] / se_b if se_b > 0 else np.nan
        y_u = fitted_u + w * resid_u
        beta_u = xtx_inv @ (x_mat.T @ y_u)
        beta_stars[b] = beta_u[idx]

    abs_t = np.abs(t_stars)
    wcr_p = float((1.0 + np.nansum(abs_t >= abs(t_obs))) / (n_boot + 1.0))
    return {
        "variable": x,
        "cluster": cluster_col if cluster_col not in {INDUSTRY, "_auditor"} else (
            "industry_clustered" if cluster_col == INDUSTRY else "auditor_clustered"
        ),
        "n_clusters": g,
        "n_boot": n_boot,
        "seed": seed,
        "coefficient": float(beta[idx]),
        "sandwich_se": se_obs,
        "sandwich_t": t_obs,
        "sandwich_p": float(2 * norm.sf(abs(t_obs))) if np.isfinite(t_obs) else np.nan,
        "wcr_p": wcr_p,
        "ci_low": float(np.nanpercentile(beta_stars, 2.5)),
        "ci_high": float(np.nanpercentile(beta_stars, 97.5)),
        "method": "restricted_webb_wcr",
        "ci_method": "unrestricted_webb_percentile",
        "n": n,
    }


def main() -> None:
    for folder in [OUT / "coefficients", OUT / "diagnostics", OUT / "tables", OUT / "logs"]:
        folder.mkdir(parents=True, exist_ok=True)
    data = load()
    save_variable_diagnostics(data)
    save_cluster_diagnostics(data)

    all_rows, samples = [], []
    for x in CORE:
        sample = model_sample(data, x)
        result = smf.ols(formula_for(x), data=sample).fit()
        for label, column in ONEWAY:
            rows = fit_oneway(result, sample, x, label, column)
            all_rows.append(rows)
            samples.append({"model": f"{x}_{label}", "n": int(result.nobs), "cluster": label, "core": x})
        for label, column1, column2 in TWOWAY:
            rows = fit_twoway(result, sample, x, label, column1, column2)
            all_rows.append(rows)
            samples.append({"model": f"{x}_{label}", "n": int(result.nobs), "cluster": label, "core": x})

    coefficients = pd.concat(all_rows, ignore_index=True)
    coefficients.to_csv(OUT / "coefficients/robustness_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(OUT / "diagnostics/robustness_sample_info.csv", index=False, encoding="utf-8-sig")

    wcr_rows = []
    for x in CORE:
        sample = model_sample(data, x)
        for label, column in WCR_CLUSTERS:
            wcr_rows.append(wild_cluster(sample, x, column, N_BOOT, BOOT_SEED))
    pd.DataFrame(wcr_rows).to_csv(OUT / "coefficients/wild_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")

    metadata = pd.DataFrame({
        "item": [
            "input_panel", "raw_rows", "firms", "auditors", "auditor_industry",
            "auditor_industry_year", "industries", "years", "fixed_effects",
            "oneway", "twoway", "wcr", "n_boot", "boot_seed",
        ],
        "value": [
            str(INPUT), len(data), data[FIRM].nunique(), data["_auditor"].nunique(),
            data["_auditor_industry"].nunique(), data["_auditor_industry_year"].nunique(),
            data[INDUSTRY].nunique(), data[YEAR].nunique(), "industry + year",
            ";".join(label for label, _ in ONEWAY),
            ";".join(label for label, _, _ in TWOWAY),
            ";".join(label for label, _ in WCR_CLUSTERS),
            N_BOOT, BOOT_SEED,
        ],
    })
    metadata.to_csv(OUT / "diagnostics/input_metadata.csv", index=False, encoding="utf-8-sig")
    (OUT / "logs/run_summary.txt").write_text(
        "\n".join([
            f"input={INPUT}",
            f"rows={len(data)}",
            f"models={len(samples)}",
            "fixed_effects=industry_year",
            f"oneway={[label for label, _ in ONEWAY]}",
            f"twoway={[label for label, _, _ in TWOWAY]}",
            f"wcr={[label for label, _ in WCR_CLUSTERS]}",
            f"n_boot={N_BOOT}",
            f"boot_seed={BOOT_SEED}",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"input={INPUT}\nrows={len(data)}\noutputs={OUT}")


if __name__ == "__main__":
    main()
