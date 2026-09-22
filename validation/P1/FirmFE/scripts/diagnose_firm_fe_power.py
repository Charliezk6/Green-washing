"""Power, Mundlak decomposition, and pre-specified firm-FE specification checks.

Reports minimum detectable effects, whether the firm-FE interval contains the
industry-FE estimate, a cluster-robust Wald test of equality, the Mundlak
within/between split, and specification variants that were fixed before seeing
results: sparse controls, firm FE plus industry-by-year FE, firm FE with
leave-one-out IMS, and PanelOLS degrees-of-freedom correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPTS))

from run_firm_fe_regression import (  # noqa: E402
    ALL_CONTROLS,
    CORE,
    DEPENDENT,
    fit,
    load_data,
    significance,
    winsorize_controls,
)

DIAG = ROOT / "outputs" / "diagnostics"
TABLES = ROOT / "outputs" / "tables"
FIRM = "证券代码"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"
MARKET = PROJECT / "X_reproduction" / "data" / "input" / "full_a_share_audit_market_2021_2024.csv"

# Controls whose within-firm SD share is under about 8 percent.
SPARSE_DROP = ["Size", "Lev", "FirmAge", "SOE", "Big4", "LnAuditFee"]
SPARSE_CONTROLS = [c for c in ALL_CONTROLS if c not in SPARSE_DROP]


def prepare_common() -> pd.DataFrame:
    data = load_data()
    required = [DEPENDENT, *CORE, *ALL_CONTROLS, FIRM, YEAR, INDUSTRY, "InstitutionID", "OperatingRevenue"]
    common = data.dropna(subset=[DEPENDENT, *CORE, *ALL_CONTROLS, FIRM, YEAR, INDUSTRY]).copy()
    common, _ = winsorize_controls(common)
    return common.sort_values([FIRM, YEAR]).reset_index(drop=True)


def mde(se: float, alpha: float, power: float) -> float:
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)


def achieved_power(se: float, effect: float, alpha: float = 0.05) -> float:
    if se <= 0 or not np.isfinite(se):
        return np.nan
    z = norm.ppf(1 - alpha / 2)
    delta = abs(effect) / se
    return float(norm.cdf(delta - z) + norm.cdf(-delta - z))


def fit_industry(data: pd.DataFrame, core: str, controls: list[str]):
    sample = data.dropna(subset=[DEPENDENT, core, FIRM, YEAR, INDUSTRY, *controls]).copy()
    terms = [core, *controls, f"C({YEAR})", f"C({INDUSTRY})"]
    model = smf.ols(f"Q('{DEPENDENT}') ~ " + " + ".join(terms), data=sample)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
    return result, sample


def cluster_cross_covariance(result_a, result_b, name_a: str, name_b: str, groups: pd.Series) -> float:
    """Cluster-robust covariance between one coefficient from each regression.

    Marginal variances are rescaled to the statsmodels clustered standard errors
    so the Wald test uses the same finite-sample correction as the reported SEs.
    """
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    idx_a = list(result_a.model.exog_names).index(name_a)
    idx_b = list(result_b.model.exog_names).index(name_b)
    score_a = pd.DataFrame(result_a.model.exog * np.asarray(result_a.resid)[:, None])
    score_b = pd.DataFrame(result_b.model.exog * np.asarray(result_b.resid)[:, None])
    sa = score_a.groupby(groups).sum()
    sb = score_b.groupby(groups).sum()
    meat = sa.T.to_numpy() @ sb.to_numpy()
    g_count = sa.shape[0]
    meat *= g_count / (g_count - 1)
    bread_a = np.linalg.pinv(result_a.model.exog.T @ result_a.model.exog)
    bread_b = np.linalg.pinv(result_b.model.exog.T @ result_b.model.exog)
    cov = bread_a @ meat @ bread_b.T
    var_a = (bread_a @ (sa.T.to_numpy() @ sa.to_numpy()) * g_count / (g_count - 1) @ bread_a.T)[idx_a, idx_a]
    var_b = (bread_b @ (sb.T.to_numpy() @ sb.to_numpy()) * g_count / (g_count - 1) @ bread_b.T)[idx_b, idx_b]
    scale_a = result_a.bse[name_a] / np.sqrt(var_a)
    scale_b = result_b.bse[name_b] / np.sqrt(var_b)
    return float(cov[idx_a, idx_b] * scale_a * scale_b)


def power_row(spec: str, core: str, estimate: float, se: float, baseline: float, baseline_se: float,
              cov: float, n_obs: int, n_firms: int, p_value: float) -> dict:
    diff = estimate - baseline
    # Without a cross-model covariance, use the independent-estimator bound.
    used_cov = 0.0 if cov is None or not np.isfinite(cov) else cov
    cov_note = "独立估计" if cov is None or not np.isfinite(cov) else "聚类得分交叉协方差"
    var_diff = se ** 2 + baseline_se ** 2 - 2 * used_cov
    se_diff = float(np.sqrt(var_diff)) if var_diff > 0 else np.nan
    z_eq = diff / se_diff if se_diff and np.isfinite(se_diff) else np.nan
    p_eq = float(2 * (1 - norm.cdf(abs(z_eq)))) if np.isfinite(z_eq) else np.nan
    ci_low = estimate - 1.95996398454 * se
    ci_high = estimate + 1.95996398454 * se
    return {
        "规格": spec,
        "核心变量": core,
        "FirmFE系数": estimate,
        "FirmFE标准误": se,
        "FirmFE_p值": p_value,
        "FirmFE显著性": significance(p_value),
        "FirmFE_95CI下限": ci_low,
        "FirmFE_95CI上限": ci_high,
        "置信区间包含基准点估计": bool(ci_low <= baseline <= ci_high),
        "基准系数": baseline,
        "基准标准误": baseline_se,
        "系数差_FirmFE减基准": diff,
        "系数差标准误_含协方差": se_diff,
        "系数差协方差假设": cov_note,
        "Wald_H0_FirmFE等于基准_z": z_eq,
        "Wald_H0_FirmFE等于基准_p": p_eq,
        "MDE_5pct_80功效": mde(se, 0.05, 0.80),
        "MDE_5pct_90功效": mde(se, 0.05, 0.90),
        "MDE_10pct_80功效": mde(se, 0.10, 0.80),
        "MDE_10pct_90功效": mde(se, 0.10, 0.90),
        "对基准效应的实际功效_5pct": achieved_power(se, baseline, 0.05),
        "对基准效应的实际功效_10pct": achieved_power(se, baseline, 0.10),
        "样本数": n_obs,
        "公司数": n_firms,
    }


def mundlak(data: pd.DataFrame, core: str, controls: list[str]):
    sample = data.dropna(subset=[DEPENDENT, core, FIRM, YEAR, INDUSTRY, *controls]).copy()
    sample[f"{core}_bar"] = sample.groupby(FIRM)[core].transform("mean")
    sample[f"{core}_within"] = sample[core] - sample[f"{core}_bar"]
    terms = [f"{core}_within", f"{core}_bar", *controls, f"C({YEAR})", f"C({INDUSTRY})"]
    model = smf.ols(f"Q('{DEPENDENT}') ~ " + " + ".join(terms), data=sample)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
    return result, sample


def mundlak_rows(result, core: str) -> list[dict]:
    within = f"{core}_within"
    between = f"{core}_bar"
    contrast = result.t_test(f"{within} - {between} = 0")
    rows = []
    for label, name in (("within", within), ("between", between)):
        ci = result.conf_int().loc[name]
        rows.append({
            "核心变量": core,
            "成分": label,
            "变量": name,
            "系数": float(result.params[name]),
            "标准误": float(result.bse[name]),
            "t值": float(result.tvalues[name]),
            "p值": float(result.pvalues[name]),
            "显著性": significance(float(result.pvalues[name])),
            "置信区间下限": float(ci[0]),
            "置信区间上限": float(ci[1]),
            "样本数": int(result.nobs),
            "调整R方": float(result.rsquared_adj),
            "Hausman_within减between": float(result.params[within] - result.params[between]),
            "Hausman_t": float(np.asarray(contrast.tvalue).ravel()[0]),
            "Hausman_p": float(np.asarray(contrast.pvalue).ravel()[0]),
            "FE": "行业+年份，Mundlak分解",
            "标准误类型": "公司聚类",
        })
    return rows


def panelols_fit(data: pd.DataFrame, core: str, controls: list[str]) -> dict:
    sample = data.dropna(subset=[DEPENDENT, core, FIRM, YEAR, *controls]).copy()
    sample = sample.set_index([FIRM, YEAR])
    counts = sample.groupby(level=0).size()
    sample = sample.loc[sample.index.get_level_values(0).isin(counts[counts >= 2].index)]
    model = PanelOLS(
        sample[DEPENDENT],
        sample[[core, *controls]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)
    return {
        "系数": float(result.params[core]),
        "标准误": float(result.std_errors[core]),
        "t值": float(result.tstats[core]),
        "p值": float(result.pvalues[core]),
        "样本数": int(result.nobs),
        "公司数": int(sample.index.get_level_values(0).nunique()),
    }


def fit_firm_industry_year(data: pd.DataFrame, core: str, controls: list[str]) -> dict:
    sample = data.dropna(subset=[DEPENDENT, core, FIRM, YEAR, INDUSTRY, *controls]).copy()
    sample["ind_year"] = sample[INDUSTRY].astype(str) + "_" + sample[YEAR].astype(int).astype(str)
    sample = sample.set_index([FIRM, YEAR])
    counts = sample.groupby(level=0).size()
    sample = sample.loc[sample.index.get_level_values(0).isin(counts[counts >= 2].index)]
    other = sample[["ind_year"]]
    model = PanelOLS(
        sample[DEPENDENT],
        sample[[core, *controls]],
        entity_effects=True,
        other_effects=other,
        drop_absorbed=True,
    )
    result = model.fit(cov_type="clustered", cluster_entity=True)
    return {
        "系数": float(result.params[core]),
        "标准误": float(result.std_errors[core]),
        "t值": float(result.tstats[core]),
        "p值": float(result.pvalues[core]),
        "样本数": int(result.nobs),
        "公司数": int(sample.index.get_level_values(0).nunique()),
    }


def expertise_code(industry_code: object) -> str:
    code = str(industry_code).strip()
    return code if code.startswith("C") else code[:1]


def add_ims_loo(common: pd.DataFrame) -> pd.DataFrame:
    market = pd.read_csv(MARKET, dtype={"InstitutionID": str, "证券代码": str}, encoding="utf-8-sig")
    market[YEAR] = pd.to_numeric(market[YEAR], errors="coerce")
    market["OperatingRevenue"] = pd.to_numeric(market["OperatingRevenue"], errors="coerce")
    market["InstitutionID"] = market["InstitutionID"].map(
        lambda v: str(int(float(v))) if pd.notna(v) and str(v).strip() else None
    )
    market[INDUSTRY] = market["IndustryCode"].map(expertise_code)
    market = market[market[YEAR].isin([2021, 2022, 2023, 2024]) & market["OperatingRevenue"].gt(0)]
    market = market[market[INDUSTRY].ne("O")].dropna(subset=["InstitutionID", INDUSTRY])
    auditor_total = market.groupby(["InstitutionID", INDUSTRY, YEAR])["OperatingRevenue"].sum()
    industry_total = market.groupby([INDUSTRY, YEAR])["OperatingRevenue"].sum()
    result = common.copy()
    result["InstitutionID"] = result["InstitutionID"].map(
        lambda v: str(int(float(v))) if pd.notna(v) and str(v).strip() else None
    )
    auditor_keys = pd.MultiIndex.from_frame(result[["InstitutionID", INDUSTRY, YEAR]])
    industry_keys = pd.MultiIndex.from_frame(result[[INDUSTRY, YEAR]])
    numerator = auditor_keys.map(auditor_total) - result["OperatingRevenue"].where(result["OperatingRevenue"].gt(0))
    denominator = industry_keys.map(industry_total) - result["OperatingRevenue"].where(result["OperatingRevenue"].gt(0))
    valid = numerator.ge(0) & denominator.gt(0)
    result["IMS_LOO"] = (numerator / denominator).where(valid)
    result["IMS_LOO_D"] = (result["IMS_LOO"] >= 0.20).astype(float).where(result["IMS_LOO"].notna())
    return result


def spec_record(name: str, core: str, estimate: dict, note: str) -> dict:
    return {
        "规格": name,
        "核心变量": core,
        "系数": estimate["系数"],
        "标准误": estimate["标准误"],
        "t值": estimate["t值"],
        "p值": estimate["p值"],
        "显著性": significance(estimate["p值"]),
        "样本数": estimate["样本数"],
        "公司数": estimate["公司数"],
        "说明": note,
    }


def from_statsmodels(result, sample: pd.DataFrame, core: str) -> dict:
    return {
        "系数": float(result.params[core]),
        "标准误": float(result.bse[core]),
        "t值": float(result.tvalues[core]),
        "p值": float(result.pvalues[core]),
        "样本数": int(result.nobs),
        "公司数": int(sample[FIRM].nunique()),
    }


def main() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    common = prepare_common()
    print(f"common_sample={len(common)} firms={common[FIRM].nunique()}")

    power_rows = []
    spec_rows = []
    mundlak_out = []
    for core in CORE:
        baseline_result, baseline_sample = fit_industry(common, core, ALL_CONTROLS)
        firm_result, firm_sample = fit(common, core, ALL_CONTROLS)
        baseline = from_statsmodels(baseline_result, baseline_sample, core)
        firm = from_statsmodels(firm_result, firm_sample, core)
        cov = cluster_cross_covariance(
            firm_result, baseline_result, core, core, firm_sample[FIRM]
        )
        power_rows.append(power_row(
            "企业+年份FE_全部控制_demean", core, firm["系数"], firm["标准误"],
            baseline["系数"], baseline["标准误"], cov, firm["样本数"], firm["公司数"], firm["p值"],
        ))
        spec_rows.append(spec_record(
            "企业+年份FE_全部控制_demean", core, firm,
            "与正文相同控制，行业FE替换为企业FE；demean后标准误未计入企业哑变量自由度",
        ))
        spec_rows.append(spec_record(
            "行业+年份FE_全部控制", core, baseline, "正文基准规格，用作对照",
        ))

        panel = panelols_fit(common, core, ALL_CONTROLS)
        power_rows.append(power_row(
            "企业+年份FE_全部控制_PanelOLS", core, panel["系数"], panel["标准误"],
            baseline["系数"], baseline["标准误"], cov, panel["样本数"], panel["公司数"], panel["p值"],
        ))
        spec_rows.append(spec_record(
            "企业+年份FE_全部控制_PanelOLS", core, panel,
            "linearmodels吸收企业与年份效应，标准误按吸收后的自由度计算",
        ))

        sparse_result, sparse_sample = fit(common, core, SPARSE_CONTROLS)
        sparse = from_statsmodels(sparse_result, sparse_sample, core)
        power_rows.append(power_row(
            "企业+年份FE_剔除近似时不变控制", core, sparse["系数"], sparse["标准误"],
            baseline["系数"], baseline["标准误"], np.nan, sparse["样本数"], sparse["公司数"], sparse["p值"],
        ))
        spec_rows.append(spec_record(
            "企业+年份FE_剔除近似时不变控制", core, sparse,
            "剔除组内变异极低的Size、Lev、FirmAge、SOE、Big4、LnAuditFee",
        ))

        ind_year = fit_firm_industry_year(common, core, ALL_CONTROLS)
        power_rows.append(power_row(
            "企业FE+行业乘年份FE", core, ind_year["系数"], ind_year["标准误"],
            baseline["系数"], baseline["标准误"], np.nan,
            ind_year["样本数"], ind_year["公司数"], ind_year["p值"],
        ))
        spec_rows.append(spec_record(
            "企业FE+行业乘年份FE", core, ind_year,
            "企业固定效应加行业×年份固定效应；行业×年份已吸收年度效应",
        ))

        decomposed = mundlak(common, core, ALL_CONTROLS)[0]
        mundlak_out.extend(mundlak_rows(decomposed, core))

    loo = add_ims_loo(common)
    for core in ["IMS_LOO", "IMS_LOO_D"]:
        loo_result, loo_sample = fit(loo, core, ALL_CONTROLS)
        estimate = from_statsmodels(loo_result, loo_sample, core)
        baseline_core = "IMS" if core == "IMS_LOO" else "IMS_D"
        baseline_match = next(
            row for row in spec_rows
            if row["规格"] == "行业+年份FE_全部控制" and row["核心变量"] == baseline_core
        )
        power_rows.append(power_row(
            "企业+年份FE_LOO处理变量", core, estimate["系数"], estimate["标准误"],
            baseline_match["系数"], baseline_match["标准误"], np.nan,
            estimate["样本数"], estimate["公司数"], estimate["p值"],
        ))
        spec_rows.append(spec_record(
            "企业+年份FE_LOO处理变量", core, estimate,
            "组内回归使用leave-one-out份额，避免本公司营收同时进入分子和分母",
        ))

    power = pd.DataFrame(power_rows)
    specs = pd.DataFrame(spec_rows)
    mundlak_table = pd.DataFrame(mundlak_out)
    power.to_csv(TABLES / "firm_fe_power_and_mde.csv", index=False, encoding="utf-8-sig")
    specs.to_csv(DIAG / "firm_fe_spec_extensions.csv", index=False, encoding="utf-8-sig")
    mundlak_table.to_csv(TABLES / "mundlak_decomposition.csv", index=False, encoding="utf-8-sig")
    print(power[["规格", "核心变量", "FirmFE系数", "FirmFE标准误", "Wald_H0_FirmFE等于基准_p", "MDE_5pct_80功效"]].to_string(index=False))
    print(mundlak_table[["核心变量", "成分", "系数", "标准误", "p值", "Hausman_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
