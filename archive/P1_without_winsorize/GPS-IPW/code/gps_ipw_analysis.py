from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import logit
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[4]
INPUT = PROJECT_ROOT / "Regression/outputs/paper_tables/common_sample_14778.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "results"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, IMS = "Greenwashing连续_基准", "IMS"
CONTROLS = [
    "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE", "Dual",
    "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee",
    "AuditorTenure", "IsChangeFirm",
]
CORE_CONTROLS = [
    "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE",
]
GOVERNANCE_CONTROLS = ["Dual", "BoardSize", "IndependentDirectorRatio"]
AUDIT_SELECTION_CONTROLS = ["Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm"]
FIRST_STAGE_SPECS = {
    "核心共同原因": [GW, *CORE_CONTROLS],
    "完整共同原因": [GW, *CORE_CONTROLS, *GOVERNANCE_CONTROLS],
    "完整共同原因加审计选择": [GW, *CORE_CONTROLS, *GOVERNANCE_CONTROLS, *AUDIT_SELECTION_CONTROLS],
}


def significance(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig", low_memory=False)
    required = [FIRM, YEAR, INDUSTRY, GW, IMS, *CONTROLS]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    for column in [YEAR, GW, IMS, *CONTROLS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values([FIRM, YEAR]).copy()
    prior_year = data.groupby(FIRM)[YEAR].shift(1)
    consecutive = data[YEAR].sub(prior_year).eq(1)
    for column in [GW, *CONTROLS]:
        data[f"L_{column}"] = data.groupby(FIRM)[column].shift(1).where(consecutive)
    return data


def fit_wls(formula: str, sample: pd.DataFrame, weights: str) -> object:
    return smf.wls(formula, sample, weights=sample[weights]).fit(
        cov_type="cluster", cov_kwds={"groups": sample[FIRM]}
    )


def effective_sample_size(weights: pd.Series) -> float:
    return float(weights.sum() ** 2 / np.square(weights).sum())


def formula_terms(columns: list[str]) -> str:
    return " + ".join([f"Q('{column}')" for column in columns])


def weighted_mean(values: pd.Series, weights: pd.Series | None = None) -> float:
    if weights is None:
        return float(values.mean())
    return float(np.average(values, weights=weights))


def standardized_mean_difference(sample: pd.DataFrame, variable: str, group: pd.Series, weights: pd.Series | None = None) -> float:
    treated = sample.loc[group, variable]
    comparison = sample.loc[~group, variable]
    treated_weights = None if weights is None else weights.loc[group]
    comparison_weights = None if weights is None else weights.loc[~group]
    mean_treated = weighted_mean(treated, treated_weights)
    mean_comparison = weighted_mean(comparison, comparison_weights)
    if weights is None:
        variance_treated = treated.var(ddof=1)
        variance_comparison = comparison.var(ddof=1)
    else:
        variance_treated = np.average((treated - mean_treated) ** 2, weights=treated_weights)
        variance_comparison = np.average((comparison - mean_comparison) ** 2, weights=comparison_weights)
    pooled_sd = np.sqrt((variance_treated + variance_comparison) / 2)
    return float((mean_treated - mean_comparison) / pooled_sd) if pooled_sd > 0 else np.nan


def write_ims_distribution(sample: pd.DataFrame) -> None:
    quantiles = [0, .01, .05, .10, .25, .50, .75, .90, .95, .99, 1]
    distribution = pd.DataFrame({"分位点": quantiles, "IMS": sample[IMS].quantile(quantiles).to_numpy()})
    distribution["观测数"] = len(sample)
    distribution.to_csv(OUTPUT / "ims_distribution_quantiles.csv", index=False, encoding="utf-8-sig")
    sample["IMS分组"] = pd.qcut(sample[IMS], q=5, duplicates="drop")
    groups = sample.groupby("IMS分组", observed=True)[IMS].agg(["count", "min", "max", "mean", "median"]).reset_index()
    groups["占比"] = groups["count"] / len(sample)
    groups.to_csv(OUTPUT / "ims_distribution_quintiles.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    data = load_data()
    spec_columns = {
        name: [f"L_{column}" for column in columns]
        for name, columns in FIRST_STAGE_SPECS.items()
    }
    all_lagged = sorted({column for columns in spec_columns.values() for column in columns})
    sample = data.dropna(subset=[GW, IMS, FIRM, YEAR, INDUSTRY, *all_lagged]).copy()
    sample["IMS_logit"] = logit(np.clip(sample[IMS], 1e-5, 1 - 1e-5))

    write_ims_distribution(sample)
    sample["IMS分组"] = pd.qcut(sample[IMS], q=5, duplicates="drop")
    rows, diagnostics, treatment_diagnostics, balance_rows = [], [], [], []
    weight_specs = [
        ("未截尾", "gps_ipw_raw"),
        ("P0.5--P99.5截尾", "gps_ipw_trim_05_995"),
        ("P1--P99截尾", "gps_ipw_trim_1_99"),
    ]
    outcome_specs = [("G1_纯GPS-IPW", [IMS]), ("G2_增强GPS-IPW", [IMS, *all_lagged])]

    for spec_name, covariates in spec_columns.items():
        treatment_formula = "IMS_logit ~ " + " + ".join(
            [formula_terms(covariates), f"C(Q('{YEAR}'))", f"C(Q('{INDUSTRY}'))"]
        )
        treatment_fit = smf.ols(treatment_formula, sample).fit()
        conditional_sd = float(np.sqrt(treatment_fit.mse_resid))
        marginal_mean = sample["IMS_logit"].mean()
        marginal_sd = sample["IMS_logit"].std(ddof=1)
        conditional_density = norm.pdf(sample["IMS_logit"], treatment_fit.fittedvalues, conditional_sd)
        marginal_density = norm.pdf(sample["IMS_logit"], marginal_mean, marginal_sd)
        sample["gps_ipw_raw"] = marginal_density / np.clip(conditional_density, 1e-12, None)
        p005, p995 = sample["gps_ipw_raw"].quantile([0.005, 0.995])
        p01, p99 = sample["gps_ipw_raw"].quantile([0.01, 0.99])
        sample["gps_ipw_trim_05_995"] = sample["gps_ipw_raw"].clip(p005, p995)
        sample["gps_ipw_trim_1_99"] = sample["gps_ipw_raw"].clip(p01, p99)
        groups = sample["IMS分组"]
        for group_name in groups.cat.categories:
            treated = groups.eq(group_name)
            for variable in covariates:
                for adjustment_name, adjustment_weights in [("未调整", None), ("GPS原始权重", sample["gps_ipw_raw"])]:
                    smd = standardized_mean_difference(sample, variable, treated, adjustment_weights)
                    balance_rows.append({
                        "第一阶段规格": spec_name, "IMS剂量组": str(group_name), "协变量": variable,
                        "调整方式": adjustment_name,
                        "剂量组均值": weighted_mean(sample.loc[treated, variable], None if adjustment_weights is None else adjustment_weights.loc[treated]),
                        "其余样本均值": weighted_mean(sample.loc[~treated, variable], None if adjustment_weights is None else adjustment_weights.loc[~treated]),
                        "SMD": smd, "绝对SMD小于0.1": abs(smd) < .1,
                    })
        treatment_diagnostics.append({
            "第一阶段规格": spec_name,
            "第一阶段协变量": ";".join(covariates),
            "样本数": len(sample),
            "企业数": sample[FIRM].nunique(),
            "处理模型残差标准差": conditional_sd,
            "处理模型调整R方": treatment_fit.rsquared_adj,
            "IMS_P05": sample[IMS].quantile(.05),
            "IMS_P95": sample[IMS].quantile(.95),
            "GPS条件密度P01": np.quantile(conditional_density, .01),
            "GPS条件密度P50": np.quantile(conditional_density, .50),
            "GPS条件密度P99": np.quantile(conditional_density, .99),
        })
        for trim_name, weight_name in weight_specs:
            weight = sample[weight_name]
            diagnostics.append({
                "第一阶段规格": spec_name, "权重规格": trim_name, "权重变量": weight_name,
                "样本数": len(sample), "均值": weight.mean(), "标准差": weight.std(),
                "最小值": weight.min(), "P01": weight.quantile(.01), "P05": weight.quantile(.05),
                "中位数": weight.median(), "P95": weight.quantile(.95), "P99": weight.quantile(.99),
                "最大值": weight.max(), "ESS": effective_sample_size(weight),
            })
            for model_name, regressors in outcome_specs:
                formula = f"Q('{GW}') ~ " + " + ".join(
                    [formula_terms(regressors), f"C(Q('{YEAR}'))", f"C(Q('{INDUSTRY}'))"]
                )
                result = fit_wls(formula, sample, weight_name)
                term = f"Q('{IMS}')"
                p_value = float(result.pvalues[term])
                rows.append({
                    "第一阶段规格": spec_name, "权重规格": trim_name, "模型": model_name,
                    "变量": IMS, "系数": result.params[term], "标准误": result.bse[term],
                    "t值": result.tvalues[term], "p值": p_value, "显著性": significance(p_value),
                    "样本数": int(result.nobs), "企业数": sample[FIRM].nunique(),
                    "调整R方": result.rsquared_adj, "有效样本量ESS": effective_sample_size(weight),
                })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT / "gps_ipw_regressions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "gps_ipw_weight_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(treatment_diagnostics).to_csv(OUTPUT / "gps_ipw_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(balance_rows).to_csv(OUTPUT / "gps_covariate_balance.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
