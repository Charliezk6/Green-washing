from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit, logit
from scipy.stats import ttest_ind

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INPUT = PACKAGE_ROOT / "data/input/combined_regression_panel_14778.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, TREAT = "Greenwashing连续_基准", "IMS_D"
CORE = ["Greenwashing连续_基准", "Size", "Lev", "ROA", "CFO", "Growth", "FirmAgeLog",
        "SOE", "Dual", "BoardSize", "IndependentDirectorRatio"]
AUDIT = ["Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm"]
LAGS = [f"L_{variable}" for variable in [*CORE, *AUDIT]]
CORE_LAGS = [f"L_{variable}" for variable in CORE]
MAIN_FACTOR = 0.20
SPECIFICATIONS = [
    ("主规格_Logit_年度行业内_有放回_0.20SD", CORE_LAGS, MAIN_FACTOR),
    ("窄caliper_Logit_年度行业内_有放回_0.10SD", CORE_LAGS, 0.10),
    ("宽caliper_Logit_年度行业内_有放回_0.25SD", CORE_LAGS, 0.25),
    ("完整审计控制_Logit_年度行业内_有放回_0.20SD", LAGS, MAIN_FACTOR),
]


def significance(p_value: float) -> str:
    return "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig")
    required = [FIRM, YEAR, INDUSTRY, GW, TREAT, *CORE, *AUDIT]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    for column in [YEAR, GW, TREAT, *CORE, *AUDIT]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values([FIRM, YEAR]).copy()
    previous_year = data.groupby(FIRM)[YEAR].shift(1)
    consecutive = data[YEAR].sub(previous_year).eq(1)
    for column in [*CORE, *AUDIT]:
        data[f"L_{column}"] = data.groupby(FIRM)[column].shift(1).where(consecutive)
    data["row_id"] = np.arange(len(data))
    return data


def tidy_propensity_model(result, model_name: str) -> pd.DataFrame:
    intervals = result.conf_int()
    rows = []
    for variable in result.params.index:
        if variable.startswith("C("):
            continue
        rows.append({"规格": model_name, "变量": variable, "系数": result.params[variable],
                     "标准误": result.bse[variable], "z值": result.tvalues[variable],
                     "p值": result.pvalues[variable], "显著性": significance(result.pvalues[variable]),
                     "置信区间下限": intervals.loc[variable, 0], "置信区间上限": intervals.loc[variable, 1],
                     "样本数": int(result.nobs), "收敛": bool(result.mle_retvals.get("converged", False))})
    return pd.DataFrame(rows)


def fit_propensity(data: pd.DataFrame, covariates: list[str]):
    formula = f"{TREAT} ~ " + " + ".join(covariates + [f"C({YEAR})", f"C({INDUSTRY})"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.logit(formula, data).fit(disp=False, maxiter=500)
    if not result.mle_retvals.get("converged", False):
        raise RuntimeError("Logit 倾向得分模型未收敛")
    if any("converg" in str(item.message).lower() for item in caught):
        raise RuntimeError("Logit 倾向得分模型出现收敛警告")
    score = np.clip(result.predict(data), 1e-6, 1 - 1e-6)
    data = data.copy()
    data["propensity_score"] = score
    data["logit_score"] = logit(score)
    return data, result


def match_with_replacement(data: pd.DataFrame, caliper: float) -> tuple[pd.DataFrame, dict]:
    treated = data.loc[data[TREAT].eq(1)].copy()
    controls = data.loc[data[TREAT].eq(0)].copy()
    score_sd = float(data["logit_score"].std(ddof=1))
    caliper_value = caliper * score_sd
    pairs = []
    for _, treated_row in treated.iterrows():
        same_stratum = controls.loc[
            controls[YEAR].eq(treated_row[YEAR]) & controls[INDUSTRY].eq(treated_row[INDUSTRY])
        ]
        if same_stratum.empty:
            continue
        distances = (same_stratum["logit_score"] - treated_row["logit_score"]).abs()
        nearest = distances.idxmin()
        distance = float(distances.loc[nearest])
        if distance <= caliper_value:
            pairs.append({"pair_id": len(pairs) + 1,
                          "treated_row_id": int(treated_row["row_id"]),
                          "control_row_id": int(same_stratum.loc[nearest, "row_id"]),
                          "distance": distance})
    pair_data = pd.DataFrame(pairs)
    if pair_data.empty:
        return pair_data, {"caliper_sd": score_sd, "caliper_value": caliper_value,
                           "matched_treated": 0, "unmatched_treated": len(treated),
                           "unique_controls": 0, "total_pairs": 0}
    control_use = pair_data["control_row_id"].value_counts().to_dict()
    indexed = data.set_index("row_id")
    rows = []
    for pair in pair_data.itertuples(index=False):
        treated_row = indexed.loc[pair.treated_row_id].to_dict()
        treated_row.update({"row_id": pair.treated_row_id, "pair_id": pair.pair_id,
                            "match_role": "treated", "match_distance": pair.distance,
                            "match_weight": 1.0})
        control_row = indexed.loc[pair.control_row_id].to_dict()
        control_row.update({"row_id": pair.control_row_id, "pair_id": pair.pair_id,
                            "match_role": "control", "match_distance": pair.distance,
                            "match_weight": float(control_use[pair.control_row_id])})
        rows.extend([treated_row, control_row])
    matched = pd.DataFrame(rows).drop_duplicates(["row_id", "match_role"])
    return matched, {"caliper_sd": score_sd, "caliper_value": caliper_value,
                     "matched_treated": len(pair_data), "unmatched_treated": len(treated) - len(pair_data),
                     "unique_controls": len(control_use), "total_pairs": len(pair_data)}


def weighted_smd(data: pd.DataFrame, variable: str) -> tuple[float, float, float]:
    treated = data.loc[data[TREAT].eq(1)]
    controls = data.loc[data[TREAT].eq(0)]
    treated_values = treated[variable].dropna()
    control_values = controls[variable].dropna()
    treated_weights = treated.loc[treated_values.index, "match_weight"]
    control_weights = controls.loc[control_values.index, "match_weight"]
    treated_mean = np.average(treated_values, weights=treated_weights)
    control_mean = np.average(control_values, weights=control_weights)
    treated_var = np.average((treated_values - treated_mean) ** 2, weights=treated_weights)
    control_var = np.average((control_values - control_mean) ** 2, weights=control_weights)
    denominator = np.sqrt((treated_var + control_var) / 2)
    smd = (treated_mean - control_mean) / denominator if denominator > 0 else np.nan
    return float(treated_mean), float(control_mean), float(smd)


def balance_tables(before: pd.DataFrame, after: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    balance_rows, test_rows = [], []
    for stage, data in [("匹配前", before), ("匹配后", after)]:
        for variable in LAGS:
            if stage == "匹配后":
                treated_mean, control_mean, smd = weighted_smd(data, variable)
            else:
                treated_values = data.loc[data[TREAT].eq(1), variable]
                control_values = data.loc[data[TREAT].eq(0), variable]
                treated_mean, control_mean = treated_values.mean(), control_values.mean()
                denominator = np.sqrt((treated_values.var(ddof=0) + control_values.var(ddof=0)) / 2)
                smd = (treated_mean - control_mean) / denominator if denominator > 0 else np.nan
            treated = data.loc[data[TREAT].eq(1), variable].dropna()
            controls = data.loc[data[TREAT].eq(0), variable].dropna()
            test = ttest_ind(treated, controls, equal_var=False, nan_policy="omit")
            balance_rows.append({"规格": name, "阶段": stage, "变量": variable,
                                 "处理组均值": treated_mean, "对照组均值": control_mean,
                                 "标准化均值差": smd, "绝对SMD小于0.10": bool(abs(smd) < .10)})
            test_rows.append({"规格": name, "阶段": stage, "变量": variable,
                              "Welch_t值": test.statistic, "p值": test.pvalue,
                              "均值差不显著": bool(test.pvalue >= .05)})
    return pd.DataFrame(balance_rows), pd.DataFrame(test_rows)


def outcome_model(data: pd.DataFrame, covariates: list[str], name: str) -> dict:
    regressors = [TREAT, *covariates]
    sample = data.dropna(subset=[GW, FIRM, YEAR, INDUSTRY, *regressors]).copy()
    formula = f"Q('{GW}') ~ " + " + ".join(regressors + [f"C({YEAR})", f"C({INDUSTRY})"])
    result = smf.wls(formula, sample, weights=sample["match_weight"]).fit(
        cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
    p_value = float(result.pvalues[TREAT])
    return {"规格": name, "系数": result.params[TREAT], "标准误": result.bse[TREAT],
            "t值": result.tvalues[TREAT], "p值": p_value, "显著性": significance(p_value),
            "样本数": int(result.nobs), "有效权重样本数": float(sample["match_weight"].sum()),
            "企业数": sample[FIRM].nunique(), "平均匹配距离": data["match_distance"].mean()}


def make_figures(before: pd.DataFrame, after: pd.DataFrame, balance: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.linspace(0, 1, 31)
    for treatment, label in [(0, "Control"), (1, "Treated")]:
        axes[0].hist(before.loc[before[TREAT].eq(treatment), "propensity_score"], bins=bins, alpha=.55, label=label, density=True)
        axes[1].hist(after.loc[after[TREAT].eq(treatment), "propensity_score"], bins=bins, alpha=.55, label=label, density=True, weights=after.loc[after[TREAT].eq(treatment), "match_weight"])
    axes[0].set_title("Before matching")
    axes[1].set_title("After matching")
    for axis in axes:
        axis.set_xlabel("Propensity score")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Density")
    figure.tight_layout()
    figure.savefig(OUTPUT / "figures/propensity_score_before_after.png", dpi=180)
    plt.close(figure)


def main() -> None:
    for directory in [OUTPUT / "tables", OUTPUT / "figures", OUTPUT / "matched_data"]:
        directory.mkdir(parents=True, exist_ok=True)
    data = load_data()
    results, all_balance, all_tests, propensity_rows, flow_rows = [], [], [], [], []
    for name, covariates, factor in SPECIFICATIONS:
        initial = data.dropna(subset=[TREAT, FIRM, YEAR, INDUSTRY, *covariates]).copy()
        industry_variation = initial.groupby(INDUSTRY, observed=True)[TREAT].transform("nunique")
        sample = initial.loc[industry_variation.gt(1)].copy()
        sample, propensity = fit_propensity(sample, covariates)
        matched, matching_info = match_with_replacement(sample, factor)
        if matched.empty:
            raise RuntimeError(f"{name} 没有成功配对")
        balance, tests = balance_tables(sample, matched, name)
        regression = outcome_model(matched, covariates, name)
        results.append(regression)
        all_balance.append(balance)
        all_tests.append(tests)
        propensity_rows.append(tidy_propensity_model(propensity, name))
        flow_rows.append({"规格": name, "倾向模型样本": len(sample), "倾向模型处理组": int(sample[TREAT].sum()),
                          "成功配对数": matching_info["total_pairs"], "未匹配处理组": matching_info["unmatched_treated"],
                          "唯一对照企业": matching_info["unique_controls"], "匹配后观测数": len(matched),
                          "匹配后有效权重样本数": matched["match_weight"].sum(), "logit得分标准差": matching_info["caliper_sd"],
                          "caliper系数": factor, "logit尺度caliper": matching_info["caliper_value"],
                          "平均匹配距离": matched["match_distance"].mean(), "允许放回": True,
                          "精确匹配维度": "年度×行业", "匹配变量": " | ".join(covariates)})
        if name.startswith("主规格"):
            matched.to_csv(OUTPUT / "matched_data/psm_logit_replacement_main.csv", index=False, encoding="utf-8-sig")
            make_figures(sample, matched, balance)
    pd.DataFrame(results).to_csv(OUTPUT / "tables/outcome_regressions.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_balance, ignore_index=True).to_csv(OUTPUT / "tables/covariate_balance.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_tests, ignore_index=True).to_csv(OUTPUT / "tables/covariate_mean_tests.csv", index=False, encoding="utf-8-sig")
    pd.concat(propensity_rows, ignore_index=True).to_csv(OUTPUT / "tables/propensity_model.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow_rows).to_csv(OUTPUT / "tables/matching_sample_flow.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow_rows).to_csv(OUTPUT / "tables/matching_parameters.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(flow_rows).to_string(index=False))
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
