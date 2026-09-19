from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT = PROJECT_ROOT / "Regression_reproduction/outputs/baseline/已构造基准回归变量的面板.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, TREAT = "Greenwashing连续_基准", "IMS_D"
BASE = ["Greenwashing连续_基准", "Size", "Lev", "ROA", "CFO", "Growth", "FirmAgeLog",
        "SOE", "Dual", "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee",
        "AuditorTenure", "IsChangeFirm"]
LAGS = [f"L_{variable}" for variable in BASE]
CALIPER = 0.03
METHOD = "Schroeder式PSM_Probit_1比1无放回"


def significance(p_value: float) -> str:
    return "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig")
    required = [FIRM, YEAR, INDUSTRY, GW, TREAT, *BASE]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    for column in [YEAR, GW, TREAT, *BASE]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values([FIRM, YEAR]).copy()
    previous_year = data.groupby(FIRM)[YEAR].shift(1)
    consecutive = data[YEAR].sub(previous_year).eq(1)
    for column in BASE:
        data[f"L_{column}"] = data.groupby(FIRM)[column].shift(1).where(consecutive)
    data["row_id"] = np.arange(len(data))
    return data


def tidy_propensity_model(result) -> pd.DataFrame:
    intervals = result.conf_int()
    rows = []
    for variable in result.params.index:
        if variable.startswith("C("):
            continue
        p_value = float(result.pvalues[variable])
        rows.append({"变量": variable, "系数": result.params[variable], "标准误": result.bse[variable],
                     "z值": result.tvalues[variable], "p值": p_value, "显著性": significance(p_value),
                     "置信区间下限": intervals.loc[variable, 0], "置信区间上限": intervals.loc[variable, 1],
                     "样本数": int(result.nobs), "收敛": bool(result.mle_retvals.get("converged", False)),
                     "迭代次数": result.mle_retvals.get("iterations")})
    return pd.DataFrame(rows)


def common_support(data: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    treated = data.loc[data[TREAT].eq(1), "propensity_score"]
    controls = data.loc[data[TREAT].eq(0), "propensity_score"]
    lower = max(float(treated.min()), float(controls.min()))
    upper = min(float(treated.max()), float(controls.max()))
    return data.loc[data["propensity_score"].between(lower, upper)].copy(), lower, upper


def optimal_pairs(data: pd.DataFrame, caliper: float) -> tuple[pd.DataFrame, int]:
    treated = data.loc[data[TREAT].eq(1)].sort_values("row_id")
    controls = data.loc[data[TREAT].eq(0)].sort_values("row_id")
    if treated.empty or controls.empty:
        return pd.DataFrame(columns=["pair_id", "treated_row_id", "control_row_id", "distance"]), len(treated)
    costs = np.abs(treated["propensity_score"].to_numpy()[:, None] -
                   controls["propensity_score"].to_numpy()[None, :])
    penalty = caliper + 1.0
    assignment_costs = np.where(costs <= caliper, costs, penalty)
    treated_index, control_index = linear_sum_assignment(assignment_costs)
    pairs = []
    for treated_position, control_position in zip(treated_index, control_index):
        distance = float(costs[treated_position, control_position])
        if distance > caliper:
            continue
        pairs.append({"pair_id": len(pairs) + 1,
                      "treated_row_id": int(treated.iloc[treated_position]["row_id"]),
                      "control_row_id": int(controls.iloc[control_position]["row_id"]),
                      "distance": distance})
    return pd.DataFrame(pairs), len(treated) - len(pairs)


def expand_pairs(data: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    indexed = data.set_index("row_id")
    rows = []
    for pair in pairs.itertuples(index=False):
        for role, row_id in [("treated", pair.treated_row_id), ("control", pair.control_row_id)]:
            row = indexed.loc[row_id].to_dict()
            row.update({"row_id": row_id, "pair_id": pair.pair_id, "match_role": role,
                        "match_distance": pair.distance, "matching_method": METHOD, "match_weight": 1.0})
            rows.append(row)
    return pd.DataFrame(rows)


def standardized_mean_difference(data: pd.DataFrame, variable: str) -> tuple[float, float, float]:
    treated = data.loc[data[TREAT].eq(1), variable]
    controls = data.loc[data[TREAT].eq(0), variable]
    denominator = np.sqrt((treated.var(ddof=0) + controls.var(ddof=0)) / 2)
    smd = (treated.mean() - controls.mean()) / denominator if denominator > 0 else np.nan
    return float(treated.mean()), float(controls.mean()), float(smd)


def balance_tables(before: pd.DataFrame, after: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    balance_rows, test_rows = [], []
    for stage, data in [("匹配前", before), ("匹配后", after)]:
        for variable in LAGS:
            treated_mean, control_mean, smd = standardized_mean_difference(data, variable)
            treated = data.loc[data[TREAT].eq(1), variable]
            controls = data.loc[data[TREAT].eq(0), variable]
            test = ttest_ind(treated, controls, equal_var=False, nan_policy="omit")
            balance_rows.append({"匹配方法": METHOD, "阶段": stage, "变量": variable,
                                 "处理组均值": treated_mean, "对照组均值": control_mean,
                                 "标准化均值差": smd,
                                 "绝对SMD小于0.10": bool(abs(smd) < .10) if pd.notna(smd) else False})
            test_rows.append({"匹配方法": METHOD, "阶段": stage, "变量": variable,
                              "处理组均值": treated_mean, "对照组均值": control_mean,
                              "均值差": treated_mean - control_mean, "Welch_t值": test.statistic,
                              "p值": test.pvalue, "均值差不显著": bool(test.pvalue >= .05)})
    return pd.DataFrame(balance_rows), pd.DataFrame(test_rows)


def outcome_models(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specifications = [("处理变量+固定效应", [TREAT]),
                      ("完整滞后协变量+固定效应", [TREAT, *LAGS])]
    for model_name, regressors in specifications:
        sample = data.dropna(subset=[GW, FIRM, YEAR, INDUSTRY, *regressors]).copy()
        formula = f"Q('{GW}') ~ " + " + ".join(regressors + [f"C({YEAR})", f"C({INDUSTRY})"])
        try:
            result = smf.ols(formula, sample).fit(
                cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
            p_value = float(result.pvalues[TREAT])
            rows.append({"匹配方法": METHOD, "模型": model_name, "解释变量": TREAT,
                         "系数": result.params[TREAT], "标准误": result.bse[TREAT],
                         "t值": result.tvalues[TREAT], "p值": p_value, "显著性": significance(p_value),
                         "样本数": int(result.nobs), "企业数": sample[FIRM].nunique(), "错误": ""})
        except Exception as exception:
            rows.append({"匹配方法": METHOD, "模型": model_name, "解释变量": TREAT,
                         "系数": np.nan, "标准误": np.nan, "t值": np.nan, "p值": np.nan,
                         "显著性": "", "样本数": len(sample), "企业数": sample[FIRM].nunique(),
                         "错误": f"{type(exception).__name__}: {exception}"})
    return pd.DataFrame(rows)


def make_figures(before: pd.DataFrame, after: pd.DataFrame, balance: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.linspace(0, 1, 31)
    for treatment, label in [(0, "Control"), (1, "Treated")]:
        axes[0].hist(before.loc[before[TREAT].eq(treatment), "propensity_score"], bins=bins,
                     alpha=.55, label=label, density=True)
        axes[1].hist(after.loc[after[TREAT].eq(treatment), "propensity_score"], bins=bins,
                     alpha=.55, label=label, density=True)
    axes[0].set_title("Before matching")
    axes[1].set_title("After Schroeder-style PSM")
    for axis in axes:
        axis.set_xlabel("Propensity score")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Density")
    figure.tight_layout()
    figure.savefig(OUTPUT / "figures/propensity_score_before_after.png", dpi=180)
    plt.close(figure)

    plot_data = balance.copy()
    plot_data["绝对SMD"] = plot_data["标准化均值差"].abs()
    variables = LAGS[::-1]
    display_variables = ["L_GW" if variable == "L_Greenwashing连续_基准" else variable for variable in variables]
    positions = np.arange(len(variables))
    figure, axis = plt.subplots(figsize=(8, 7))
    for stage, label, marker in [("匹配前", "Before", "o"), ("匹配后", "After", "s")]:
        values = plot_data.loc[plot_data["阶段"].eq(stage)].set_index("变量").reindex(variables)
        axis.scatter(values["绝对SMD"], positions, label=label, marker=marker)
    axis.axvline(.10, color="red", linestyle="--", linewidth=1, label="|SMD| = 0.10")
    axis.set_yticks(positions, display_variables)
    axis.set_xlabel("Absolute standardized mean difference")
    axis.set_title("Covariate balance")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figures/covariate_balance_love_plot.png", dpi=180)
    plt.close(figure)


def main() -> None:
    for directory in [OUTPUT / "tables", OUTPUT / "figures", OUTPUT / "matched_data"]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_data()
    initial = data.dropna(subset=[TREAT, FIRM, YEAR, INDUSTRY, *LAGS]).copy()
    industry_variation = initial.groupby(INDUSTRY, observed=True)[TREAT].transform("nunique")
    sample = initial.loc[industry_variation.gt(1)].copy()
    fixed_effect_dropped = initial.loc[industry_variation.eq(1)].copy()
    dropped_industries = " | ".join(map(str, sorted(fixed_effect_dropped[INDUSTRY].unique())))

    formula = f"{TREAT} ~ " + " + ".join(LAGS + [f"C({YEAR})", f"C({INDUSTRY})"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        propensity_model = smf.probit(formula, sample).fit(disp=False, maxiter=500)
    if not propensity_model.mle_retvals.get("converged", False):
        raise RuntimeError("Schroeder 式 Probit 倾向得分模型未收敛")
    severe_warnings = [str(item.message) for item in caught if "converg" in str(item.message).lower()]
    if severe_warnings:
        raise RuntimeError("Probit 倾向得分模型出现收敛警告: " + " | ".join(severe_warnings))

    sample["propensity_score"] = propensity_model.predict(sample)
    support, lower, upper = common_support(sample)
    pairs, unmatched = optimal_pairs(support, CALIPER)
    if pairs.empty:
        raise RuntimeError("共同支撑内没有处理组在 0.03 caliper 内找到对照组")
    matched = expand_pairs(support, pairs)

    balance, mean_tests = balance_tables(sample, matched)
    regressions = outcome_models(matched)
    propensity_table = tidy_propensity_model(propensity_model)
    propensity_table.to_csv(OUTPUT / "tables/propensity_model.csv", index=False, encoding="utf-8-sig")
    balance.to_csv(OUTPUT / "tables/covariate_balance.csv", index=False, encoding="utf-8-sig")
    mean_tests.to_csv(OUTPUT / "tables/covariate_mean_tests.csv", index=False, encoding="utf-8-sig")
    regressions.to_csv(OUTPUT / "tables/outcome_regressions.csv", index=False, encoding="utf-8-sig")
    matched.to_csv(OUTPUT / "matched_data/psm_schroeder_without_replacement.csv", index=False,
                   encoding="utf-8-sig")

    support_deleted = sample.loc[~sample.index.isin(support.index)]
    flow = pd.DataFrame([{"匹配方法": METHOD, "输入完整案例样本": len(initial),
                          "固定效应识别剔除行业数": fixed_effect_dropped[INDUSTRY].nunique(),
                          "固定效应识别剔除观测数": len(fixed_effect_dropped),
                          "固定效应识别剔除处理组数": int(fixed_effect_dropped[TREAT].sum()),
                          "倾向模型样本": len(sample), "倾向模型处理组": int(sample[TREAT].sum()),
                          "共同支撑样本": len(support),
                          "共同支撑删除处理组": int(support_deleted[TREAT].sum()),
                          "共同支撑删除对照组": int(support_deleted[TREAT].eq(0).sum()),
                          "成功配对数": len(pairs), "caliper内未匹配处理组": unmatched,
                          "匹配后观测数": len(matched), "匹配后企业数": matched[FIRM].nunique()}])
    flow.to_csv(OUTPUT / "tables/matching_sample_flow.csv", index=False, encoding="utf-8-sig")

    parameters = pd.DataFrame([{"匹配方法": METHOD, "倾向模型": "Probit", "匹配比例": "1:1",
                                "允许放回": False, "匹配算法": "全局最小距离最优指派",
                                "共同支撑下限": lower, "共同支撑上限": upper,
                                "caliper标尺": "原始倾向得分绝对差", "caliper": CALIPER,
                                "年度控制": "是", "行业控制": "是",
                                "倾向模型固定效应剔除行业": dropped_industries,
                                "Probit警告": " | ".join(str(item.message) for item in caught)}])
    parameters.to_csv(OUTPUT / "tables/matching_parameters.csv", index=False, encoding="utf-8-sig")
    make_figures(sample, matched, balance)

    failed_balance = balance.loc[balance["阶段"].eq("匹配后") & ~balance["绝对SMD小于0.10"], "变量"].tolist()
    print(flow.to_string(index=False))
    print(f"匹配后未通过|SMD|<0.10的变量: {failed_balance or '无'}")
    print(regressions.to_string(index=False))


if __name__ == "__main__":
    main()
