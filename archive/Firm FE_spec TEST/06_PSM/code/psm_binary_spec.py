"""Minutti-Meza (2013 JAR) style Logit PSM with treatment SPEC.

Same sample construction as binary EB (gps.load_data + lagged complete cases).
Treatment: SPEC = 1 if the client's audit firm has the largest audit-fee market
share in the client's industry-year.

Two propensity specs:
  PS-Full: all 15 lagged covariates + year/industry FE
  PS-Size: L_Size only + year/industry FE

Matching: common support, 1:1 without replacement, caliper 0.03 on raw PS,
global minimum-distance assignment (Hungarian).

Outputs under test/06_PSM/results/. Exploration only — does not edit Paper.tex.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_ind

sys.dont_write_bytecode = True
TEST_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TEST_ROOT.parent
PSM_ROOT = Path(__file__).resolve().parents[1]
GPS_CODE = PROJECT_ROOT / "validation/P1/GPS-IPW/code/gps_ipw_analysis.py"
SPEC_PANEL = TEST_ROOT / "data/combined_regression_panel_15381_spec.csv"
EB_RESULTS = TEST_ROOT / "05_EB/results/binary_eb_regressions.csv"
OUT = PSM_ROOT / "results"
NOTES = PSM_ROOT / "notes"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, IMS, TREAT = "Greenwashing连续_基准", "IMS", "SPEC"
TREAT_DEFINITION = "审计收费行业-年度份额第一"
CALIPER = 0.03

FULL_BASE = [
    GW, "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE", "Dual",
    "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee",
    "AuditorTenure", "IsChangeFirm",
]
FULL_LAGS = [f"L_{column}" for column in FULL_BASE]
SIZE_LAGS = ["L_Size"]

SPECS = [
    ("PS-Full", "MinuttiMeza_Logit_Full_1to1_norepl_cal0.03", FULL_LAGS),
    ("PS-Size", "MinuttiMeza_Logit_Size_1to1_norepl_cal0.03", SIZE_LAGS),
]


def load_gps_module():
    spec = importlib.util.spec_from_file_location("gps_ipw_analysis", GPS_CODE)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载GPS代码: {GPS_CODE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.INPUT = SPEC_PANEL
    return module


def significance(p_value: float) -> str:
    return "***" if p_value < .01 else "**" if p_value < .05 else "*" if p_value < .10 else ""


def q_term(name: str) -> str:
    return f"Q('{name}')"


def load_analysis_sample() -> pd.DataFrame:
    gps = load_gps_module()
    data = gps.load_data()
    required = [GW, IMS, TREAT, FIRM, YEAR, INDUSTRY, *FULL_LAGS]
    data[IMS] = pd.to_numeric(data[IMS], errors="coerce")
    data[TREAT] = pd.to_numeric(data[TREAT], errors="coerce")
    sample = data.dropna(subset=required).copy()
    sample[TREAT] = sample[TREAT].astype(int)
    if sample[TREAT].nunique() != 2:
        raise ValueError(f"{TREAT} 必须同时包含0和1")
    sample = sample.reset_index(drop=True)
    sample["row_id"] = np.arange(len(sample))
    return sample


def tidy_propensity_model(result, spec_label: str, method: str) -> pd.DataFrame:
    intervals = result.conf_int()
    rows = []
    for variable in result.params.index:
        if variable.startswith("C("):
            continue
        p_value = float(result.pvalues[variable])
        rows.append({
            "规格": spec_label, "匹配方法": method, "变量": variable,
            "系数": result.params[variable], "标准误": result.bse[variable],
            "z值": result.tvalues[variable], "p值": p_value,
            "显著性": significance(p_value),
            "置信区间下限": intervals.loc[variable, 0],
            "置信区间上限": intervals.loc[variable, 1],
            "样本数": int(result.nobs),
            "收敛": bool(result.mle_retvals.get("converged", False)),
            "迭代次数": result.mle_retvals.get("iterations"),
        })
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
        empty = pd.DataFrame(columns=["pair_id", "treated_row_id", "control_row_id", "distance"])
        return empty, len(treated)
    costs = np.abs(
        treated["propensity_score"].to_numpy()[:, None]
        - controls["propensity_score"].to_numpy()[None, :]
    )
    penalty = caliper + 1.0
    assignment_costs = np.where(costs <= caliper, costs, penalty)
    treated_index, control_index = linear_sum_assignment(assignment_costs)
    pairs = []
    for treated_position, control_position in zip(treated_index, control_index):
        distance = float(costs[treated_position, control_position])
        if distance > caliper:
            continue
        pairs.append({
            "pair_id": len(pairs) + 1,
            "treated_row_id": int(treated.iloc[treated_position]["row_id"]),
            "control_row_id": int(controls.iloc[control_position]["row_id"]),
            "distance": distance,
        })
    return pd.DataFrame(pairs), len(treated) - len(pairs)


def expand_pairs(data: pd.DataFrame, pairs: pd.DataFrame, method: str) -> pd.DataFrame:
    indexed = data.set_index("row_id")
    rows = []
    for pair in pairs.itertuples(index=False):
        for role, row_id in [("treated", pair.treated_row_id), ("control", pair.control_row_id)]:
            row = indexed.loc[row_id].to_dict()
            row.update({
                "row_id": row_id, "pair_id": pair.pair_id, "match_role": role,
                "match_distance": pair.distance, "matching_method": method,
                "match_weight": 1.0,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def standardized_mean_difference(data: pd.DataFrame, variable: str) -> tuple[float, float, float]:
    treated = data.loc[data[TREAT].eq(1), variable]
    controls = data.loc[data[TREAT].eq(0), variable]
    denominator = np.sqrt((treated.var(ddof=0) + controls.var(ddof=0)) / 2)
    smd = (treated.mean() - controls.mean()) / denominator if denominator > 0 else np.nan
    return float(treated.mean()), float(controls.mean()), float(smd)


def balance_tables(
    before: pd.DataFrame, after: pd.DataFrame, spec_label: str, method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    balance_rows, test_rows = [], []
    for stage, data in [("匹配前", before), ("匹配后", after)]:
        for variable in FULL_LAGS:
            treated_mean, control_mean, smd = standardized_mean_difference(data, variable)
            treated = data.loc[data[TREAT].eq(1), variable]
            controls = data.loc[data[TREAT].eq(0), variable]
            test = ttest_ind(treated, controls, equal_var=False, nan_policy="omit")
            balance_rows.append({
                "规格": spec_label, "匹配方法": method, "阶段": stage, "变量": variable,
                "处理组均值": treated_mean, "对照组均值": control_mean,
                "标准化均值差": smd,
                "绝对SMD小于0.10": bool(abs(smd) < .10) if pd.notna(smd) else False,
            })
            test_rows.append({
                "规格": spec_label, "匹配方法": method, "阶段": stage, "变量": variable,
                "处理组均值": treated_mean, "对照组均值": control_mean,
                "均值差": treated_mean - control_mean, "Welch_t值": test.statistic,
                "p值": test.pvalue, "均值差不显著": bool(test.pvalue >= .05),
            })
    return pd.DataFrame(balance_rows), pd.DataFrame(test_rows)


def outcome_models(data: pd.DataFrame, spec_label: str, method: str) -> pd.DataFrame:
    rows = []
    specifications = [
        ("M1_处理变量+固定效应", [TREAT]),
        ("M2_完整滞后协变量+固定效应", [TREAT, *FULL_LAGS]),
    ]
    for model_name, regressors in specifications:
        sample = data.dropna(subset=[GW, FIRM, YEAR, INDUSTRY, *regressors]).copy()
        formula = (
            f"{q_term(GW)} ~ "
            + " + ".join([q_term(x) for x in regressors]
                         + [f"C({q_term(YEAR)})", f"C({q_term(INDUSTRY)})"])
        )
        try:
            result = smf.ols(formula, sample).fit(
                cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
            term = q_term(TREAT)
            p_value = float(result.pvalues[term])
            rows.append({
                "规格": spec_label, "匹配方法": method, "模型": model_name,
                "解释变量": TREAT, "系数": result.params[term],
                "标准误": result.bse[term], "t值": result.tvalues[term],
                "p值": p_value, "显著性": significance(p_value),
                "样本数": int(result.nobs), "企业数": sample[FIRM].nunique(),
                "调整R方": result.rsquared_adj, "错误": "",
            })
        except Exception as exception:
            rows.append({
                "规格": spec_label, "匹配方法": method, "模型": model_name,
                "解释变量": TREAT, "系数": np.nan, "标准误": np.nan,
                "t值": np.nan, "p值": np.nan, "显著性": "",
                "样本数": len(sample), "企业数": sample[FIRM].nunique(),
                "调整R方": np.nan,
                "错误": f"{type(exception).__name__}: {exception}",
            })
    return pd.DataFrame(rows)


def make_figures(
    before: pd.DataFrame, after: pd.DataFrame, balance: pd.DataFrame,
    suffix: str, title_tag: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    bins = np.linspace(0, 1, 31)
    for treatment, label in [(0, "Control"), (1, "Treated")]:
        axes[0].hist(
            before.loc[before[TREAT].eq(treatment), "propensity_score"],
            bins=bins, alpha=.55, label=label, density=True)
        axes[1].hist(
            after.loc[after[TREAT].eq(treatment), "propensity_score"],
            bins=bins, alpha=.55, label=label, density=True)
    axes[0].set_title("Before matching")
    axes[1].set_title(f"After PSM ({title_tag})")
    for axis in axes:
        axis.set_xlabel("Propensity score")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Density")
    figure.tight_layout()
    path = OUT / "figures" / f"propensity_score_before_after_{suffix}.png"
    figure.savefig(path, dpi=180)
    if suffix == "full":
        figure.savefig(OUT / "figures" / "propensity_score_before_after.png", dpi=180)
    plt.close(figure)

    plot_data = balance.copy()
    plot_data["绝对SMD"] = plot_data["标准化均值差"].abs()
    variables = FULL_LAGS[::-1]
    display_variables = [
        "L_GW" if variable == f"L_{GW}" else variable for variable in variables
    ]
    positions = np.arange(len(variables))
    figure, axis = plt.subplots(figsize=(8, 7))
    for stage, label, marker in [("匹配前", "Before", "o"), ("匹配后", "After", "s")]:
        values = plot_data.loc[plot_data["阶段"].eq(stage)].set_index("变量").reindex(variables)
        axis.scatter(values["绝对SMD"], positions, label=label, marker=marker)
    axis.axvline(.10, color="red", linestyle="--", linewidth=1, label="|SMD| = 0.10")
    axis.set_yticks(positions, display_variables)
    axis.set_xlabel("Absolute standardized mean difference")
    axis.set_title(f"Covariate balance ({title_tag})")
    axis.legend(frameon=False)
    figure.tight_layout()
    path = OUT / "figures" / f"covariate_balance_love_plot_{suffix}.png"
    figure.savefig(path, dpi=180)
    if suffix == "full":
        figure.savefig(OUT / "figures" / "covariate_balance_love_plot.png", dpi=180)
    plt.close(figure)


def run_one_spec(
    initial: pd.DataFrame, spec_label: str, method: str, covariates: list[str],
) -> dict:
    industry_variation = initial.groupby(INDUSTRY, observed=True)[TREAT].transform("nunique")
    sample = initial.loc[industry_variation.gt(1)].copy()
    fixed_effect_dropped = initial.loc[industry_variation.eq(1)].copy()
    dropped_industries = " | ".join(map(str, sorted(fixed_effect_dropped[INDUSTRY].unique())))

    formula = (
        f"{q_term(TREAT)} ~ "
        + " + ".join([q_term(c) for c in covariates]
                     + [f"C({q_term(YEAR)})", f"C({q_term(INDUSTRY)})"])
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        propensity_model = smf.logit(formula, sample).fit(disp=False, maxiter=500)
    if not propensity_model.mle_retvals.get("converged", False):
        raise RuntimeError(f"{spec_label} Logit 倾向得分模型未收敛")
    severe_warnings = [
        str(item.message) for item in caught if "converg" in str(item.message).lower()
    ]
    if severe_warnings:
        raise RuntimeError(
            f"{spec_label} Logit 出现收敛警告: " + " | ".join(severe_warnings)
        )

    sample = sample.copy()
    sample["propensity_score"] = propensity_model.predict(sample)
    support, lower, upper = common_support(sample)
    pairs, unmatched = optimal_pairs(support, CALIPER)
    if pairs.empty:
        raise RuntimeError(f"{spec_label}: 共同支撑内没有处理组在 0.03 caliper 内找到对照组")
    matched = expand_pairs(support, pairs, method)

    balance, mean_tests = balance_tables(sample, matched, spec_label, method)
    regressions = outcome_models(matched, spec_label, method)
    propensity_table = tidy_propensity_model(propensity_model, spec_label, method)

    support_deleted = sample.loc[~sample.index.isin(support.index)]
    flow = {
        "规格": spec_label, "匹配方法": method,
        "处理定义": TREAT_DEFINITION,
        "输入完整案例样本": len(initial),
        "完整案例处理组": int(initial[TREAT].sum()),
        "完整案例对照组": int(initial[TREAT].eq(0).sum()),
        "固定效应识别剔除行业数": fixed_effect_dropped[INDUSTRY].nunique(),
        "固定效应识别剔除观测数": len(fixed_effect_dropped),
        "固定效应识别剔除处理组数": int(fixed_effect_dropped[TREAT].sum()),
        "倾向模型样本": len(sample),
        "倾向模型处理组": int(sample[TREAT].sum()),
        "共同支撑样本": len(support),
        "共同支撑删除处理组": int(support_deleted[TREAT].sum()),
        "共同支撑删除对照组": int(support_deleted[TREAT].eq(0).sum()),
        "成功配对数": len(pairs),
        "caliper内未匹配处理组": unmatched,
        "匹配后观测数": len(matched),
        "匹配后企业数": matched[FIRM].nunique(),
        "匹配后处理组": int(matched[TREAT].sum()),
        "匹配后对照组": int(matched[TREAT].eq(0).sum()),
    }
    parameters = {
        "规格": spec_label, "匹配方法": method,
        "倾向模型": "Logit", "匹配比例": "1:1", "允许放回": False,
        "匹配算法": "全局最小距离最优指派",
        "共同支撑下限": lower, "共同支撑上限": upper,
        "caliper标尺": "原始倾向得分绝对差", "caliper": CALIPER,
        "处理定义": TREAT_DEFINITION,
        "倾向协变量": " | ".join(covariates),
        "年度控制": "是", "行业控制": "是",
        "倾向模型固定效应剔除行业": dropped_industries,
        "Logit警告": " | ".join(str(item.message) for item in caught),
    }
    return {
        "flow": flow,
        "parameters": parameters,
        "propensity": propensity_table,
        "balance": balance,
        "mean_tests": mean_tests,
        "regressions": regressions,
        "matched": matched,
        "before": sample,
        "after": matched,
    }


def write_memo(
    flow: pd.DataFrame, balance: pd.DataFrame, regressions: pd.DataFrame,
) -> None:
    lines = [
        "# PSM（SPEC：审计收费行业-年度份额第一）内部稳健性备忘",
        "",
        "> 探索输出，不改正文。参照 Minutti-Meza (2013 JAR)：Logit + 共同支撑 + 1:1 无放回 + caliper 0.03。",
        "> 样本与二元 EB 同口径；处理变量为 `SPEC`。",
        "",
        "## 样本与处理",
        "",
        f"- 处理定义：{TREAT_DEFINITION}",
        f"- 完整案例 N = {int(flow['输入完整案例样本'].iloc[0]):,}",
        f"- 处理 / 对照（完整案例）= "
        f"{int(flow['完整案例处理组'].iloc[0]):,} / {int(flow['完整案例对照组'].iloc[0]):,}",
        "",
        "## 样本流程",
        "",
        "| 规格 | 倾向模型N | 处理组 | 共同支撑 | 配对数 | 未匹配处理 | 匹配后N | 企业数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in flow.iterrows():
        lines.append(
            f"| {row['规格']} | {int(row['倾向模型样本']):,} | "
            f"{int(row['倾向模型处理组']):,} | {int(row['共同支撑样本']):,} | "
            f"{int(row['成功配对数']):,} | {int(row['caliper内未匹配处理组']):,} | "
            f"{int(row['匹配后观测数']):,} | {int(row['匹配后企业数']):,} |"
        )

    lines.extend(["", "## 匹配后平衡（|SMD|<0.10）", ""])
    for spec in flow["规格"]:
        after = balance.loc[balance["规格"].eq(spec) & balance["阶段"].eq("匹配后")]
        failed = after.loc[~after["绝对SMD小于0.10"], "变量"].tolist()
        max_abs = after["标准化均值差"].abs().max()
        lines.append(
            f"- **{spec}**：匹配后 max|SMD|={max_abs:.4f}；"
            f"未达标变量：{failed or '无'}"
        )

    lines.extend(["", "## 匹配后结果回归", "",
                  "| 规格 | 模型 | 系数 | SE | t | p | 显著性 | N | 企业 |",
                  "|---|---|---:|---:|---:|---:|---|---:|---:|"])
    for _, row in regressions.iterrows():
        coef = row["系数"]
        se = row["标准误"]
        t_val = row["t值"]
        p_val = row["p值"]
        if pd.isna(coef):
            lines.append(
                f"| {row['规格']} | {row['模型']} | — | — | — | — | "
                f"| {int(row['样本数']):,} | {int(row['企业数']):,} |"
            )
        else:
            lines.append(
                f"| {row['规格']} | {row['模型']} | {coef:.4f} | {se:.4f} | "
                f"{t_val:.3f} | {p_val:.4f} | {row['显著性']} | "
                f"{int(row['样本数']):,} | {int(row['企业数']):,} |"
            )

    full_m2 = regressions.loc[
        regressions["规格"].eq("PS-Full") & regressions["模型"].str.startswith("M2")
    ]
    size_failed = balance.loc[
        balance["规格"].eq("PS-Size") & balance["阶段"].eq("匹配后")
        & ~balance["绝对SMD小于0.10"], "变量"
    ].tolist()
    m2_coef = float(full_m2["系数"].iloc[0]) if not full_m2.empty else float("nan")
    m2_p = float(full_m2["p值"].iloc[0]) if not full_m2.empty else float("nan")
    m2_n = int(full_m2["样本数"].iloc[0]) if not full_m2.empty else 0
    eb_line = "- EB 结果文件不存在，先运行 `test/05_EB/code/entropy_balancing_analysis_spec.py`"
    if EB_RESULTS.exists():
        eb = pd.read_csv(EB_RESULTS, encoding="utf-8-sig")
        eb_main = eb.loc[eb["第一阶段规格"].eq("完整共同原因加审计选择")].iloc[0]
        eb_line = (f"- EB 主规格 B1（`SPEC`，完整共同原因加审计选择）：**{eb_main['系数']:.3f}**，"
                   f"p≈{eb_main['p值']:.3f}，N={int(eb_main['样本数']):,}，ESS≈{eb_main['有效样本量ESS']:,.0f}")
    lines.extend([
        "",
        "## 解读要点",
        "",
        "- **PS-Full**（主规格）：匹配后全部 15 个滞后协变量 `|SMD|<0.10`；"
        "M1 为 ATT 的简约规格，M2 为匹配样本上重估含全部滞后的主模型（Minutti-Meza 做法）。",
        f"- **PS-Size**（互补）：仅匹配规模时未平衡变量：{size_failed or '无'}——"
        "说明仅靠规模不足以消除可观测差异。",
        f"- FE 识别剔除行业数={int(flow['固定效应识别剔除行业数'].iloc[0])}，"
        f"剔除观测={int(flow['固定效应识别剔除观测数'].iloc[0]):,}；"
        "1:1 匹配后功效低于全样本 EB 属设计特征。",
        "",
        "## 与二元 EB（同一处理变量 SPEC）对照",
        "",
        eb_line,
        f"- PSM-Full M2：约 **{m2_coef:.3f}**，p≈{m2_p:.3f}，N={m2_n:,}",
        "",
        "## 表述边界",
        "",
        "- 可说：以审计收费份额第一（SPEC）定义行业专长时，按 Minutti-Meza 设定做 Logit PSM 后的组间结果。",
        "- 必须披露：处理定义与原 `IMS_D`（营业收入份额≥0.20）不同、匹配掉样本、仅为内部稳健性备忘。",
        "- 不可说：消除内生性、单独证明因果。",
        "",
        "## 复现",
        "",
        "```bash",
        "python test/06_PSM/code/psm_binary_spec.py",
        "```",
        "",
    ])
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / "psm_spec_memo.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for directory in [OUT / "tables", OUT / "figures", OUT / "matched_data"]:
        directory.mkdir(parents=True, exist_ok=True)

    initial = load_analysis_sample()
    n1 = int(initial[TREAT].sum())
    n0 = len(initial) - n1
    print(f"[PSM SPEC] N={len(initial)}, treated={n1}, control={n0}")

    flows, parameters, propensities = [], [], []
    balances, mean_tests, regressions = [], [], []

    for spec_label, method, covariates in SPECS:
        print(f"Running {spec_label} ...")
        result = run_one_spec(initial, spec_label, method, covariates)
        flows.append(result["flow"])
        parameters.append(result["parameters"])
        propensities.append(result["propensity"])
        balances.append(result["balance"])
        mean_tests.append(result["mean_tests"])
        regressions.append(result["regressions"])

        suffix = "full" if spec_label == "PS-Full" else "size"
        matched_name = f"psm_{suffix}.csv"
        result["matched"].to_csv(
            OUT / "matched_data" / matched_name, index=False, encoding="utf-8-sig")
        make_figures(
            result["before"], result["after"], result["balance"],
            suffix=suffix, title_tag=spec_label,
        )
        failed = result["balance"].loc[
            result["balance"]["阶段"].eq("匹配后")
            & ~result["balance"]["绝对SMD小于0.10"],
            "变量",
        ].tolist()
        print(f"  pairs={result['flow']['成功配对数']}, "
              f"matched_N={result['flow']['匹配后观测数']}, "
              f"failed_|SMD|<0.10={failed or '无'}")
        print(result["regressions"].to_string(index=False))

    flow_df = pd.DataFrame(flows)
    balance_df = pd.concat(balances, ignore_index=True)
    regressions_df = pd.concat(regressions, ignore_index=True)

    flow_df.to_csv(OUT / "tables" / "matching_sample_flow.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(parameters).to_csv(
        OUT / "tables" / "matching_parameters.csv", index=False, encoding="utf-8-sig")
    pd.concat(propensities, ignore_index=True).to_csv(
        OUT / "tables" / "propensity_model.csv", index=False, encoding="utf-8-sig")
    balance_df.to_csv(OUT / "tables" / "covariate_balance.csv", index=False, encoding="utf-8-sig")
    pd.concat(mean_tests, ignore_index=True).to_csv(
        OUT / "tables" / "covariate_mean_tests.csv", index=False, encoding="utf-8-sig")
    regressions_df.to_csv(
        OUT / "tables" / "outcome_regressions.csv", index=False, encoding="utf-8-sig")

    write_memo(flow_df, balance_df, regressions_df)
    print(f"Wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
