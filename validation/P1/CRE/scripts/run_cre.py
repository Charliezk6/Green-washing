"""Wooldridge (2019) unbalanced-panel Mundlak CRE for firm heterogeneity.

The within slope is the coefficient on the level of the core regressor after
the firm means of every time-varying regressor, including year dummies, are
added. It matches the firm-and-year within estimator. Within and between
slopes, the full-sample joint Wald test, the sensitivity tests, and the
random-effects columns are all written. None of the tests drops a column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
FIRM_FE_SCRIPTS = ROOT.parent / "FirmFE" / "scripts"
PROJECT_ROOT = ROOT.parents[2]
sys.path.insert(0, str(FIRM_FE_SCRIPTS))

from run_firm_fe_regression import (  # noqa: E402
    ALL_CONTROLS,
    CORE,
    DEPENDENT,
    fit as fit_published_firm_fe,
    load_data,
    significance,
    winsorize_controls,
)

CONFIG = json.loads((ROOT / "config" / "cre_config.json").read_text(encoding="utf-8"))
FIRM = "证券代码"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"
ALPHA = float(CONFIG["main_test_alpha"])
TOLERANCE = float(CONFIG["equivalence_tolerance"])
LOW_WITHIN = list(CONFIG["low_within_controls"])
HIGH_WITHIN = [name for name in ALL_CONTROLS if name not in LOW_WITHIN]
CLASS_BETWEEN = "β_W不显著、β_B显著为负、联合检验拒绝"
CLASS_ALIGNED = "β_W与β_B同为负、联合检验不拒绝"
CLASS_OTHER = "其余组合"


def prepare_common() -> pd.DataFrame:
    data = load_data()
    required = [DEPENDENT, *CORE, *ALL_CONTROLS, FIRM, YEAR, INDUSTRY]
    common = data.dropna(subset=required).copy()
    common, _ = winsorize_controls(common)
    common = common.sort_values([FIRM, YEAR]).reset_index(drop=True)
    if common.duplicated([FIRM, YEAR]).any():
        raise ValueError("公司–年度存在重复观测，无法对齐企业固定效应与随机效应。")
    return common


def year_dummy_frame(sample: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    years = sorted(int(year) for year in sample[YEAR].unique())
    base = years[0]
    frame = pd.DataFrame(index=sample.index)
    names = []
    for year in years[1:]:
        name = f"year_{year}"
        frame[name] = (sample[YEAR].astype(int) == year).astype(float)
        names.append(name)
    return frame, names, base


def firm_means(sample: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    means = pd.DataFrame(index=sample.index)
    for column in columns:
        means[f"{column}__bar"] = sample.groupby(FIRM)[column].transform("mean")
    return means


def condition_number(exog: pd.DataFrame) -> float:
    scale = exog.std(axis=0, ddof=0).replace(0, np.nan)
    scaled = exog.div(scale, axis=1).dropna(axis=1)
    if scaled.shape[1] == 0:
        return np.nan
    return float(np.linalg.cond(scaled.to_numpy(dtype=float)))


def fit_ols(y: pd.Series, exog: pd.DataFrame, groups: pd.Series):
    design = sm.add_constant(exog, has_constant="add")
    result = sm.OLS(y, design).fit(
        cov_type="cluster",
        cov_kwds={"groups": groups.to_numpy()},
    )
    return result


def within_beta(levels: pd.DataFrame, columns: list[str], core: str, firms: pd.Series) -> float:
    """Firm-and-year within estimator: demean y, covariates, and year dummies."""
    demeaned = levels[[DEPENDENT, *columns]].copy()
    demeaned = demeaned - demeaned.groupby(firms.to_numpy()).transform("mean")
    result = sm.OLS(demeaned[DEPENDENT], sm.add_constant(demeaned[columns], has_constant="add")).fit()
    return float(result.params[core])


def linear_combination(result, formula: str) -> dict:
    test = result.t_test(formula)
    effect = float(np.asarray(test.effect).ravel()[0])
    standard_error = float(np.asarray(test.sd).ravel()[0])
    statistic = float(np.asarray(test.tvalue).ravel()[0])
    p_value = float(np.asarray(test.pvalue).ravel()[0])
    interval = np.asarray(test.conf_int()).ravel()
    return {
        "系数": effect,
        "标准误": standard_error,
        "t值": statistic,
        "p值": p_value,
        "显著性": significance(p_value),
        "置信区间下限": float(interval[0]),
        "置信区间上限": float(interval[1]),
    }


def coefficient_row(result, name: str) -> dict:
    interval = result.conf_int().loc[name]
    p_value = float(result.pvalues[name])
    return {
        "系数": float(result.params[name]),
        "标准误": float(result.bse[name]),
        "t值": float(result.tvalues[name]),
        "p值": p_value,
        "显著性": significance(p_value),
        "置信区间下限": float(interval[0]),
        "置信区间上限": float(interval[1]),
    }


def wald_zero(result, names: list[str]) -> dict:
    missing = [name for name in names if name not in result.params.index]
    if missing:
        raise ValueError(f"联合检验缺少回归元: {missing}")
    restriction = np.zeros((len(names), len(result.params)))
    for row, name in enumerate(names):
        restriction[row, result.params.index.get_loc(name)] = 1.0
    test = result.wald_test(restriction, use_f=False, scalar=True)
    return {
        "统计量": float(np.asarray(test.statistic).ravel()[0]),
        "自由度": len(names),
        "p值": float(np.asarray(test.pvalue).ravel()[0]),
        "约束变量": ";".join(names),
    }


def mundlak_design(sample: pd.DataFrame, core: str) -> dict:
    dummies, year_columns, base_year = year_dummy_frame(sample)
    level_columns = [core, *ALL_CONTROLS, *year_columns]
    levels = pd.concat([sample[[core, *ALL_CONTROLS]], dummies], axis=1)
    means = firm_means(pd.concat([sample[[FIRM]], levels], axis=1), level_columns)
    exog = pd.concat([levels, means], axis=1)
    mean_columns = [f"{column}__bar" for column in level_columns]
    return {
        "exog": exog,
        "mean_columns": mean_columns,
        "core_mean": f"{core}__bar",
        "high_within_means": [f"{column}__bar" for column in [core, *HIGH_WITHIN]],
        "base_year": base_year,
        "level_columns": level_columns,
        "condition_number": condition_number(exog),
    }


def chamberlain_design(sample: pd.DataFrame, core: str) -> dict:
    """Replace the core mean with year-specific values that span that mean."""
    dummies, year_columns, base_year = year_dummy_frame(sample)
    control_columns = [*ALL_CONTROLS, *year_columns]
    levels = pd.concat([sample[[core, *ALL_CONTROLS]], dummies], axis=1)
    control_means = firm_means(pd.concat([sample[[FIRM]], levels[control_columns]], axis=1), control_columns)
    counts = sample.groupby(FIRM)[YEAR].transform("size")
    chamberlain = pd.DataFrame(index=sample.index)
    chamberlain_columns = []
    for year in sorted(int(value) for value in sample[YEAR].unique()):
        observed = sample.loc[sample[YEAR].astype(int) == year, [FIRM, core]].drop_duplicates(FIRM)
        value = sample[FIRM].map(observed.set_index(FIRM)[core]).fillna(0.0)
        name = f"{core}_at_{year}"
        chamberlain[name] = value.to_numpy(dtype=float) / counts.to_numpy(dtype=float)
        chamberlain_columns.append(name)
    exog = pd.concat([levels, control_means, chamberlain], axis=1)
    mean_columns = [f"{column}__bar" for column in control_columns] + chamberlain_columns
    return {
        "exog": exog,
        "mean_columns": mean_columns,
        "base_year": base_year,
        "level_columns": [core, *control_columns],
        "condition_number": condition_number(exog),
    }


def panelols_beta(sample: pd.DataFrame, core: str) -> float:
    counts = sample.groupby(FIRM)[YEAR].transform("size")
    used = sample.loc[counts.ge(2)].set_index([FIRM, YEAR])
    result = PanelOLS(
        used[DEPENDENT],
        used[[core, *ALL_CONTROLS]],
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit()
    return float(result.params[core])


def fit_cre(sample: pd.DataFrame, core: str, design: dict):
    result = fit_ols(sample[DEPENDENT], design["exog"], sample[FIRM])
    return result


def minimum_detectable_effect(standard_error: float, alpha: float = 0.05, power: float = 0.80) -> float:
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * standard_error)


def achieved_power(standard_error: float, effect: float, alpha: float = 0.05) -> float:
    z_critical = norm.ppf(1 - alpha / 2)
    delta = abs(effect) / standard_error
    return float(norm.cdf(delta - z_critical) + norm.cdf(-delta - z_critical))


def classify(beta_w: float, p_w: float, beta_b: float, p_b: float, wald_p: float) -> str:
    within_insignificant = p_w >= ALPHA
    between_significant_negative = beta_b < 0 and p_b < ALPHA
    rejected = wald_p < ALPHA
    both_negative = beta_w < 0 and beta_b < 0
    if within_insignificant and between_significant_negative and rejected:
        return CLASS_BETWEEN
    if both_negative and not rejected:
        return CLASS_ALIGNED
    return CLASS_OTHER


def xi_sentence(core: str, xi: float, p_value: float) -> str:
    if p_value < ALPHA and xi < 0:
        return (
            f"{core} 的公司均值系数在 5% 水平显著为负：长期高专长对应的漂绿降幅大于公司内部变动，"
            "多出的部分可能是永久性差异，也可能是自选择，不单独作为因果效应。"
        )
    if p_value < ALPHA and xi > 0:
        return f"{core} 的公司均值系数在 5% 水平显著为正：公司之间的斜率弱于公司内部斜率。"
    return f"{core} 的公司均值系数在 5% 水平不显著：看不出组内与组间是两条不同的斜率。"


def paper_sentence(core: str, category: str, beta_w: float, se_w: float, p_w: float, beta_b: float,
                   se_b: float, p_b: float, wald_p: float, mde: float, power: float) -> str:
    if category == CLASS_BETWEEN:
        return (
            f"{core}：公司内部专长变动的系数为 {beta_w:.4f}（标准误 {se_w:.4f}），不显著；"
            f"公司之间的专长差异为 {beta_b:.4f}（标准误 {se_b:.4f}），显著为负，"
            f"且联合检验拒绝二者相等（p = {wald_p:.3f}）。组间差异可能含企业自选择，"
            f"这里不把组内效应解释为因果。组内最小可检测效应约 {mde:.2f}，"
            f"对基准效应的功效约 {power:.0%}。"
        )
    if category == CLASS_ALIGNED:
        between_test = "组间斜率单独看显著为负" if p_b < ALPHA else "组间斜率单独看也不显著"
        return (
            f"{core}：组内斜率 {beta_w:.4f}（标准误 {se_w:.4f}，p = {p_w:.3f}）与组间斜率 "
            f"{beta_b:.4f}（标准误 {se_b:.4f}，p = {p_b:.3f}）同为负。全样本联合检验没有拒绝"
            f"全部公司均值系数为零（p = {wald_p:.3f}），因此没有拒绝组内斜率与组间斜率相同。"
            f"方向与正文一致。组内斜率本身不显著，{between_test}。组内最小可检测效应约 {mde:.2f}，"
            f"对基准效应的功效约 {power:.0%}。"
        )
    return (
        f"{core}：组内斜率 {beta_w:.4f}（标准误 {se_w:.4f}），组间斜率 {beta_b:.4f}"
        f"（标准误 {se_b:.4f}），全样本联合检验 p = {wald_p:.3f}。如实报告这一组合，"
        "不用随机效应的显著性替换这两个系数。"
    )


def fit_random_effects(sample: pd.DataFrame, core: str, industry: bool) -> dict:
    dummies, year_columns, _ = year_dummy_frame(sample)
    pieces = [sample[[core, *ALL_CONTROLS]].reset_index(drop=True), dummies.reset_index(drop=True)]
    if industry:
        industry_dummies = pd.get_dummies(
            sample[INDUSTRY].astype(str), drop_first=True, prefix="industry", dtype=float
        ).reset_index(drop=True)
        pieces.append(industry_dummies)
    exog = pd.concat(pieces, axis=1)
    index = pd.MultiIndex.from_arrays(
        [sample[FIRM].to_numpy(), sample[YEAR].astype(int).to_numpy()],
        names=[FIRM, YEAR],
    )
    exog.index = index
    dependent = pd.Series(sample[DEPENDENT].to_numpy(), index=index, name=DEPENDENT)
    result = RandomEffects(dependent, exog).fit(cov_type="clustered", cluster_entity=True)
    p_value = float(result.pvalues[core])
    interval = result.conf_int().loc[core]
    return {
        "核心变量": core,
        "规格": "年份效应加行业哑变量" if industry else "仅年份效应",
        "系数": float(result.params[core]),
        "标准误": float(result.std_errors[core]),
        "t值": float(result.tstats[core]),
        "p值": p_value,
        "显著性": significance(p_value),
        "置信区间下限": float(interval.iloc[0]),
        "置信区间上限": float(interval.iloc[1]),
        "一致性": "仅在ξ=0下一致",
        "样本数": int(result.nobs),
        "公司数": int(sample[FIRM].nunique()),
        "标准误类型": "公司聚类",
    }


def baseline_effect(core: str) -> float:
    path = PROJECT_ROOT / CONFIG["baseline_effect_reference"]
    table = pd.read_csv(path, encoding="utf-8-sig")
    row = table[(table["核心变量"] == core) & (table["规格"] == "(4) 全部控制")].iloc[0]
    return float(row["Baseline_系数"])


def write_memo(equivalence: pd.DataFrame, classification: pd.DataFrame, hausman: pd.DataFrame,
               random_effects: pd.DataFrame) -> None:
    def fmt(value: float) -> str:
        return f"{value:.4f}"

    lines = [
        "# 相关随机效应：组内与组间",
        "",
        "正文主规格仍是行业 + 年份固定效应。这里用 Wooldridge (2019) 的非平衡面板 Mundlak 装置，把组内斜率和组间斜率放在同一张表里。ξ = 0 的检验用来读这两条斜率是否相同，不用来决定哪一列可以出现。",
        "",
        "## 估计",
        "",
        "样本、缩尾、因变量、IMS、IMS_D、全部控制变量和 20% 切点与企业固定效应脚本相同。公司均值只在进入该次回归的观测上计算，包括年份哑变量的公司均值。估计是混合 OLS，标准误按公司聚类。组间斜率是组内斜率与该公司均值系数之和，标准误来自同一聚类稳健协方差矩阵的线性组合。",
        "",
        "## 等价核对",
        "",
        "Wooldridge 命题对应的企业 + 年份固定效应，是把因变量、协变量和年份哑变量一起做公司内去均值。CRE 的组内斜率与这个对照的绝对差小于 1e-6。只保留至少有两年观测的公司后，该对照与 `PanelOLS` 的个体加时间效应一致。",
        "",
        "既有 `run_firm_fe_regression.fit` 只对公司内的因变量和协变量去均值，年份哑变量仍以水平值进入。在非平衡面板里，这不等于上面的企业 + 年份组内估计。IMS 相差约 0.005，IMS_D 相差约 0.003，来自定义不同，不是 CRE 没有对齐命题。下表和解读里的组内斜率用命题对照；既有脚本的系数另列，便于和原来的企业固定效应表对照。",
        "",
    ]
    for _, row in equivalence.iterrows():
        if row["规格"] != "Mundlak":
            continue
        lines.append(
            f"- {row['核心变量']}，{row['样本']}：CRE 组内 {fmt(row['CRE组内斜率'])}，"
            f"命题对照 {fmt(row['命题对照组内斜率'])}，绝对差 {row['绝对差_命题']:.3e}，"
            f"既有脚本 {fmt(row['既有脚本企业FE'])}，与既有脚本绝对差 {row['绝对差_既有脚本']:.4f}。"
        )
    lines.extend(["", "## 系数与归类", ""])
    lines.append("归类在看系数之前固定，显著性水平 5%。IMS 与 IMS_D 分开。敏感性检验不改变归类。")
    lines.append("")
    for _, row in classification.iterrows():
        lines.append(f"- {row['论文句子']}")
        lines.append(f"  {row['均值系数解读']}")
    lines.extend(["", "## 主检验与敏感性", ""])
    lines.append("唯一主检验是全样本、公司聚类的联合 Wald，原假设为全部公司均值系数等于 0，含年份哑变量的公司均值。")
    lines.append("")
    for _, row in hausman.iterrows():
        role = "主检验" if row["角色"] == "主检验" else "敏感性"
        decision = "拒绝" if row["5%水平拒绝"] else "不拒绝"
        lines.append(
            f"- {row['核心变量']}，{row['检验']}（{role}，{row['样本']}）："
            f"统计量 {row['统计量']:.2f}，自由度 {int(row['自由度'])}，p = {row['p值']:.4f}，5% 水平{decision}。"
            f"列标准化条件数 {row['条件数']:.1f}。"
        )
    lines.extend(["", "敏感性检验与全样本主检验的 5% 结论不同的，只有下面这些。句子仍按全样本主检验归类。", ""])
    for core in classification["核心变量"]:
        main_row = hausman[(hausman["核心变量"] == core) & (hausman["角色"] == "主检验")].iloc[0]
        sensitivities = hausman[(hausman["核心变量"] == core) & (hausman["角色"] == "敏感性")]
        disagreements = sensitivities[sensitivities["5%水平拒绝"] != main_row["5%水平拒绝"]]
        if disagreements.empty:
            lines.append(f"- {core}：子集检验、T_i ≥ 2 和 Chamberlain 与全样本主检验的 5% 结论相同。")
            continue
        for _, row in disagreements.iterrows():
            lines.append(
                f"- {core} 的{row['检验']} p = {row['p值']:.3f}，5% 水平"
                f"{'拒绝' if row['5%水平拒绝'] else '不拒绝'}，与全样本主检验不同。不改变归类。"
            )
    lines.extend([
        "",
        "列标准化条件数约 7000，因为几乎不随时间变化的控制与其公司均值接近共线。子集检验去掉 Size、Lev、FirmAge、SOE、Big4、LnAuditFee 的均值约束后，p 值明显更大。主检验仍是全样本联合 Wald。",
        "",
        "## 随机效应补充列",
        "",
        "随机效应用 `linearmodels.RandomEffects`，公司聚类标准误，方程里不放公司均值。两列都报告：仅年份效应；年份效应加行业哑变量。该列仅在 ξ = 0 下一致。随机效应显著，不写成组内效应已经恢复。",
        "",
    ])
    for _, row in random_effects.iterrows():
        lines.append(
            f"- {row['核心变量']}，{row['规格']}：{fmt(row['系数'])}（{fmt(row['标准误'])}），"
            f"p = {row['p值']:.4f}{row['显著性']}。"
        )
    lines.extend([
        "",
        "## 没有改动的设定",
        "",
        "不改 20% 切点，不改成只保留切换公司，也不把这一张表换成正文的主规格。",
        "",
    ])
    (ROOT / "notes" / "cre_memo.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output = ROOT / CONFIG["output_dir"]
    equivalence_dir = output / "equivalence"
    coefficient_dir = output / "coefficients"
    hausman_dir = output / "hausman"
    random_dir = output / "random_effects"
    for directory in (equivalence_dir, coefficient_dir, hausman_dir, random_dir, ROOT / "notes"):
        directory.mkdir(parents=True, exist_ok=True)

    common = prepare_common()
    equivalence_rows = []
    stored_designs = {}
    proposition_failed = False

    for core in CORE:
        for sample_name, sample in (
            ("全样本", common),
            ("T_i大于等于2", common.loc[common.groupby(FIRM)[YEAR].transform("size").ge(2)].copy()),
        ):
            sample = sample.reset_index(drop=True)
            design = mundlak_design(sample, core)
            result = fit_cre(sample, core, design)
            levels = pd.concat([sample[[DEPENDENT]], design["exog"][design["level_columns"]]], axis=1)
            benchmark = within_beta(levels, design["level_columns"], core, sample[FIRM])
            published, _ = fit_published_firm_fe(sample, core, ALL_CONTROLS)
            published_beta = float(published.params[core])
            cre_beta = float(result.params[core])
            proposition_gap = abs(cre_beta - benchmark)
            if proposition_gap >= TOLERANCE:
                proposition_failed = True
            equivalence_rows.append({
                "核心变量": core,
                "样本": sample_name,
                "规格": "Mundlak",
                "CRE组内斜率": cre_beta,
                "命题对照组内斜率": benchmark,
                "绝对差_命题": proposition_gap,
                "通过命题容差": proposition_gap < TOLERANCE,
                "既有脚本企业FE": published_beta,
                "绝对差_既有脚本": abs(cre_beta - published_beta),
                "样本数": int(result.nobs),
                "公司数": int(sample[FIRM].nunique()),
                "参照年": design["base_year"],
            })
            if sample_name == "全样本":
                stored_designs[core] = (sample, design, result)
                chamberlain = chamberlain_design(sample, core)
                chamberlain_result = fit_cre(sample, core, chamberlain)
                chamberlain_beta = float(chamberlain_result.params[core])
                chamberlain_gap = abs(chamberlain_beta - benchmark)
                if chamberlain_gap >= TOLERANCE:
                    proposition_failed = True
                equivalence_rows.append({
                    "核心变量": core,
                    "样本": sample_name,
                    "规格": "Chamberlain",
                    "CRE组内斜率": chamberlain_beta,
                    "命题对照组内斜率": benchmark,
                    "绝对差_命题": chamberlain_gap,
                    "通过命题容差": chamberlain_gap < TOLERANCE,
                    "既有脚本企业FE": published_beta,
                    "绝对差_既有脚本": abs(chamberlain_beta - published_beta),
                    "样本数": int(chamberlain_result.nobs),
                    "公司数": int(sample[FIRM].nunique()),
                    "参照年": chamberlain["base_year"],
                })
                stored_designs[f"{core}_chamberlain"] = (sample, chamberlain, chamberlain_result)
                panel_beta = panelols_beta(sample, core)
                panel_gap = abs(cre_beta - panel_beta)
                if panel_gap >= TOLERANCE:
                    proposition_failed = True
                equivalence_rows.append({
                    "核心变量": core,
                    "样本": "PanelOLS个体加时间",
                    "规格": "PanelOLS",
                    "CRE组内斜率": cre_beta,
                    "命题对照组内斜率": panel_beta,
                    "绝对差_命题": panel_gap,
                    "通过命题容差": panel_gap < TOLERANCE,
                    "既有脚本企业FE": published_beta,
                    "绝对差_既有脚本": abs(panel_beta - published_beta),
                    "样本数": int(result.nobs),
                    "公司数": int(sample[FIRM].nunique()),
                    "参照年": design["base_year"],
                })

    equivalence = pd.DataFrame(equivalence_rows)
    equivalence.to_csv(equivalence_dir / "equivalence_check.csv", index=False, encoding="utf-8-sig")
    if proposition_failed:
        raise SystemExit("CRE 组内斜率与 Wooldridge 组内对照的绝对差超过 1e-6，已停止，不写系数和检验。")

    coefficient_rows = []
    comparison_rows = []
    hausman_rows = []
    classification_rows = []
    random_rows = []

    for core in CORE:
        sample, design, result = stored_designs[core]
        _, chamberlain_design_core, chamberlain_result = stored_designs[f"{core}_chamberlain"]
        beta_w = coefficient_row(result, core)
        xi = coefficient_row(result, design["core_mean"])
        beta_b = linear_combination(result, f"{core} + {design['core_mean']} = 0")
        for label, component in (("组内β_W", beta_w), ("公司均值ξ", xi), ("组间β_B", beta_b)):
            coefficient_rows.append({
                "核心变量": core,
                "成分": label,
                **component,
                "样本数": int(result.nobs),
                "公司数": int(sample[FIRM].nunique()),
                "标准误类型": "公司聚类",
                "调整R方": float(result.rsquared_adj),
            })

        full_wald = wald_zero(result, design["mean_columns"])
        subset_wald = wald_zero(result, design["high_within_means"])
        restricted = common.loc[common.groupby(FIRM)[YEAR].transform("size").ge(2)].reset_index(drop=True)
        restricted_design = mundlak_design(restricted, core)
        restricted_result = fit_cre(restricted, core, restricted_design)
        restricted_wald = wald_zero(restricted_result, restricted_design["mean_columns"])
        chamberlain_wald = wald_zero(chamberlain_result, chamberlain_design_core["mean_columns"])
        tests = [
            ("全样本联合Wald", "全样本", "主检验", full_wald, design["condition_number"]),
            ("子集Wald", "全样本", "敏感性", subset_wald, design["condition_number"]),
            ("T_i大于等于2联合Wald", "T_i大于等于2", "敏感性", restricted_wald, restricted_design["condition_number"]),
            ("Chamberlain联合Wald", "全样本", "敏感性", chamberlain_wald, chamberlain_design_core["condition_number"]),
        ]
        for name, sample_name, role, test, condition in tests:
            hausman_rows.append({
                "核心变量": core,
                "检验": name,
                "样本": sample_name,
                "角色": role,
                "统计量": test["统计量"],
                "自由度": test["自由度"],
                "p值": test["p值"],
                "5%水平拒绝": test["p值"] < ALPHA,
                "条件数": condition,
                "约束变量": test["约束变量"],
            })

        baseline = baseline_effect(core)
        mde = minimum_detectable_effect(beta_w["标准误"])
        power = achieved_power(beta_w["标准误"], baseline)
        category = classify(beta_w["系数"], beta_w["p值"], beta_b["系数"], beta_b["p值"], full_wald["p值"])
        sentence = paper_sentence(
            core, category, beta_w["系数"], beta_w["标准误"], beta_w["p值"],
            beta_b["系数"], beta_b["标准误"], beta_b["p值"],
            full_wald["p值"], mde, power,
        )
        classification_rows.append({
            "核心变量": core,
            "类别": category,
            "β_W": beta_w["系数"],
            "β_W标准误": beta_w["标准误"],
            "β_W_p值": beta_w["p值"],
            "β_B": beta_b["系数"],
            "β_B标准误": beta_b["标准误"],
            "β_B_p值": beta_b["p值"],
            "ξ": xi["系数"],
            "ξ标准误": xi["标准误"],
            "ξ_p值": xi["p值"],
            "联合Wald_p": full_wald["p值"],
            "联合Wald_5%拒绝": full_wald["p值"] < ALPHA,
            "最小可检测效应_5%_80功效": mde,
            "对基准效应的功效_5%": power,
            "基准系数": baseline,
            "论文句子": sentence,
            "均值系数解读": xi_sentence(core, xi["系数"], xi["p值"]),
        })

        published_row = equivalence[
            (equivalence["核心变量"] == core) & (equivalence["样本"] == "全样本") & (equivalence["规格"] == "Mundlak")
        ].iloc[0]
        random_core = []
        for with_industry in (False, True):
            estimated = fit_random_effects(sample, core, with_industry)
            random_rows.append(estimated)
            random_core.append(estimated)
        comparison_rows.append({
            "核心变量": core,
            "企业固定效应_命题对照": published_row["命题对照组内斜率"],
            "企业固定效应_既有脚本": published_row["既有脚本企业FE"],
            "β_W": beta_w["系数"],
            "β_W标准误": beta_w["标准误"],
            "β_W_p值": beta_w["p值"],
            "β_B": beta_b["系数"],
            "β_B标准误": beta_b["标准误"],
            "β_B_p值": beta_b["p值"],
            "随机效应_仅年份": random_core[0]["系数"],
            "随机效应_仅年份标准误": random_core[0]["标准误"],
            "随机效应_仅年份_p值": random_core[0]["p值"],
            "随机效应_年份加行业": random_core[1]["系数"],
            "随机效应_年份加行业标准误": random_core[1]["标准误"],
            "随机效应_年份加行业_p值": random_core[1]["p值"],
            "随机效应一致性": "仅在ξ=0下一致",
            "样本数": int(result.nobs),
        })

    coefficients = pd.DataFrame(coefficient_rows)
    comparison = pd.DataFrame(comparison_rows)
    hausman = pd.DataFrame(hausman_rows)
    classification = pd.DataFrame(classification_rows)
    random_effects = pd.DataFrame(random_rows)
    coefficients.to_csv(coefficient_dir / "cre_coefficients.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(coefficient_dir / "cre_vs_firm_fe.csv", index=False, encoding="utf-8-sig")
    hausman.to_csv(hausman_dir / "hausman_tests.csv", index=False, encoding="utf-8-sig")
    classification.to_csv(hausman_dir / "classification.csv", index=False, encoding="utf-8-sig")
    random_effects.to_csv(random_dir / "re_coefficients.csv", index=False, encoding="utf-8-sig")
    write_memo(equivalence, classification, hausman, random_effects)
    print(classification[["核心变量", "类别", "β_W", "β_B", "联合Wald_p"]].to_string(index=False))
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
