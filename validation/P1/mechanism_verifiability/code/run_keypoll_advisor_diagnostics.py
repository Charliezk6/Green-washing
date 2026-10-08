"""Advisor diagnostics for the KeyPoll heterogeneity: correlation, slopes, and specification choice.

The full interaction used here is not hetero_common.full_interact_formula. Controls, year
fixed effects, and industry fixed effects all interact with the moderator, matching the
split-sample regression. Inference uses that specification only when the joint Wald test
rejects that every interaction other than S x V is zero. Otherwise inference stays with
the constrained interaction.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from hetero_common import (  # noqa: E402
    CONTROLS,
    DEPENDENT,
    TABLES,
    Y,
    constrained_formula,
    controls_except,
    fit_cluster,
    load_analysis_panel,
    stars,
    subsample_formula,
)

WALD_ALPHA = 0.05


def as_float(value: object) -> float:
    return float(np.asarray(value, dtype=float).ravel()[0])


def mentions_moderator(name: str, moderator: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(moderator)}(?![A-Za-z0-9_])", name) is not None


def level_from_term(name: str) -> str | None:
    match = re.search(r"\[T\.(.+?)\]", name)
    return match.group(1) if match else None


def find_interaction(params: list[str], left: str, right: str) -> str | None:
    options = {f"{left}:{right}", f"{right}:{left}"}
    for name in params:
        if name in options:
            return name
    return None


def full_fe_interact_formula(core: str, moderator: str) -> str:
    """Full interaction with controls, year FE, and industry FE all varying by V.

    This is intentionally not hetero_common.full_interact_formula, which leaves the
    fixed effects common to both groups.
    """
    terms = [core, moderator, f"{core}:{moderator}"]
    for control in controls_except(moderator):
        terms.extend([control, f"{control}:{moderator}"])
    terms.extend(
        [
            "C(时间)",
            f"C(时间):{moderator}",
            "C(ExpertiseIndustryCode)",
            f"C(ExpertiseIndustryCode):{moderator}",
        ]
    )
    return f"{Y} ~ " + " + ".join(terms)


def linear_combination(result, weights: dict[str, float]) -> dict[str, float]:
    names = list(result.params.index)
    restriction = np.zeros((1, len(names)))
    for name, weight in weights.items():
        restriction[0, names.index(name)] = weight
    contrast = result.t_test(restriction)
    interval = np.asarray(contrast.conf_int(), dtype=float).ravel()
    return {
        "coefficient": as_float(contrast.effect),
        "std_error": as_float(contrast.sd),
        "t": as_float(contrast.tvalue),
        "p_value": as_float(contrast.pvalue),
        "ci_low": float(interval[0]),
        "ci_high": float(interval[1]),
    }


def coefficient_row(result, name: str) -> dict[str, float]:
    return linear_combination(result, {name: 1.0})


def joint_wald(result, names: list[str]) -> dict[str, object]:
    present = [name for name in names if name in result.params.index]
    if not present:
        return {
            "statistic": np.nan,
            "p_value": np.nan,
            "df_num": 0,
            "df_denom": np.nan,
            "distribution": "",
            "n_restrictions": 0,
            "notes": "no estimated terms",
        }
    names_index = list(result.params.index)
    restriction = np.zeros((len(present), len(names_index)))
    for row, name in enumerate(present):
        restriction[row, names_index.index(name)] = 1.0
    test = result.wald_test(restriction, scalar=True)
    raw_distribution = getattr(test, "distribution", "")
    distribution = raw_distribution if isinstance(raw_distribution, str) else type(raw_distribution).__name__
    # Cluster-robust OLS in this environment reports a chi-square Wald. statsmodels
    # stores that chi-square degrees of freedom, after any rank adjustment, in df_denom.
    chi_df = as_float(test.df_denom) if distribution == "chi2" else np.nan
    f_num = as_float(test.df_num) if getattr(test, "df_num", None) is not None else np.nan
    f_den = as_float(test.df_denom) if distribution == "F" else np.nan
    used_df = chi_df if distribution == "chi2" else f_num
    note = ""
    if np.isfinite(used_df) and int(used_df) < len(present):
        note = f"constraint covariance rank={int(used_df)}; p-value uses this rank"
    return {
        "statistic": as_float(test.statistic),
        "p_value": as_float(test.pvalue),
        "df_num": used_df if distribution == "chi2" else f_num,
        "df_denom": np.nan if distribution == "chi2" else f_den,
        "distribution": distribution,
        "n_restrictions": len(present),
        "notes": note,
    }


def blank_estimate() -> dict[str, float]:
    return {
        "coefficient": np.nan,
        "std_error": np.nan,
        "t": np.nan,
        "p_value": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
    }


def sample_counts(sample: pd.DataFrame, moderator: str) -> dict[str, int]:
    return {
        "n": int(len(sample)),
        "n_firms": int(sample["证券代码"].nunique()),
        "n_moderator_1": int(sample[moderator].eq(1).sum()),
        "n_moderator_0": int(sample[moderator].eq(0).sum()),
    }


def fe_reference(sample: pd.DataFrame, column: str, main_effect_names: list[str]) -> str:
    present = {level_from_term(name) for name in main_effect_names}
    present.discard(None)
    observed = []
    for value in sample[column].dropna().unique().tolist():
        text = str(int(value)) if isinstance(value, (int, np.integer)) or (isinstance(value, float) and float(value).is_integer()) else str(value)
        observed.append(text)
    missing = [value for value in sorted(set(observed)) if value not in present and f"{value}.0" not in present]
    return ",".join(missing)


def collect_terms(result, moderator: str) -> dict[str, list[str]]:
    params = list(result.params.index)
    control_names = []
    omitted_controls = []
    for control in controls_except(moderator):
        name = find_interaction(params, control, moderator)
        if name is None:
            omitted_controls.append(control)
        else:
            control_names.append(name)
    year_main = [name for name in params if "C(时间)" in name and ":" not in name]
    industry_main = [name for name in params if "ExpertiseIndustryCode" in name and ":" not in name]
    year_x = [name for name in params if "C(时间)" in name and mentions_moderator(name, moderator)]
    industry_x = [name for name in params if "ExpertiseIndustryCode" in name and mentions_moderator(name, moderator)]
    return {
        "control_x_v": control_names,
        "year_x_v": sorted(year_x, key=lambda name: level_from_term(name) or name),
        "industry_x_v": sorted(industry_x, key=lambda name: level_from_term(name) or name),
        "omitted_controls": omitted_controls,
        "year_main": year_main,
        "industry_main": industry_main,
    }


def omitted_fe(sample: pd.DataFrame, column: str, main_names: list[str], interaction_names: list[str]) -> list[str]:
    main_levels = {level_from_term(name) for name in main_names}
    main_levels.discard(None)
    interaction_levels = {level_from_term(name) for name in interaction_names}
    interaction_levels.discard(None)
    return sorted(main_levels - interaction_levels)


def evaluate_spec(frame: pd.DataFrame, spec: str, core: str, moderator: str) -> tuple[list[dict], list[dict], list[dict]]:
    needed = [DEPENDENT, core, moderator, "证券代码", "时间", "ExpertiseIndustryCode", *CONTROLS]
    sample = frame.dropna(subset=needed).reset_index(drop=True)
    counts = sample_counts(sample, moderator)
    constrained = fit_cluster(sample, constrained_formula(core, moderator))
    full = fit_cluster(sample, full_fe_interact_formula(core, moderator))
    terms = collect_terms(full, moderator)
    wald_controls = joint_wald(full, terms["control_x_v"])
    wald_fe = joint_wald(full, terms["year_x_v"] + terms["industry_x_v"])
    wald_all = joint_wald(full, terms["control_x_v"] + terms["year_x_v"] + terms["industry_x_v"])
    all_p = wald_all["p_value"]
    inference = "full" if pd.notna(all_p) and float(all_p) < WALD_ALPHA else "constrained"

    constrained_names = list(constrained.params.index)
    full_names = list(full.params.index)
    constrained_interaction = find_interaction(constrained_names, core, moderator)
    full_interaction = find_interaction(full_names, core, moderator)
    if core not in constrained_names or constrained_interaction is None:
        raise KeyError(f"Constrained model is missing {core} or its interaction with {moderator}")
    if core not in full_names or full_interaction is None:
        raise KeyError(f"Full model is missing {core} or its interaction with {moderator}")

    slope_rows: list[dict] = []
    lincom_rows: list[dict] = []

    def add_slope(layer: str, quantity: str, estimate: dict[str, float], result, group: str, n_obs: int, n_firms: int, adjusted_r2: float) -> None:
        row = {
            "spec": spec,
            "core": core,
            "moderator": moderator,
            "layer": layer,
            "quantity": quantity,
            "group": group,
            "stars": stars(estimate["p_value"]),
            "n": n_obs,
            "n_firms": n_firms,
            "n_moderator_1": counts["n_moderator_1"] if layer != "split" else np.nan,
            "adjusted_r2": adjusted_r2,
            "inference_layer": inference,
            "used_for_inference": int(layer == inference and quantity in {"beta", "gamma", "beta_plus_gamma"}),
        }
        row.update(estimate)
        slope_rows.append(row)
        if quantity == "beta_plus_gamma":
            lincom = dict(row)
            lincom["hypothesis"] = f"{core} + {core}:{moderator} = 0"
            lincom_rows.append(lincom)

    for layer, result, interaction in [
        ("constrained", constrained, constrained_interaction),
        ("full", full, full_interaction),
    ]:
        beta = coefficient_row(result, core)
        gamma = coefficient_row(result, interaction)
        total = linear_combination(result, {core: 1.0, interaction: 1.0})
        shared = {
            "result": result,
            "n_obs": int(result.nobs),
            "n_firms": counts["n_firms"],
            "adjusted_r2": float(result.rsquared_adj),
        }
        add_slope(layer, "beta", beta, **shared, group="0")
        add_slope(layer, "gamma", gamma, **shared, group="")
        add_slope(layer, "beta_plus_gamma", total, **shared, group="1")

    for value, tag in [(1, "1"), (0, "0")]:
        subset = sample[sample[moderator].eq(value)].reset_index(drop=True)
        split = fit_cluster(subset, subsample_formula(core))
        estimate = coefficient_row(split, core)
        add_slope(
            "split",
            "group_slope",
            estimate,
            split,
            tag,
            int(split.nobs),
            int(subset["证券代码"].nunique()),
            float(split.rsquared_adj),
        )

    other_rows: list[dict] = []
    year_reference = fe_reference(sample, "时间", terms["year_main"])
    industry_reference = fe_reference(sample, "ExpertiseIndustryCode", terms["industry_main"])
    other_rows.append(
        {
            "spec": spec,
            "core": core,
            "moderator": moderator,
            "term_group": "reference",
            "variable": "omitted_baseline",
            "statistic": np.nan,
            "df_num": np.nan,
            "df_denom": np.nan,
            "distribution": "",
            "n_restrictions": np.nan,
            "n": counts["n"],
            "n_firms": counts["n_firms"],
            "adjusted_r2": float(full.rsquared_adj),
            "inference_layer": inference,
            "notes": f"year reference={year_reference}; industry reference={industry_reference}",
            **blank_estimate(),
            "stars": "",
        }
    )

    if moderator not in full.params.index:
        raise KeyError(f"Full model is missing the main effect of {moderator}")
    main_effect = coefficient_row(full, moderator)
    other_rows.append(
        {
            "spec": spec,
            "core": core,
            "moderator": moderator,
            "term_group": "v_main",
            "variable": moderator,
            "statistic": np.nan,
            "df_num": np.nan,
            "df_denom": np.nan,
            "distribution": "",
            "n_restrictions": np.nan,
            "n": counts["n"],
            "n_firms": counts["n_firms"],
            "adjusted_r2": float(full.rsquared_adj),
            "inference_layer": inference,
            "notes": "intercept shift at the reference level of controls and fixed effects; not the average KeyPoll gap",
            "stars": stars(main_effect["p_value"]),
            **main_effect,
        }
    )

    def add_term(group: str, name: str, note: str) -> None:
        estimate = coefficient_row(full, name) if name in full.params.index else blank_estimate()
        other_rows.append(
            {
                "spec": spec,
                "core": core,
                "moderator": moderator,
                "term_group": group,
                "variable": name,
                "statistic": np.nan,
                "df_num": np.nan,
                "df_denom": np.nan,
                "distribution": "",
                "n_restrictions": np.nan,
                "n": counts["n"],
                "n_firms": counts["n_firms"],
                "adjusted_r2": float(full.rsquared_adj),
                "inference_layer": inference,
                "notes": note,
                "stars": stars(estimate["p_value"]),
                **estimate,
            }
        )

    for name in terms["control_x_v"]:
        add_term("control_x_v", name, "")
    for control in terms["omitted_controls"]:
        add_term("omitted", f"{control}:{moderator}", "omitted_collinearity")
    for name in terms["year_x_v"]:
        add_term("year_x_v", name, f"relative to year reference {year_reference}")
    for level in omitted_fe(sample, "时间", terms["year_main"], terms["year_x_v"]):
        add_term("omitted", f"C(时间)[T.{level}]:{moderator}", "omitted_collinearity")
    for name in terms["industry_x_v"]:
        add_term("industry_x_v", name, f"relative to industry reference {industry_reference}")
    for level in omitted_fe(sample, "ExpertiseIndustryCode", terms["industry_main"], terms["industry_x_v"]):
        add_term("omitted", f"C(ExpertiseIndustryCode)[T.{level}]:{moderator}", "omitted_collinearity")

    for label, test in [
        ("controls_x_v", wald_controls),
        ("fe_x_v", wald_fe),
        ("all_other_x_v", wald_all),
    ]:
        other_rows.append(
            {
                "spec": spec,
                "core": core,
                "moderator": moderator,
                "term_group": "wald",
                "variable": label,
                "coefficient": np.nan,
                "std_error": np.nan,
                "t": np.nan,
                "p_value": test["p_value"],
                "stars": stars(test["p_value"]) if pd.notna(test["p_value"]) else "",
                "ci_low": np.nan,
                "ci_high": np.nan,
                "statistic": test["statistic"],
                "df_num": test["df_num"],
                "df_denom": test["df_denom"],
                "distribution": test["distribution"],
                "n_restrictions": test["n_restrictions"],
                "n": counts["n"],
                "n_firms": counts["n_firms"],
                "adjusted_r2": float(full.rsquared_adj),
                "inference_layer": inference,
                "notes": test["notes"],
            }
        )

    other_rows.append(
        {
            "spec": spec,
            "core": core,
            "moderator": moderator,
            "term_group": "decision",
            "variable": "inference_layer",
            "coefficient": np.nan,
            "std_error": np.nan,
            "t": np.nan,
            "p_value": all_p,
            "stars": "",
            "ci_low": np.nan,
            "ci_high": np.nan,
            "statistic": np.nan,
            "df_num": wald_all["df_num"],
            "df_denom": wald_all["df_denom"],
            "distribution": wald_all["distribution"],
            "n_restrictions": wald_all["n_restrictions"],
            "n": counts["n"],
            "n_firms": counts["n_firms"],
            "adjusted_r2": float(full.rsquared_adj),
            "inference_layer": inference,
            "notes": (
                f"threshold={WALD_ALPHA}; constrained_r2={constrained.rsquared_adj:.6f}; "
                f"full_r2={full.rsquared_adj:.6f}; n_control_x={len(terms['control_x_v'])}; "
                f"n_year_x={len(terms['year_x_v'])}; n_industry_x={len(terms['industry_x_v'])}; "
                f"n_omitted_controls={len(terms['omitted_controls'])}"
            ),
        }
    )
    print(
        f"{spec} {core} x {moderator}: Wald all-other p={all_p:.4g} -> {inference}; "
        f"N={counts['n']}; full adjR2={full.rsquared_adj:.4f}"
    )
    return slope_rows, lincom_rows, other_rows


def residualize(sample: pd.DataFrame, column: str, formula_fe: str) -> pd.Series:
    fitted = smf.ols(f"{column} ~ {formula_fe}", data=sample).fit()
    return fitted.resid


def correlation_rows(data: pd.DataFrame) -> pd.DataFrame:
    needed = ["KeyPollMonUnit", "IMS", "IMS_D", "证券代码", "时间", "ExpertiseIndustryCode"]
    sample = data.dropna(subset=needed).copy()
    rows: list[dict] = []

    def add(block: str, variable: str, statistic: str, value: float, p_value: float, n: int, notes: str, std_error: float = np.nan) -> None:
        rows.append(
            {
                "block": block,
                "variable": variable,
                "statistic": statistic,
                "value": value,
                "std_error": std_error,
                "p_value": p_value,
                "n": n,
                "notes": notes,
            }
        )

    for variable in ["IMS", "IMS_D"]:
        pearson = stats.pearsonr(sample["KeyPollMonUnit"], sample[variable])
        spearman = stats.spearmanr(sample["KeyPollMonUnit"], sample[variable])
        add("association", variable, "pearson", float(pearson.statistic), float(pearson.pvalue), len(sample), "not cluster-robust")
        add("association", variable, "spearman", float(spearman.statistic), float(spearman.pvalue), len(sample), "not cluster-robust")

    for label, formula in [
        ("residual_industry_year", "C(时间) + C(ExpertiseIndustryCode)"),
        ("residual_industry", "C(ExpertiseIndustryCode)"),
    ]:
        residual_v = residualize(sample, "KeyPollMonUnit", formula)
        for variable in ["IMS", "IMS_D"]:
            residual_s = residualize(sample, variable, formula)
            pearson = stats.pearsonr(residual_v, residual_s)
            spearman = stats.spearmanr(residual_v, residual_s)
            add(label, variable, "pearson", float(pearson.statistic), float(pearson.pvalue), len(sample), "residuals after listed FE; p not cluster-robust")
            add(label, variable, "spearman", float(spearman.statistic), float(spearman.pvalue), len(sample), "residuals after listed FE; p not cluster-robust")

    for variable in ["IMS", "IMS_D"]:
        for value in [1, 0]:
            subset = sample.loc[sample["KeyPollMonUnit"].eq(value), variable]
            add("group_moment", variable, "mean", float(subset.mean()), np.nan, int(subset.shape[0]), f"KeyPollMonUnit={value}")
            add("group_moment", variable, "median", float(subset.median()), np.nan, int(subset.shape[0]), f"KeyPollMonUnit={value}")
            add("group_moment", variable, "std", float(subset.std(ddof=1)), np.nan, int(subset.shape[0]), f"KeyPollMonUnit={value}")
        difference = fit_cluster(sample, f"{variable} ~ KeyPollMonUnit")
        estimate = coefficient_row(difference, "KeyPollMonUnit")
        add(
            "mean_difference_cluster",
            variable,
            "KeyPoll1_minus_0",
            estimate["coefficient"],
            estimate["p_value"],
            int(difference.nobs),
            "OLS coefficient on KeyPollMonUnit; firm-clustered SE",
            estimate["std_error"],
        )
        high = sample.loc[sample["KeyPollMonUnit"].eq(1), variable]
        low = sample.loc[sample["KeyPollMonUnit"].eq(0), variable]
        rank = stats.mannwhitneyu(high, low, alternative="two-sided")
        add("ranksum", variable, "mannwhitney_u", float(rank.statistic), float(rank.pvalue), len(sample), "not cluster-robust")

    counts = pd.crosstab(sample["KeyPollMonUnit"], sample["IMS_D"])
    for keypoll in counts.index:
        for ims_d in counts.columns:
            count = int(counts.loc[keypoll, ims_d])
            add("crosstab", "IMS_D", "count", count, np.nan, len(sample), f"KeyPollMonUnit={int(keypoll)}; IMS_D={int(ims_d)}")
            row_n = int(counts.loc[keypoll].sum())
            add("crosstab", "IMS_D", "row_share", count / row_n, np.nan, row_n, f"KeyPollMonUnit={int(keypoll)}; IMS_D={int(ims_d)}")
    chi2 = stats.chi2_contingency(counts)
    add("crosstab", "IMS_D", "chi2", float(chi2.statistic), float(chi2.pvalue), len(sample), f"df={int(chi2.dof)}; not cluster-robust")
    return pd.DataFrame(rows)


def tex_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def tex_p(value: float) -> str:
    if pd.isna(value):
        return "--"
    if value < 0.001:
        return "$<$0.001"
    return f"{value:.3f}"


def tex_coef(value: float, p_value: float) -> str:
    if pd.isna(value):
        return "--"
    star = stars(p_value) if pd.notna(p_value) else ""
    body = f"{value:.3f}"
    if star:
        body += f"^{{{star}}}"
    return f"${body}$"


def tex_int(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{int(round(value)):,}".replace(",", "{,}")


def pick_slope(slopes: pd.DataFrame, spec: str, core: str, layer: str, quantity: str, group: str | None = None) -> pd.Series:
    hit = slopes[slopes["spec"].eq(spec) & slopes["core"].eq(core) & slopes["layer"].eq(layer) & slopes["quantity"].eq(quantity)]
    if group is not None:
        hit = hit[hit["group"].astype(str).eq(group)]
    if hit.empty:
        raise KeyError(f"missing slope {spec} {core} {layer} {quantity} {group}")
    return hit.iloc[0]


def pick_wald(others: pd.DataFrame, spec: str, core: str, variable: str) -> pd.Series:
    hit = others[others["spec"].eq(spec) & others["core"].eq(core) & others["term_group"].eq("wald") & others["variable"].eq(variable)]
    return hit.iloc[0]


def corr_value(correlation: pd.DataFrame, block: str, variable: str, statistic: str, notes_contains: str | None = None) -> pd.Series:
    hit = correlation[correlation["block"].eq(block) & correlation["variable"].eq(variable) & correlation["statistic"].eq(statistic)]
    if notes_contains is not None:
        hit = hit[hit["notes"].fillna("").str.contains(notes_contains, regex=False)]
    return hit.iloc[0]


def term_label(name: str) -> str:
    level = level_from_term(name)
    if level is not None and "时间" in name:
        return f"{level} $\\times V$"
    if level is not None and "ExpertiseIndustryCode" in name:
        return f"\\texttt{{{level}}} $\\times V$"
    if ":" in name:
        left = name.split(":", 1)[0].replace("_", r"\_")
        return f"\\texttt{{{left}}} $\\times V$"
    return name.replace("_", r"\_")


def write_advisor_tables(correlation: pd.DataFrame, slopes: pd.DataFrame, others: pd.DataFrame) -> None:
    lines: list[str] = []

    def coef_se(row: pd.Series) -> str:
        return f"{tex_coef(row['coefficient'], row['p_value'])} & {tex_num(row['std_error'])}"

    ims_mean_1 = corr_value(correlation, "group_moment", "IMS", "mean", "KeyPollMonUnit=1")
    ims_mean_0 = corr_value(correlation, "group_moment", "IMS", "mean", "KeyPollMonUnit=0")
    ims_med_1 = corr_value(correlation, "group_moment", "IMS", "median", "KeyPollMonUnit=1")
    ims_med_0 = corr_value(correlation, "group_moment", "IMS", "median", "KeyPollMonUnit=0")
    ims_sd_1 = corr_value(correlation, "group_moment", "IMS", "std", "KeyPollMonUnit=1")
    ims_sd_0 = corr_value(correlation, "group_moment", "IMS", "std", "KeyPollMonUnit=0")
    d_mean_1 = corr_value(correlation, "group_moment", "IMS_D", "mean", "KeyPollMonUnit=1")
    d_mean_0 = corr_value(correlation, "group_moment", "IMS_D", "mean", "KeyPollMonUnit=0")
    diff_ims = corr_value(correlation, "mean_difference_cluster", "IMS", "KeyPoll1_minus_0")
    diff_d = corr_value(correlation, "mean_difference_cluster", "IMS_D", "KeyPoll1_minus_0")
    pearson_ims = corr_value(correlation, "association", "IMS", "pearson")
    spearman_ims = corr_value(correlation, "association", "IMS", "spearman")
    pearson_d = corr_value(correlation, "association", "IMS_D", "pearson")
    spearman_d = corr_value(correlation, "association", "IMS_D", "spearman")
    partial_ims = corr_value(correlation, "residual_industry_year", "IMS", "pearson")
    partial_d = corr_value(correlation, "residual_industry_year", "IMS_D", "pearson")

    def cross(keypoll: int, ims_d: int, statistic: str) -> pd.Series:
        return corr_value(correlation, "crosstab", "IMS_D", statistic, f"KeyPollMonUnit={keypoll}; IMS_D={ims_d}")

    lines.extend(
        [
            "\\begin{center}",
            "\\begin{threeparttable}",
            "\\caption{重点排污单位与行业专长的相关}",
            "\\label{tab:keypoll-corr}",
            "\\begin{tabular}{lcc}",
            "\\toprule",
            " & \\texttt{IMS} & \\texttt{IMS\\_D} \\\\",
            "\\midrule",
            f"Pearson & {tex_num(pearson_ims['value'])} & {tex_num(pearson_d['value'])} \\\\",
            f"Spearman & {tex_num(spearman_ims['value'])} & {tex_num(spearman_d['value'])} \\\\",
            f"行业与年份残差后的 Pearson & {tex_num(partial_ims['value'])} & {tex_num(partial_d['value'])} \\\\",
            "\\midrule",
            f"重点单位均值 & {tex_num(ims_mean_1['value'])} & {tex_num(d_mean_1['value'])} \\\\",
            f"非重点单位均值 & {tex_num(ims_mean_0['value'])} & {tex_num(d_mean_0['value'])} \\\\",
            f"重点单位中位数 & {tex_num(ims_med_1['value'])} & 0 \\\\",
            f"非重点单位中位数 & {tex_num(ims_med_0['value'])} & 0 \\\\",
            f"重点单位标准差 & {tex_num(ims_sd_1['value'])} & {tex_num(corr_value(correlation, 'group_moment', 'IMS_D', 'std', 'KeyPollMonUnit=1')['value'])} \\\\",
            f"非重点单位标准差 & {tex_num(ims_sd_0['value'])} & {tex_num(corr_value(correlation, 'group_moment', 'IMS_D', 'std', 'KeyPollMonUnit=0')['value'])} \\\\",
            f"均值差（重点减非重点，公司聚类） & {tex_coef(diff_ims['value'], diff_ims['p_value'])} & {tex_coef(diff_d['value'], diff_d['p_value'])} \\\\",
            f"均值差的标准误 & {tex_num(diff_ims['std_error'])} & {tex_num(diff_d['std_error'])} \\\\",
            "\\midrule",
            f"\\texttt{{IMS\\_D}}$=0$，非重点 & \\multicolumn{{2}}{{c}}{{{tex_int(cross(0, 0, 'count')['value'])}（{tex_num(100 * cross(0, 0, 'row_share')['value'], 1)}\\%）}} \\\\",
            f"\\texttt{{IMS\\_D}}$=1$，非重点 & \\multicolumn{{2}}{{c}}{{{tex_int(cross(0, 1, 'count')['value'])}（{tex_num(100 * cross(0, 1, 'row_share')['value'], 1)}\\%）}} \\\\",
            f"\\texttt{{IMS\\_D}}$=0$，重点 & \\multicolumn{{2}}{{c}}{{{tex_int(cross(1, 0, 'count')['value'])}（{tex_num(100 * cross(1, 0, 'row_share')['value'], 1)}\\%）}} \\\\",
            f"\\texttt{{IMS\\_D}}$=1$，重点 & \\multicolumn{{2}}{{c}}{{{tex_int(cross(1, 1, 'count')['value'])}（{tex_num(100 * cross(1, 1, 'row_share')['value'], 1)}\\%）}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}\\footnotesize",
            "\\item 注：共同样本 $N=15{,}381$。均值差来自以 \\texttt{KeyPollMonUnit} 为唯一回归元的 OLS，标准误按公司聚类。Pearson、Spearman 与交叉表的 $p$ 值没有按公司聚类；残差相关是先对行业和年份虚拟变量回归后再算 Pearson。\\texttt{IMS\\_D} 的均值是组内取 1 的比例。",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{center}",
            "",
        ]
    )

    quantities = [
        ("beta", None, r"非重点单位斜率 \(\beta\)"),
        ("beta_plus_gamma", None, r"重点单位斜率 \(\beta+\gamma\)"),
        ("gamma", None, r"斜率差 \(\gamma\)"),
    ]
    lines.extend(
        [
            "\\begin{center}",
            "\\begin{threeparttable}",
            "\\caption{当期重点排污单位：\\texttt{IMS} 的组斜率}",
            "\\label{tab:keypoll-slopes-advisor}",
            "\\begin{tabular}{lcccccc}",
            "\\toprule",
            " & \\multicolumn{2}{c}{约束交互} & \\multicolumn{2}{c}{完全交互} & \\multicolumn{2}{c}{分组回归} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
            " & 系数 & 标准误 & 系数 & 标准误 & 系数 & 标准误 \\\\",
            "\\midrule",
        ]
    )
    for quantity, _group, label in quantities:
        constrained = pick_slope(slopes, "current", "IMS", "constrained", quantity)
        full = pick_slope(slopes, "current", "IMS", "full", quantity)
        if quantity == "gamma":
            split_cells = "-- & --"
        else:
            split_group = "0" if quantity == "beta" else "1"
            split = pick_slope(slopes, "current", "IMS", "split", "group_slope", split_group)
            split_cells = coef_se(split)
        lines.append(
            f"{label} & {coef_se(constrained)} & {coef_se(full)} & {split_cells} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}\\footnotesize",
            "\\item 注：完全交互让控制变量、年份固定效应和行业固定效应都与 \\texttt{KeyPollMonUnit} 相乘。联合检验拒绝其他交互项为零，推断用完全交互一列。该列的 \(\\beta\) 与 \(\\beta+\\gamma\) 等于分组回归的两个系数。分组回归不检验 \(\\gamma\)。公司聚类。$N=15{,}381$。",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{center}",
            "",
            "\\begin{center}",
            "\\begin{threeparttable}",
            "\\caption{重点单位专长斜率的线性组合检验}",
            "\\label{tab:keypoll-lincom}",
            "\\begin{tabular}{llccccc}",
            "\\toprule",
            "核心变量 & 规格 & \(\\beta+\\gamma\) & 标准误 & \(t\) & \(p\) & 用于推断 \\\\",
            "\\midrule",
        ]
    )
    for core, core_label in [("IMS", r"\texttt{IMS}"), ("IMS_D", r"\texttt{IMS\_D}")]:
        for layer, layer_label in [("constrained", "约束交互"), ("full", "完全交互")]:
            row = pick_slope(slopes, "current", core, layer, "beta_plus_gamma")
            used = "是" if layer == "full" else "否"
            lines.append(
                f"{core_label} & {layer_label} & {tex_coef(row['coefficient'], row['p_value'])} & {tex_num(row['std_error'])} & {tex_num(row['t'], 2)} & {tex_p(row['p_value'])} & {used} \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}\\footnotesize",
            "\\item 注：原假设是 \(\\beta+\\gamma=0\)，即重点排污单位组的专长斜率为零。标准误与 \(t\) 来自公司聚类协方差上的线性组合，对应 Stata 的 \\texttt{lincom}。两行「完全交互」是推断所用规格。",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{center}",
            "",
        ]
    )

    current_ims = others[others["spec"].eq("current") & others["core"].eq("IMS")]
    lines.extend(
        [
            "\\begin{center}",
            "\\begin{threeparttable}",
            "\\caption{完全交互的其他项：控制变量与年份}",
            "\\label{tab:keypoll-other}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "项 & 系数 & 标准误 & \(p\) \\\\",
            "\\midrule",
            "\\multicolumn{4}{l}{\\textit{控制变量 $\\times V$}} \\\\",
        ]
    )
    for _, row in current_ims[current_ims["term_group"].eq("control_x_v")].iterrows():
        lines.append(f"{term_label(row['variable'])} & {tex_coef(row['coefficient'], row['p_value'])} & {tex_num(row['std_error'])} & {tex_p(row['p_value'])} \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{4}{l}{\\textit{年份 $\\times V$，参照 2021}} \\\\")
    for _, row in current_ims[current_ims["term_group"].eq("year_x_v")].iterrows():
        lines.append(f"{term_label(row['variable'])} & {tex_coef(row['coefficient'], row['p_value'])} & {tex_num(row['std_error'])} & {tex_p(row['p_value'])} \\\\")
    v_main = current_ims[current_ims["term_group"].eq("v_main")].iloc[0]
    lines.extend(
        [
            "\\midrule",
            "\\multicolumn{4}{l}{\\textit{调节变量主效应（参照水平上的截距差）}} \\\\",
            f"\\texttt{{KeyPollMonUnit}} & {tex_coef(v_main['coefficient'], v_main['p_value'])} & {tex_num(v_main['std_error'])} & {tex_p(v_main['p_value'])} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}\\footnotesize",
            "\\item 注：样本为当期 \\texttt{IMS} 与 \\texttt{KeyPollMonUnit}，$N=15{,}381$。\\texttt{KeyPollMonUnit} 主效应是控制变量和固定效应取参照水平时的截距差，不是重点单位的平均漂绿差距。行业交互见表~\\ref{tab:keypoll-industry}，联合检验见表~\\ref{tab:keypoll-wald}。",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{center}",
            "",
            "\\begin{center}",
            "\\begin{threeparttable}",
            "\\caption{其他交互项的联合检验}",
            "\\label{tab:keypoll-wald}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "原假设 & $\\chi^2$ & 自由度 & \(p\) \\\\",
            "\\midrule",
        ]
    )
    wald_labels = [
        ("controls_x_v", r"控制 $\times V=0$"),
        ("fe_x_v", r"年份与行业 $\times V=0$"),
        ("all_other_x_v", r"全部其他交互 $=0$"),
    ]
    for variable, label in wald_labels:
        row = pick_wald(others, "current", "IMS", variable)
        df_text = tex_int(row["df_num"])
        if pd.notna(row["notes"]) and "rank=" in str(row["notes"]):
            df_text += f"（约束 {tex_int(row['n_restrictions'])} 个）"
        lines.append(f"{label} & {tex_num(row['statistic'], 2)} & {df_text} & {tex_p(row['p_value'])} \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}\\footnotesize",
            "\\item 注：公司聚类下的 $\\chi^2$ Wald，不含 \(S\\times V\)。全部其他交互在 5\\% 水平拒绝，表~\\ref{tab:keypoll-slopes-advisor} 的推断用完全交互。自由度是协方差矩阵的秩。",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{center}",
            "",
            "\\begin{center}",
            "\\begin{threeparttable}",
            "\\caption{完全交互的行业虚拟变量 $\\times$ 重点排污单位}",
            "\\label{tab:keypoll-industry}",
            "\\footnotesize",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "项 & 系数 & 标准误 & \(p\) \\\\",
            "\\midrule",
        ]
    )
    industry_rows = current_ims[current_ims["term_group"].eq("industry_x_v")].reset_index(drop=True)
    midpoint = (len(industry_rows) + 1) // 2
    for i, row in industry_rows.iterrows():
        lines.append(f"{term_label(row['variable'])} & {tex_coef(row['coefficient'], row['p_value'])} & {tex_num(row['std_error'])} & {tex_p(row['p_value'])} \\\\")
        if i == midpoint - 1:
            lines.extend(
                [
                    "\\bottomrule",
                    "\\end{tabular}",
                    "\\end{threeparttable}",
                    "\\end{center}",
                    "",
                    "\\begin{center}",
                    "\\begin{threeparttable}",
                    "\\caption{完全交互的行业虚拟变量 $\\times$ 重点排污单位（续）}",
                    "\\footnotesize",
                    "\\begin{tabular}{lccc}",
                    "\\toprule",
                    "项 & 系数 & 标准误 & \(p\) \\\\",
                    "\\midrule",
                ]
            )
    n10 = int((industry_rows["p_value"] < 0.10).sum())
    n05 = int((industry_rows["p_value"] < 0.05).sum())
    n01 = int((industry_rows["p_value"] < 0.01).sum())
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}\\footnotesize",
            "\\item 注：参照行业为代码 A。系数是该行业重点单位相对参照行业的截距差，不是专长斜率。",
            f"共 {len(industry_rows)} 项，其中 $p<0.10$ 的有 {n10} 项，$p<0.05$ 的有 {n05} 项，$p<0.01$ 的有 {n01} 项。单项显著不单独决定用哪一个规格。",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{center}",
            "",
        ]
    )

    lines.extend(
        [
            "\\begin{center}",
            "\\begin{threeparttable}",
            "\\caption{附录：\\texttt{IMS\\_D}、滞后身份与剔除处罚当年}",
            "\\label{tab:keypoll-advisor-appendix}",
            "\\begin{tabular}{llcccc}",
            "\\toprule",
            "样本 & 核心 & 其他项联合 $p$ & 完全交互 \(\\beta\) & 完全交互 \(\\beta+\\gamma\) & 完全交互 \(\\gamma\) \\\\",
            "\\midrule",
        ]
    )
    appendix_jobs = [
        ("current", "IMS_D", "当期"),
        ("lag", "IMS", "滞后 KeyPoll"),
        ("lag", "IMS_D", "滞后 KeyPoll"),
        ("drop_penalty", "IMS", "剔除处罚当年"),
        ("drop_penalty", "IMS_D", "剔除处罚当年"),
    ]
    rank_notes = []
    for spec, core, label in appendix_jobs:
        wald = pick_wald(others, spec, core, "all_other_x_v")
        beta = pick_slope(slopes, spec, core, "full", "beta")
        total = pick_slope(slopes, spec, core, "full", "beta_plus_gamma")
        gamma = pick_slope(slopes, spec, core, "full", "gamma")
        core_label = r"\texttt{IMS}" if core == "IMS" else r"\texttt{IMS\_D}"
        lines.append(
            f"{label} & {core_label} & {tex_p(wald['p_value'])} & {tex_coef(beta['coefficient'], beta['p_value'])} & {tex_coef(total['coefficient'], total['p_value'])} & {tex_coef(gamma['coefficient'], gamma['p_value'])} \\\\"
        )
        if pd.notna(wald["notes"]) and "rank=" in str(wald["notes"]):
            rank_notes.append(f"{label}、{core_label} 的全部其他交互约束 {tex_int(wald['n_restrictions'])} 个，协方差秩为 {tex_int(wald['df_num'])}")
    note = "注：每一行的联合检验都在 5\\% 水平拒绝，推断用完全交互，表中系数即该规格。标准误按公司聚类。组斜率的标准误见 \\texttt{keypoll\\_advisor\\_slopes.csv}，逐项系数见 \\texttt{keypoll\\_full\\_other\\_terms.csv}。"
    if rank_notes:
        note += " " + "；".join(rank_notes) + "，$p$ 值按该秩计算。"
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}\\footnotesize",
            f"\\item {note}",
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{center}",
            "",
        ]
    )
    destination = TABLES.parents[1] / "tex" / "keypoll_advisor_tables.tex"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {destination}")


def main() -> None:
    data = load_analysis_panel()
    correlation = correlation_rows(data)
    correlation.to_csv(TABLES / "keypoll_ims_correlation.csv", index=False, encoding="utf-8-sig")

    lagged = data.dropna(subset=["L_KeyPollMonUnit"])
    no_penalty = data[data["Penalty_D"].fillna(0).eq(0)]
    jobs = [
        ("current", "IMS", "KeyPollMonUnit", data),
        ("current", "IMS_D", "KeyPollMonUnit", data),
        ("lag", "IMS", "L_KeyPollMonUnit", lagged),
        ("lag", "IMS_D", "L_KeyPollMonUnit", lagged),
        ("drop_penalty", "IMS", "KeyPollMonUnit", no_penalty),
        ("drop_penalty", "IMS_D", "KeyPollMonUnit", no_penalty),
    ]
    slopes: list[dict] = []
    lincoms: list[dict] = []
    others: list[dict] = []
    for spec, core, moderator, frame in jobs:
        slope_rows, lincom_rows, other_rows = evaluate_spec(frame, spec, core, moderator)
        slopes.extend(slope_rows)
        lincoms.extend(lincom_rows)
        others.extend(other_rows)

    pd.DataFrame(slopes).to_csv(TABLES / "keypoll_advisor_slopes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lincoms).to_csv(TABLES / "keypoll_advisor_lincom.csv", index=False, encoding="utf-8-sig")
    other_df = pd.DataFrame(others)
    other_df.to_csv(TABLES / "keypoll_full_other_terms.csv", index=False, encoding="utf-8-sig")
    write_advisor_tables(correlation, pd.DataFrame(slopes), other_df)
    print(f"wrote tables under {TABLES}")


if __name__ == "__main__":
    main()
