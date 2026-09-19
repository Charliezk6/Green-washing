from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
INPUT = PACKAGE_ROOT / "data/input/combined_regression_panel_14778.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, TREAT = "Greenwashing连续_基准", "IMS_D"
GI, ESGE = "GI_z_行业年度", "ESGE_z_行业年度"
CONTROLS = ["Size", "Lev", "ROA", "CFO", "Growth", "FirmAgeLog", "SOE", "Dual",
            "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee",
            "AuditorTenure", "IsChangeFirm"]
BASE_LAGS = [f"L_{x}" for x in CONTROLS]
W1_LAGS = ["L_Greenwashing连续_基准", *BASE_LAGS]
W2_LAGS = [f"L_{x}" for x in [GI, ESGE]] + BASE_LAGS
NO_GW_LAGS = BASE_LAGS
SPECS = {"W1": W1_LAGS, "W2": W2_LAGS, "W3_no_GW": NO_GW_LAGS}
MAIN_WEIGHT = "ATE_stabilized_trim_1_99"
OVERLAP_WEIGHT = "ATO_overlap"


def sig(p):
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def load_data():
    data = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig")
    required = [FIRM, YEAR, INDUSTRY, GW, TREAT, GI, ESGE, *CONTROLS]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    for col in [YEAR, GW, TREAT, GI, ESGE, *CONTROLS]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values([FIRM, YEAR]).copy()
    previous_year = data.groupby(FIRM)[YEAR].shift(1)
    consecutive = data[YEAR].sub(previous_year).eq(1)
    for col in [GW, GI, ESGE, *CONTROLS]:
        data[f"L_{col}"] = data.groupby(FIRM)[col].shift(1).where(consecutive)
    data["row_id"] = np.arange(len(data))
    return data


def formula_ps(treatment_lags):
    rhs = [*(f"Q('{x}')" for x in treatment_lags), f"C(Q('{YEAR}'))", f"C(Q('{INDUSTRY}'))"]
    return f"Q('{TREAT}') ~ " + " + ".join(rhs)


def fit_propensity(sample, lags):
    formula = formula_ps(lags)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = smf.logit(formula, sample).fit(disp=False, maxiter=500)
    converged = bool(result.mle_retvals.get("converged", False))
    if not converged:
        raise RuntimeError("Logit 倾向得分模型未收敛")
    score = pd.Series(result.predict(sample), index=sample.index)
    if not np.isfinite(score).all() or score.min() <= 0 or score.max() >= 1:
        raise RuntimeError("Logit 预测概率存在无效值或等于 0/1")
    return result, score, [str(x.message) for x in caught]


def trim_weight(series, lower, upper):
    lo, hi = series.quantile([lower, upper])
    return series.clip(lo, hi), float(lo), float(hi)


def make_weights(data):
    e = data["propensity_score"]
    p = data[TREAT].mean()
    ate = np.where(data[TREAT].eq(1), p / e, (1 - p) / (1 - e))
    att = np.where(data[TREAT].eq(1), 1.0, e / (1 - e))
    result = pd.DataFrame(index=data.index)
    result["ATE_stabilized_raw"] = ate
    result["ATE_stabilized_trim_1_99"], lo, hi = trim_weight(result["ATE_stabilized_raw"], .01, .99)
    result["ATE_stabilized_trim_05_995"], lo2, hi2 = trim_weight(result["ATE_stabilized_raw"], .005, .995)
    result["ATT_raw"] = att
    result["ATT_trim_1_99"], _, _ = trim_weight(result["ATT_raw"], .01, .99)
    result[OVERLAP_WEIGHT] = np.where(data[TREAT].eq(1), 1 - e, e)
    return result, {"ATE_1_99_lower": lo, "ATE_1_99_upper": hi,
                    "ATE_05_995_lower": lo2, "ATE_05_995_upper": hi2}


def ess(weights):
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum() ** 2 / np.square(weights).sum())


def weight_table(data, weights, spec, trim_points):
    rows = []
    for name in weights.columns:
        w = weights[name]
        rows.append({"规格": spec, "权重": name, "样本数": len(w), "均值": w.mean(),
                     "标准差": w.std(), "最小值": w.min(), "p01": w.quantile(.01),
                     "p05": w.quantile(.05), "中位数": w.median(), "p95": w.quantile(.95),
                     "p99": w.quantile(.99), "最大值": w.max(), "ESS": ess(w),
                     "处理组ESS": ess(w[data[TREAT].eq(1)]),
                     "对照组ESS": ess(w[data[TREAT].eq(0)]), **trim_points})
    return pd.DataFrame(rows)


def balance_table(data, weights, lags, spec):
    rows = []
    for weight_name, weight in [("unweighted", pd.Series(1.0, index=data.index)), *weights.items()]:
        for variable in lags:
            treated = data[TREAT].eq(1)
            control = ~treated
            x = data[variable]
            valid = x.notna() & weight.notna()
            treated = treated & valid
            control = control & valid
            def wm(mask):
                return float(np.average(x[mask], weights=weight[mask])) if mask.sum() else np.nan
            def wvar(mask, mean):
                return float(np.average((x[mask] - mean) ** 2, weights=weight[mask])) if mask.sum() else np.nan
            mt, mc = wm(treated), wm(control)
            vt, vc = wvar(treated, mt), wvar(control, mc)
            pooled = np.sqrt((vt + vc) / 2)
            smd = (mt - mc) / pooled if pooled > 0 else np.nan
            rows.append({"规格": spec, "权重": weight_name, "变量": variable,
                         "处理组均值": mt, "对照组均值": mc, "处理组方差": vt,
                         "对照组方差": vc, "方差比": vt / vc if vc > 0 else np.nan,
                         "标准化均值差": smd, "绝对SMD小于0.10": bool(abs(smd) < .10)})
    return pd.DataFrame(rows)


def result_models(data, weights, lags, spec):
    rows = []
    regressors = [TREAT]
    models = [("ATE截尾简化", MAIN_WEIGHT, regressors),
              ("ATT截尾简化", "ATT_trim_1_99", regressors),
              ("ATE未截尾简化", "ATE_stabilized_raw", regressors),
              ("ATE截尾0.5_99.5简化", "ATE_stabilized_trim_05_995", regressors),
              ("ATO重叠权重简化", OVERLAP_WEIGHT, regressors),
              ("ATO重叠权重增强", OVERLAP_WEIGHT, [TREAT, *lags]),
              ("ATE截尾增强", MAIN_WEIGHT, [TREAT, *lags])]
    for model_name, weight_name, rhs in models:
        sample = data.dropna(subset=[GW, FIRM, YEAR, INDUSTRY, *rhs]).copy()
        formula = f"Q('{GW}') ~ " + " + ".join([f"Q('{x}')" for x in rhs] +
                                                   [f"C(Q('{YEAR}'))", f"C(Q('{INDUSTRY}'))"])
        try:
            fit = smf.wls(formula, sample, weights=sample[weight_name]).fit(
                cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
            treatment_term = f"Q('{TREAT}')"
            p = float(fit.pvalues[treatment_term])
            rows.append({"规格": spec, "模型": model_name, "权重": weight_name,
                         "系数": fit.params[treatment_term], "标准误": fit.bse[treatment_term],
                         "t值": fit.tvalues[treatment_term], "p值": p, "显著性": sig(p),
                         "样本数": int(fit.nobs), "企业数": sample[FIRM].nunique(), "错误": ""})
        except Exception as exc:
            rows.append({"规格": spec, "模型": model_name, "权重": weight_name,
                         "系数": np.nan, "标准误": np.nan, "t值": np.nan, "p值": np.nan,
                         "显著性": "", "样本数": len(sample), "企业数": sample[FIRM].nunique(),
                         "错误": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def plots(data, balance, weights, spec):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for value, label in [(0, "Control"), (1, "Treated")]:
        ax.hist(data.loc[data[TREAT].eq(value), "propensity_score"], bins=30, alpha=.55,
                density=True, label=label)
    ax.set_xlabel("Propensity score"); ax.set_ylabel("Density"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPUT / "figures" / f"propensity_score_overlap_{spec}.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ["ATE_stabilized_raw", MAIN_WEIGHT, "ATT_trim_1_99", OVERLAP_WEIGHT]:
        ax.hist(weights[name], bins=40, alpha=.45, density=True, label=name)
    ax.set_xlabel("Weight"); ax.set_ylabel("Density"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUTPUT / "figures" / f"weight_distribution_{spec}.png", dpi=180); plt.close(fig)

    pivot = balance[balance["权重"].isin(["unweighted", MAIN_WEIGHT])].pivot(index="变量", columns="权重", values="标准化均值差")
    fig, ax = plt.subplots(figsize=(7, 6)); y = np.arange(len(pivot))
    for col, marker in [("unweighted", "o"), (MAIN_WEIGHT, "s")]:
        if col in pivot: ax.scatter(pivot[col].abs(), y, label=col, marker=marker)
    ax.axvline(.10, color="red", linestyle="--"); ax.set_yticks(y, pivot.index); ax.set_xlabel("Absolute SMD"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUTPUT / "figures" / f"love_plot_{spec}.png", dpi=180); plt.close(fig)


def run_spec(data, spec, lags):
    required = [FIRM, YEAR, INDUSTRY, GW, TREAT, *lags]
    initial = data.dropna(subset=required).copy()
    variation = initial.groupby([YEAR, INDUSTRY], observed=True)[TREAT].transform("nunique")
    fixed_effect_dropped = initial.loc[variation.eq(1)].copy()
    sample = initial.loc[variation.gt(1)].copy()
    if sample.empty or sample[TREAT].nunique() < 2:
        raise RuntimeError("年度—行业处理变量没有足够变异")
    result, score, warnings_seen = fit_propensity(sample, lags)
    sample["propensity_score"] = score
    treated_score = sample.loc[sample[TREAT].eq(1), "propensity_score"]
    control_score = sample.loc[sample[TREAT].eq(0), "propensity_score"]
    lower, upper = max(treated_score.min(), control_score.min()), min(treated_score.max(), control_score.max())
    support = sample[sample["propensity_score"].between(lower, upper)].copy()
    excluded = sample.loc[~sample.index.isin(support.index)].copy()
    weights, trim_points = make_weights(support)
    for col in weights: support[col] = weights[col]
    support.to_csv(OUTPUT / "weighted_data" / f"ipsw_analysis_data_{spec}.csv", index=False, encoding="utf-8-sig")
    balance = balance_table(support, weights, lags, spec)
    weight_diag = weight_table(support, weights, spec, trim_points)
    outcomes = result_models(support, weights, lags, spec)
    propensity_rows = [{"规格": spec, "变量": v, "系数": result.params[v], "标准误": result.bse[v],
                        "z值": result.tvalues[v], "p值": result.pvalues[v], "显著性": sig(result.pvalues[v]),
                        "样本数": int(result.nobs), "收敛": True} for v in result.params.index if not v.startswith("C(")]
    flow = {"规格": spec, "完整案例样本": len(initial), "处理组": int(initial[TREAT].sum()),
            "固定效应剔除观测": len(fixed_effect_dropped), "倾向模型样本": len(sample),
            "共同支撑样本": len(support), "共同支撑剔除处理组": int(excluded[TREAT].sum()),
            "共同支撑剔除对照组": int(excluded[TREAT].eq(0).sum()), "倾向得分下限": lower,
            "倾向得分上限": upper, "处理比例": sample[TREAT].mean(), "Logit警告": " | ".join(warnings_seen)}
    diagnostic = {"规格": spec, "类型": "propensity_model", "状态": "success", "信息": "Logit收敛"}
    return pd.DataFrame(propensity_rows), weight_diag, balance, outcomes, pd.DataFrame([flow]), pd.DataFrame([diagnostic]), pd.DataFrame(excluded)


def main():
    for directory in [OUTPUT / "tables", OUTPUT / "diagnostics", OUTPUT / "figures", OUTPUT / "weighted_data"]:
        directory.mkdir(parents=True, exist_ok=True)
    data = load_data()
    all_tables = {key: [] for key in ["propensity", "weights", "balance", "outcomes", "flow", "diagnostics", "excluded"]}
    for spec, lags in SPECS.items():
        try:
            outputs = run_spec(data, spec, lags)
            for key, value in zip(all_tables, outputs): all_tables[key].append(value)
        except Exception as exc:
            all_tables["diagnostics"].append(pd.DataFrame([{"规格": spec, "类型": "run", "状态": "failed", "信息": f"{type(exc).__name__}: {exc}"}]))
    paths = {"propensity": OUTPUT / "tables/propensity_models.csv", "weights": OUTPUT / "tables/weight_diagnostics.csv",
             "balance": OUTPUT / "tables/covariate_balance.csv", "outcomes": OUTPUT / "tables/outcome_regressions.csv",
             "flow": OUTPUT / "tables/sample_flow.csv", "diagnostics": OUTPUT / "diagnostics/model_diagnostics.csv",
             "excluded": OUTPUT / "diagnostics/common_support_exclusions.csv"}
    for key, path in paths.items():
        frames = all_tables[key]
        pd.concat(frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig") if frames else pd.DataFrame().to_csv(path, index=False, encoding="utf-8-sig")
    balance = pd.concat(all_tables["balance"], ignore_index=True) if all_tables["balance"] else pd.DataFrame()
    failures = balance[(~balance["绝对SMD小于0.10"]) & (balance["权重"] != "unweighted")] if not balance.empty else pd.DataFrame()
    failures.to_csv(OUTPUT / "diagnostics/balance_failures.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{"规格": s, "状态": "success" if any(x.get("规格").eq(s).any() and x.get("状态").eq("success").any() for x in all_tables["diagnostics"]) else "failed"} for s in SPECS])
    summary.to_csv(OUTPUT / "tables/run_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
