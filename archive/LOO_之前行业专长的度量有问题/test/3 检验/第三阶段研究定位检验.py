from pathlib import Path
import argparse
import math
import time
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "regression_audit_expertise" / "结果" / "第二轮判断" / "05_complete_diagnostic_sample.csv"
OUT = Path(__file__).resolve().parent
ID, YEAR, INDUSTRY = "证券代码", "时间", "证监会行业新"
AUDITOR, Y = "AccountingFirmName", "Greenwashing连续_基准_w"
CONTROLS = ["Size_w", "Lev_w", "ROA_w", "CFO_w", "Top1_w", "SOE", "Dual", "IndependentDirectorRatio_w", "Age_w", "Big4"]
ASSET_VARS = ["AssetShare_w", "AssetShare_LOO_w", "ClientShare_LOO_w"]


def winsorize(s, p):
    if p == 0:
        return s
    x = s.dropna()
    return s if x.empty else s.clip(x.quantile(p), x.quantile(1 - p))


def fit(df, terms, firm_fe=False):
    needed = [Y, ID, YEAR] + CONTROLS + [INDUSTRY]
    candidates = ASSET_VARS + ["RelativeSizeQuartile", "SalesShare", "SalesShare_LOO", "HasOtherIndustryClient", "PositiveLOOIntensity", "L1_AssetShare_LOO", "L1_ClientShare_LOO"]
    for term in terms:
        needed.extend(x for x in candidates if x in term)
    needed = list(dict.fromkeys(needed))
    s = df.dropna(subset=needed).copy()
    fixed = f'C(Q("{ID}")) + C(Q("{YEAR}"))' if firm_fe else f'C(Q("{INDUSTRY}")) + C(Q("{YEAR}"))'
    formula = f'Q("{Y}") ~ ' + " + ".join(terms + CONTROLS) + " + " + fixed
    result = smf.ols(formula, data=s).fit(cov_type="cluster", cov_kwds={"groups": s[ID]})
    return result, s, formula


def result_row(model, result, sample, term, extra=None):
    ci = result.conf_int().loc[term]
    out = {
        "模型": model, "核心项": term, "系数": result.params[term], "公司聚类标准误": result.bse[term],
        "p值": result.pvalues[term], "95%置信区间下限": ci.iloc[0], "95%置信区间上限": ci.iloc[1],
        "样本量": int(result.nobs), "公司数": sample[ID].nunique(), "调整R2": result.rsquared_adj,
    }
    if extra:
        out.update(extra)
    return out


def construct_measures(df):
    valid = df[AUDITOR].notna() & df[INDUSTRY].notna() & df[YEAR].notna()
    group_iy, group_aiy = [YEAR, INDUSTRY], [YEAR, INDUSTRY, AUDITOR]
    revenue = pd.to_numeric(df["OperatingRevenue"], errors="coerce").where(lambda x: x > 0)
    df["PositiveRevenue"] = revenue
    work = df.loc[valid & revenue.notna()].copy()
    work["IndustryRevenue"] = work.groupby(group_iy)["PositiveRevenue"].transform("sum")
    work["AuditorIndustryRevenue"] = work.groupby(group_aiy)["PositiveRevenue"].transform("sum")
    work["SalesShare"] = work["AuditorIndustryRevenue"] / work["IndustryRevenue"]
    denominator = work["IndustryRevenue"] - work["PositiveRevenue"]
    work["SalesShare_LOO"] = (work["AuditorIndustryRevenue"] - work["PositiveRevenue"]) / denominator
    work.loc[denominator <= 0, "SalesShare_LOO"] = np.nan
    df = df.merge(work[[ID, YEAR, "SalesShare", "SalesShare_LOO"]], on=[ID, YEAR], how="left", validate="one_to_one")
    df["HasOtherIndustryClient"] = np.where(df["AssetShare_LOO"].notna(), (df["AssetShare_LOO"] > 0).astype(float), np.nan)
    df["PositiveLOOIntensity"] = df["AssetShare_LOO"].where(df["AssetShare_LOO"] > 0, 0)
    df["L1_AssetShare_LOO"] = df.sort_values([ID, YEAR]).groupby(ID)["AssetShare_LOO"].shift()
    df["L1_ClientShare_LOO"] = df.sort_values([ID, YEAR]).groupby(ID)["ClientShare_LOO"].shift()
    return df


def corrected_interactions(df):
    rows = []
    for x in ASSET_VARS:
        result, sample, _ = fit(df, [x, "C(RelativeSizeQuartile)", f"{x}:C(RelativeSizeQuartile)"])
        group_effects = {}
        base = result.params[x]
        base_var = result.cov_params().loc[x, x]
        group_effects[1] = (base, np.sqrt(base_var), result.pvalues[x])
        for q in [2, 3, 4]:
            interaction = f"{x}:C(RelativeSizeQuartile)[T.{float(q)}]"
            if interaction not in result.params:
                interaction = f"{x}:C(RelativeSizeQuartile)[T.{q}]"
            effect = base + result.params[interaction]
            variance = base_var + result.cov_params().loc[interaction, interaction] + 2 * result.cov_params().loc[x, interaction]
            se = np.sqrt(max(variance, 0))
            z = effect / se
            p = float(math.erfc(abs(z) / np.sqrt(2)))
            group_effects[q] = (effect, se, p)
        for q, (effect, se, p) in group_effects.items():
            rows.append({"模型": x, "规模组": f"Q{q}", "边际效应": effect, "公司聚类标准误": se, "p值": p,
                         "95%置信区间下限": effect - 1.96 * se, "95%置信区间上限": effect + 1.96 * se,
                         "样本量": int(result.nobs), "公司数": sample[ID].nunique()})
        interaction_terms = [t for t in result.params.index if t.startswith(f"{x}:C(RelativeSizeQuartile)")]
        test = result.wald_test(" = 0, ".join(interaction_terms) + " = 0", scalar=True)
        rows.append({"模型": x, "规模组": "交互项联合检验", "p值": float(test.pvalue), "样本量": int(result.nobs), "公司数": sample[ID].nunique()})
    return pd.DataFrame(rows)


def q1_validation(df):
    q1 = df[df["RelativeSizeQuartile"] == 1].copy()
    rows = []
    specifications = [("Q1基准", q1, False)]
    for year in sorted(q1[YEAR].dropna().unique()):
        specifications.append((f"Q1年度{int(year)}", q1[q1[YEAR] == year], False))
    manufacturing = q1[INDUSTRY].astype(str).str.contains("制造业", na=False)
    specifications += [("Q1制造业", q1[manufacturing], False), ("Q1非制造业", q1[~manufacturing], False), ("Q1公司固定效应", q1, True)]
    for label, sample_df, firm_fe in specifications:
        for x in ASSET_VARS:
            try:
                result, sample, _ = fit(sample_df, [x], firm_fe=firm_fe)
                rows.append(result_row(label, result, sample, x, {"公司固定效应": "是" if firm_fe else "否"}))
            except Exception as exc:
                rows.append({"模型": label, "核心项": x, "错误": str(exc)})
    output = pd.DataFrame(rows)
    valid = output["p值"].notna()
    if valid.any():
        output.loc[valid, "Holm调整p值"] = multipletests(output.loc[valid, "p值"], method="holm")[1]
        output.loc[valid, "BH调整p值"] = multipletests(output.loc[valid, "p值"], method="fdr_bh")[1]
    return output


def distribution_and_influence(df):
    q1 = df[df["RelativeSizeQuartile"] == 1].copy()
    distribution = []
    for x in ASSET_VARS:
        s = q1[x].dropna()
        distribution.append({"指标": x, "观测数": len(s), "均值": s.mean(), "标准差": s.std(), "最小值": s.min(),
                             "P5": s.quantile(.05), "中位数": s.median(), "P95": s.quantile(.95), "最大值": s.max(),
                             "零值比例": (s == 0).mean(), "事务所数": q1.loc[s.index, AUDITOR].nunique(),
                             "行业年度组数": q1.loc[s.index, [YEAR, INDUSTRY]].drop_duplicates().shape[0]})
    influence_rows = []
    for x in ASSET_VARS:
        result, sample, formula = fit(q1, [x])
        ols = smf.ols(formula, data=sample).fit()
        influence = ols.get_influence()
        cooks = influence.cooks_distance[0]
        dfbeta_idx = list(ols.params.index).index(x)
        dfbetas = influence.dfbetas[:, dfbeta_idx]
        threshold = 4 / len(sample)
        influence_rows.append({"指标": x, "样本量": len(sample), "最大Cook距离": np.nanmax(cooks),
                               "Cook距离超过4/N观测数": int(np.sum(cooks > threshold)), "最大绝对DFBETA": np.nanmax(np.abs(dfbetas)),
                               "绝对DFBETA超过2/sqrt(N)观测数": int(np.sum(np.abs(dfbetas) > 2 / np.sqrt(len(sample))))})
    return pd.DataFrame(distribution), pd.DataFrame(influence_rows)


def alternatives(df):
    rows = []
    for x in ["SalesShare", "SalesShare_LOO", "HasOtherIndustryClient", "PositiveLOOIntensity", "L1_AssetShare_LOO", "L1_ClientShare_LOO"]:
        result, sample, _ = fit(df, [x])
        rows.append(result_row("替代度量", result, sample, x))
    for p in [0, .005, .01, .02]:
        temp = df.copy()
        for raw, target in [("AssetShare", "AssetShare_w"), ("AssetShare_LOO", "AssetShare_LOO_w"), ("ClientShare_LOO", "ClientShare_LOO_w")]:
            temp[target] = winsorize(temp[raw], p)
        for x in ASSET_VARS:
            result, sample, _ = fit(temp, [x])
            rows.append(result_row(f"缩尾比例{p}", result, sample, x))
    return pd.DataFrame(rows)


def bootstrap_differences(df, reps=500, seed=20260722, progress_every=25):
    needed = [Y, ID, YEAR, INDUSTRY, "AssetShare_w", "AssetShare_LOO_w"] + CONTROLS
    sample = df.dropna(subset=needed).copy()
    common_terms = " + ".join(CONTROLS) + f' + C(Q("{INDUSTRY}")) + C(Q("{YEAR}"))'
    raw_formula = f'Q("{Y}") ~ AssetShare_w + ' + common_terms
    loo_formula = f'Q("{Y}") ~ AssetShare_LOO_w + ' + common_terms
    y_raw, x_raw_frame = patsy.dmatrices(raw_formula, data=sample, return_type="dataframe")
    y_loo, x_loo_frame = patsy.dmatrices(loo_formula, data=sample, return_type="dataframe")
    if not y_raw.index.equals(y_loo.index):
        raise ValueError("原始与LOO模型没有使用完全相同的样本")
    y = y_raw.iloc[:, 0].to_numpy(dtype=float)
    x_raw = x_raw_frame.to_numpy(dtype=float)
    x_loo = x_loo_frame.to_numpy(dtype=float)
    raw_idx = x_raw_frame.columns.get_loc("AssetShare_w")
    loo_idx = x_loo_frame.columns.get_loc("AssetShare_LOO_w")
    firm_codes, firms = pd.factorize(sample[ID], sort=False)
    firm_rows = [np.flatnonzero(firm_codes == code) for code in range(len(firms))]

    full_raw_coef = np.linalg.lstsq(x_raw, y, rcond=None)[0]
    full_loo_coef = np.linalg.lstsq(x_loo, y, rcond=None)[0]
    observed_difference = full_raw_coef[raw_idx] - full_loo_coef[loo_idx]
    rng = np.random.default_rng(seed)
    estimates = []
    started = time.time()
    for b in range(reps):
        selected = rng.integers(0, len(firm_rows), size=len(firm_rows))
        indices = np.concatenate([firm_rows[code] for code in selected])
        try:
            raw_coef = np.linalg.lstsq(x_raw[indices], y[indices], rcond=None)[0]
            loo_coef = np.linalg.lstsq(x_loo[indices], y[indices], rcond=None)[0]
            estimates.append(raw_coef[raw_idx] - loo_coef[loo_idx])
        except np.linalg.LinAlgError:
            continue
        if progress_every and ((b + 1) % progress_every == 0 or b + 1 == reps):
            elapsed = time.time() - started
            print(f"Bootstrap {b + 1}/{reps}，有效{len(estimates)}次，用时{elapsed:.1f}秒", flush=True)

    values = np.asarray(estimates)
    centered = values - values.mean()
    p_value = min(1.0, 2 * np.mean(np.abs(centered) >= abs(observed_difference)))
    summary = pd.DataFrame([{
        "比较": "同一模型中原始AssetShare-LOO AssetShare", "重复次数": reps,
        "有效重复次数": len(values), "随机种子": seed, "全样本系数差": observed_difference,
        "Bootstrap平均系数差": values.mean(), "Bootstrap标准误": values.std(ddof=1),
        "百分位置信区间下限": np.quantile(values, .025), "百分位置信区间上限": np.quantile(values, .975),
        "正态近似p值": float(math.erfc(abs(observed_difference / values.std(ddof=1)) / np.sqrt(2))),
        "中心化Bootstrap双侧p值": p_value,
    }])
    draws = pd.DataFrame({"重复序号": np.arange(1, len(values) + 1), "系数差": values})
    return summary, draws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    df = pd.read_csv(INPUT, dtype={ID: str}, encoding="utf-8-sig")
    numeric = [YEAR, Y, "TotalAssets", "OperatingRevenue", "AssetShare", "AssetShare_LOO", "ClientShare_LOO", "RelativeSize"] + ASSET_VARS + CONTROLS
    for col in set(numeric):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = construct_measures(df)
    df["RelativeSizeQuartile"] = df.groupby([YEAR, INDUSTRY])["RelativeSize"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=False, duplicates="drop") + 1 if x.notna().sum() >= 4 else np.nan
    )

    if not args.bootstrap_only:
        corrected_interactions(df).to_csv(OUT / "修正规模交互与边际效应.csv", index=False, encoding="utf-8-sig")
        q1_validation(df).to_csv(OUT / "Q1小企业稳健性检验.csv", index=False, encoding="utf-8-sig")
        distribution, influence = distribution_and_influence(df)
        distribution.to_csv(OUT / "Q1指标分布诊断.csv", index=False, encoding="utf-8-sig")
        influence.to_csv(OUT / "Q1影响点与多重检验.csv", index=False, encoding="utf-8-sig")
        alternatives(df).to_csv(OUT / "销售收入_两部式_滞后及缩尾检验.csv", index=False, encoding="utf-8-sig")
    bootstrap_summary, bootstrap_draws = bootstrap_differences(df, reps=args.bootstrap_reps, seed=args.seed)
    bootstrap_summary.to_csv(OUT / "系数差异Bootstrap检验.csv", index=False, encoding="utf-8-sig")
    bootstrap_draws.to_csv(OUT / "系数差异Bootstrap抽样分布.csv", index=False, encoding="utf-8-sig")
    print("第三阶段实证检验完成：", OUT, flush=True)


if __name__ == "__main__":
    main()
