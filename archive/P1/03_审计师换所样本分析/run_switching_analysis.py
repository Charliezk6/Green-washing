from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT = PROJECT_ROOT / "Regression_reproduction/outputs/baseline/已构造基准回归变量的面板.csv"
OUTPUT = Path(__file__).resolve().parent / "outputs"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"
GW, TREAT = "Greenwashing连续_基准", "IMS_D"
PRE_VARS = [GW, "GI_z_行业年度", "ESGE_z_行业年度", "Size", "Lev", "ROA", "CFO", "Growth",
            "FirmAgeLog", "SOE", "Dual", "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee",
            "AuditorTenure", TREAT]
CORE = ["L_Size", "L_Lev", "L_ROA", "L_SOE", "L_Big4", "L_AuditorTenure", "L_IMS_D"]
EXTENDED = ["L_Dual", "L_BoardSize", "L_IndependentDirectorRatio"]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str, "InstitutionID": str}, encoding="utf-8-sig")
    required = [FIRM, YEAR, INDUSTRY, "InstitutionID", "IsChangeFirm", TREAT, *PRE_VARS]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    numeric = [YEAR, "IsChangeFirm", TREAT, *PRE_VARS]
    for col in set(numeric):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values([FIRM, YEAR]).copy()
    previous_year = data.groupby(FIRM)[YEAR].shift(1)
    consecutive = data[YEAR].sub(previous_year).eq(1)
    for col in [*PRE_VARS, "InstitutionID"]:
        data[f"L_{col}"] = data.groupby(FIRM)[col].shift(1).where(consecutive)
    previous_gw = data.groupby(FIRM)[GW].shift(1)
    previous2_gw = data.groupby(FIRM)[GW].shift(2)
    previous2_year = data.groupby(FIRM)[YEAR].shift(2)
    strict_two_year = consecutive & data[YEAR].sub(previous2_year).eq(2)
    data["L2_GW"] = previous2_gw.where(strict_two_year)
    data["PreDeltaGW"] = (previous_gw - previous2_gw).where(strict_two_year)
    data["DeltaGW"] = (data[GW] - previous_gw).where(consecutive)
    data["FlagChange"] = data["IsChangeFirm"].eq(1)
    data["CodeChange"] = consecutive & data["InstitutionID"].notna() & data["L_InstitutionID"].notna() & data["InstitutionID"].ne(data["L_InstitutionID"])
    data["VerifiedSwitch"] = data["FlagChange"] & data["CodeChange"]
    data["SwitchInExpert"] = (data["VerifiedSwitch"] & data[TREAT].eq(1)).astype(int)
    data["UpgradeExpert"] = (data["VerifiedSwitch"] & data["L_IMS_D"].eq(0) & data[TREAT].eq(1)).astype(int)
    return data


def tidy(result, specification: str, estimator: str) -> list[dict]:
    rows = []
    for variable in result.params.index:
        if variable.startswith("C("):
            continue
        rows.append({"规格": specification, "估计方法": estimator, "变量": variable,
                     "系数": result.params[variable], "标准误": result.bse[variable],
                     "统计量": result.tvalues[variable], "p值": result.pvalues[variable],
                     "样本数": int(result.nobs)})
    return rows


def compare_groups(data: pd.DataFrame, variables: list[str], group: str, comparison: str) -> list[dict]:
    rows = []
    for variable in variables:
        usable = data.dropna(subset=[variable, group])
        one, zero = usable.loc[usable[group].eq(1), variable], usable.loc[usable[group].eq(0), variable]
        pooled = np.sqrt((one.var(ddof=1) + zero.var(ddof=1)) / 2)
        difference = one.mean() - zero.mean()
        rows.append({"比较": comparison, "变量": variable, "换入专家样本数": len(one), "换入非专家样本数": len(zero),
                     "换入专家均值": one.mean(), "换入非专家均值": zero.mean(), "均值差": difference,
                     "标准化均值差": difference / pooled if pooled > 0 else np.nan})
    return rows


def main() -> None:
    for subdir in [OUTPUT / "tables", OUTPUT / "diagnostics"]:
        subdir.mkdir(parents=True, exist_ok=True)
    data = load_data()
    mismatch = data.loc[data["FlagChange"].ne(data["CodeChange"]),
                        [FIRM, YEAR, "InstitutionID", "L_InstitutionID", "IsChangeFirm", "FlagChange", "CodeChange"]]
    mismatch.to_csv(OUTPUT / "diagnostics/switch_flag_inconsistencies.csv", index=False, encoding="utf-8-sig")
    switches = data.loc[data["VerifiedSwitch"]].copy()
    specifications = {"W1_精简控制": ["L_Greenwashing连续_基准", *CORE],
                      "W2_加入治理扩展": ["L_Greenwashing连续_基准", *CORE, *EXTENDED]}
    coefficients, diagnostics, samples = [], [], []
    for name, regressors in specifications.items():
        required = ["SwitchInExpert", FIRM, YEAR, INDUSTRY, *regressors]
        sample = switches.dropna(subset=required).copy()
        events = int(sample["SwitchInExpert"].sum())
        samples.append({"规格": name, "样本数": len(sample), "企业数": sample[FIRM].nunique(),
                        "换入专家事件数": events, "换入非专家数": len(sample)-events,
                        "年度数": sample[YEAR].nunique(), "行业数": sample[INDUSTRY].nunique()})
        formula = "SwitchInExpert ~ " + " + ".join(regressors + [f"C({YEAR})", f"C({INDUSTRY})"])
        industry_variation = sample.groupby(INDUSTRY, observed=True)["SwitchInExpert"].transform("nunique")
        logit_sample = sample.loc[industry_variation.gt(1)].copy()
        dropped_industries = " | ".join(map(str, sorted(sample.loc[industry_variation.eq(1), INDUSTRY].unique())))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = smf.logit(formula, logit_sample).fit(disp=False, maxiter=500, cov_type="cluster",
                                                              cov_kwds={"groups": logit_sample[FIRM]})
                coefficients.extend(tidy(result, name, "Logit"))
                diagnostics.append({"规格": name, "模型": "Logit", "成功": True,
                                    "收敛": bool(result.mle_retvals.get("converged", False)),
                                    "事件数": int(logit_sample["SwitchInExpert"].sum()),
                                    "聚类数": logit_sample[FIRM].nunique(),
                                    "Logit剔除无因变量变异行业数": sample.loc[industry_variation.eq(1), INDUSTRY].nunique(),
                                    "Logit剔除观测数": int(industry_variation.eq(1).sum()),
                                    "Logit剔除事件数": int(sample.loc[industry_variation.eq(1), "SwitchInExpert"].sum()),
                                    "剔除行业": dropped_industries,
                                    "警告或错误": " | ".join(str(w.message) for w in caught)})
            except Exception as exc:
                diagnostics.append({"规格": name, "模型": "Logit", "成功": False, "收敛": False,
                                    "事件数": int(logit_sample["SwitchInExpert"].sum()),
                                    "聚类数": logit_sample[FIRM].nunique(),
                                    "Logit剔除无因变量变异行业数": sample.loc[industry_variation.eq(1), INDUSTRY].nunique(),
                                    "Logit剔除观测数": int(industry_variation.eq(1).sum()),
                                    "Logit剔除事件数": int(sample.loc[industry_variation.eq(1), "SwitchInExpert"].sum()),
                                    "剔除行业": dropped_industries,
                                    "警告或错误": f"{type(exc).__name__}: {exc}"})
        try:
            result = smf.ols(formula, sample).fit(cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
            coefficients.extend(tidy(result, name, "LPM"))
            diagnostics.append({"规格": name, "模型": "LPM", "成功": True, "收敛": True,
                                "事件数": events, "聚类数": sample[FIRM].nunique(), "警告或错误": ""})
        except Exception as exc:
            diagnostics.append({"规格": name, "模型": "LPM", "成功": False, "收敛": False,
                                "事件数": events, "聚类数": sample[FIRM].nunique(),
                                "警告或错误": f"{type(exc).__name__}: {exc}"})
    pd.DataFrame(coefficients).to_csv(OUTPUT / "tables/switching_model_coefficients.csv", index=False, encoding="utf-8-sig")
    comparison_vars = [f"L_{x}" for x in PRE_VARS if x != TREAT]
    pd.DataFrame(compare_groups(switches, comparison_vars, "SwitchInExpert", "换入专家与换入非专家事前差异")).to_csv(
        OUTPUT / "tables/pre_switch_comparisons.csv", index=False, encoding="utf-8-sig")
    pretrend = switches.dropna(subset=["PreDeltaGW"]).copy()
    pd.DataFrame(compare_groups(pretrend, ["PreDeltaGW"], "SwitchInExpert", "t-2至t-1漂绿变化")).to_csv(
        OUTPUT / "tables/pretrend_change_comparison.csv", index=False, encoding="utf-8-sig")
    delta_rows = []
    regressors = ["SwitchInExpert", *CORE]
    delta_sample = switches.dropna(subset=["DeltaGW", FIRM, YEAR, INDUSTRY, *regressors]).copy()
    formula = "DeltaGW ~ " + " + ".join(regressors + [f"C({YEAR})", f"C({INDUSTRY})"])
    try:
        result = smf.ols(formula, delta_sample).fit(cov_type="cluster", cov_kwds={"groups": delta_sample[FIRM]})
        delta_rows.extend(tidy(result, "换所后漂绿变化", "OLS公司聚类"))
    except Exception as exc:
        delta_rows.append({"规格": "换所后漂绿变化", "估计方法": "OLS公司聚类", "变量": "SwitchInExpert",
                           "系数": np.nan, "标准误": np.nan, "统计量": np.nan, "p值": np.nan,
                           "样本数": len(delta_sample), "错误": f"{type(exc).__name__}: {exc}"})
    pd.DataFrame(delta_rows).to_csv(OUTPUT / "tables/delta_gw_regression.csv", index=False, encoding="utf-8-sig")
    summary = [{"口径": "输入面板", "观测数": len(data), "企业数": data[FIRM].nunique(), "换入专家数": np.nan},
               {"口径": "IsChangeFirm=1", "观测数": int(data["FlagChange"].sum()), "企业数": data.loc[data["FlagChange"], FIRM].nunique(), "换入专家数": np.nan},
               {"口径": "事务所代码变化", "观测数": int(data["CodeChange"].sum()), "企业数": data.loc[data["CodeChange"], FIRM].nunique(), "换入专家数": np.nan},
               {"口径": "核验后主换所样本", "观测数": len(switches), "企业数": switches[FIRM].nunique(), "换入专家数": int(switches["SwitchInExpert"].sum())},
               {"口径": "UpgradeExpert", "观测数": int(switches["UpgradeExpert"].sum()), "企业数": switches.loc[switches["UpgradeExpert"].eq(1), FIRM].nunique(), "换入专家数": int(switches["UpgradeExpert"].sum())}]
    pd.concat([pd.DataFrame(summary), pd.DataFrame(samples)], ignore_index=True).to_csv(
        OUTPUT / "tables/sample_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "diagnostics/model_diagnostics.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
