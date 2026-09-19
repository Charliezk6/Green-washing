from __future__ import annotations

import json
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
FINANCIAL = ["Size", "Lev", "ROA", "CFO", "Growth", "FirmAgeLog"]
GOVERNANCE = ["SOE", "Dual", "BoardSize", "IndependentDirectorRatio"]
AUDIT = ["Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm"]
BASE = FINANCIAL + GOVERNANCE + AUDIT
LAG_SOURCE = [GW, "GI_z_行业年度", "ESGE_z_行业年度", *BASE, TREAT]


def significance(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def load_and_lag() -> pd.DataFrame:
    data = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig")
    required = [FIRM, YEAR, INDUSTRY, TREAT, *LAG_SOURCE]
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ValueError(f"输入缺少字段: {missing}")
    for col in [YEAR, TREAT, *LAG_SOURCE]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values([FIRM, YEAR]).copy()
    previous_year = data.groupby(FIRM)[YEAR].shift(1)
    consecutive = data[YEAR].sub(previous_year).eq(1)
    for col in LAG_SOURCE:
        data[f"L_{col}"] = data.groupby(FIRM)[col].shift(1).where(consecutive)
    return data


def tidy(result, model: str, estimator: str) -> list[dict]:
    rows = []
    intervals = result.conf_int()
    for variable in result.params.index:
        if variable.startswith("C("):
            continue
        p = float(result.pvalues[variable])
        rows.append({"规格": model, "估计方法": estimator, "变量": variable,
                     "系数": result.params[variable], "标准误": result.bse[variable],
                     "统计量": result.tvalues[variable], "p值": p, "显著性": significance(p),
                     "置信区间下限": intervals.loc[variable, 0], "置信区间上限": intervals.loc[variable, 1],
                     "优势比": np.exp(result.params[variable]) if estimator == "Logit" else np.nan,
                     "样本数": int(result.nobs)})
    return rows


def drop_perfect_prediction_industries(sample: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    industry_stats = sample.groupby(INDUSTRY, observed=True)[TREAT].agg(["size", "sum", "nunique"])
    dropped = industry_stats.loc[industry_stats["nunique"].eq(1)]
    kept = sample.loc[~sample[INDUSTRY].isin(dropped.index)].copy()
    details = {
        "Logit剔除无因变量变异行业数": int(len(dropped)),
        "Logit剔除观测数": int(dropped["size"].sum()),
        "Logit剔除处理组数": int(dropped["sum"].sum()),
        "Logit保留样本数": int(len(kept)),
        "Logit保留企业数": int(kept[FIRM].nunique()),
        "剔除行业": " | ".join(map(str, dropped.index)),
    }
    return kept, details


def main() -> None:
    for subdir in [OUTPUT / "tables", OUTPUT / "diagnostics"]:
        subdir.mkdir(parents=True, exist_ok=True)
    data = load_and_lag()
    lag_controls = [f"L_{x}" for x in BASE]
    specs = {
        "S1_滞后漂绿": [f"L_{GW}", *lag_controls],
        "S2_漂绿构成项": ["L_GI_z_行业年度", "L_ESGE_z_行业年度", *lag_controls],
        "S3_控制既有专家关系": [f"L_{GW}", *lag_controls, f"L_{TREAT}"],
        "S4_进入专家状态": [f"L_{GW}", *lag_controls],
    }
    logit_rows, lpm_rows, margin_rows, samples, diagnostics = [], [], [], [], []
    for name, regressors in specs.items():
        required = [TREAT, FIRM, YEAR, INDUSTRY, *regressors]
        sample = data.dropna(subset=required).copy()
        if name.startswith("S4"):
            sample = sample.loc[sample[f"L_{TREAT}"].eq(0)].copy()
        logit_sample, separation = drop_perfect_prediction_industries(sample)
        samples.append({"规格": name, "样本数": len(sample), "企业数": sample[FIRM].nunique(),
                        "年度数": sample[YEAR].nunique(), "行业数": sample[INDUSTRY].nunique(),
                        "处理组数": int(sample[TREAT].sum()), "处理比例": sample[TREAT].mean(),
                        "L_IMS_D最小值": sample[f"L_{TREAT}"].min(), "L_IMS_D最大值": sample[f"L_{TREAT}"].max(),
                        **separation})
        formula = f"{TREAT} ~ " + " + ".join(regressors + [f"C({YEAR})", f"C({INDUSTRY})"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = smf.logit(formula, logit_sample).fit(
                    disp=False, maxiter=500, cov_type="cluster",
                    cov_kwds={"groups": logit_sample[FIRM]})
                logit_rows.extend(tidy(result, name, "Logit"))
                margins = result.get_margeff(at="overall", method="dydx").summary_frame()
                for variable, row in margins.iterrows():
                    if variable.startswith("C("):
                        continue
                    p = float(row["Pr(>|z|)"])
                    margin_rows.append({"规格": name, "变量": variable, "平均边际效应": row["dy/dx"],
                                        "标准误": row["Std. Err."], "z值": row["z"], "p值": p,
                                        "显著性": significance(p),
                                        "置信区间下限": row["Conf. Int. Low"], "置信区间上限": row["Cont. Int. Hi."]
                                        if "Cont. Int. Hi." in row.index else row["Conf. Int. Hi."],
                                        "样本数": int(result.nobs)})
                diagnostics.append({"规格": name, "模型": "Logit", "成功": True,
                                    "收敛": bool(result.mle_retvals.get("converged", False)),
                                    "迭代次数": result.mle_retvals.get("iterations"),
                                    "警告或错误": " | ".join(str(w.message) for w in caught),
                                    "聚类数": logit_sample[FIRM].nunique(), **separation})
            except Exception as exc:
                diagnostics.append({"规格": name, "模型": "Logit", "成功": False, "收敛": False,
                                    "迭代次数": np.nan, "警告或错误": f"{type(exc).__name__}: {exc}",
                                    "聚类数": logit_sample[FIRM].nunique(), **separation})
        try:
            result = smf.ols(formula, sample).fit(cov_type="cluster", cov_kwds={"groups": sample[FIRM]})
            lpm_rows.extend(tidy(result, name, "LPM"))
            diagnostics.append({"规格": name, "模型": "LPM", "成功": True, "收敛": True,
                                "迭代次数": np.nan, "警告或错误": "", "聚类数": sample[FIRM].nunique()})
        except Exception as exc:
            diagnostics.append({"规格": name, "模型": "LPM", "成功": False, "收敛": False,
                                "迭代次数": np.nan, "警告或错误": f"{type(exc).__name__}: {exc}",
                                "聚类数": sample[FIRM].nunique()})
    pd.DataFrame(logit_rows).to_csv(OUTPUT / "tables/logit_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(margin_rows).to_csv(OUTPUT / "tables/logit_average_marginal_effects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lpm_rows).to_csv(OUTPUT / "tables/lpm_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(samples).to_csv(OUTPUT / "tables/sample_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "diagnostics/model_diagnostics.csv", index=False, encoding="utf-8-sig")
    lag_diag = {"输入观测数": int(len(data)), "企业数": int(data[FIRM].nunique()),
                "严格连续滞后观测数": int(data[f"L_{GW}"].notna().sum()),
                "起始年度": int(data[YEAR].min()), "结束年度": int(data[YEAR].max())}
    pd.DataFrame([lag_diag]).to_csv(OUTPUT / "diagnostics/lag_data_diagnostics.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "diagnostics/run_summary.json").write_text(json.dumps({"input": str(INPUT), **lag_diag}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(samples).to_string(index=False))


if __name__ == "__main__":
    main()
