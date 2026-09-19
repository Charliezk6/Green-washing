from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

TEST_ROOT = Path(__file__).resolve().parents[2]
INPUT = TEST_ROOT / "data/input/combined_regression_panel_15381.csv"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
REGRESSION_OUTPUT = RESULTS_DIR / "lagged_ims_regression.csv"
DIAGNOSTICS_OUTPUT = RESULTS_DIR / "lagged_ims_sample_diagnostics.csv"

FIRM = "证券代码"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"
GW = "Greenwashing连续_基准"
IMS = "IMS"
CONTROLS = [
    "Size",
    "Lev",
    "ROA",
    "CFO",
    "Growth",
    "FirmAge",
    "SOE",
    "Dual",
    "BoardSize",
    "IndependentDirectorRatio",
    "Big4",
    "LnAuditFee",
    "AuditorTenure",
    "IsChangeFirm",
]
WINSORIZED_CONTROLS = [
    "Size",
    "Lev",
    "ROA",
    "CFO",
    "Growth",
    "FirmAge",
    "BoardSize",
    "IndependentDirectorRatio",
    "LnAuditFee",
    "AuditorTenure",
]


def significance(p_value: float) -> str:
    return "***" if p_value < 0.01 else "**" if p_value < 0.05 else "*" if p_value < 0.10 else ""


def load_common_baseline_sample() -> pd.DataFrame:
    """Reproduce the baseline regression's common sample and winsorization rule."""
    data = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig", low_memory=False)
    required = [FIRM, YEAR, INDUSTRY, GW, IMS, *CONTROLS]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"输入数据缺少字段: {missing}")

    for column in [YEAR, GW, IMS, *CONTROLS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    common = data.dropna(subset=required).copy()
    for column in WINSORIZED_CONTROLS:
        lower, upper = common[column].quantile([0.01, 0.99])
        common[column] = common[column].clip(lower=lower, upper=upper)

    return common.sort_values([FIRM, YEAR]).copy()


def construct_lagged_sample(common: pd.DataFrame) -> pd.DataFrame:
    data = common.copy()
    previous_year = data.groupby(FIRM)[YEAR].shift(1)
    consecutive_year = data[YEAR].sub(previous_year).eq(1)

    for column in CONTROLS:
        data[f"L_{column}"] = data.groupby(FIRM)[column].shift(1).where(consecutive_year)

    lagged_controls = [f"L_{column}" for column in CONTROLS]
    return data.dropna(subset=[GW, IMS, FIRM, YEAR, INDUSTRY, *lagged_controls]).copy()


def fit_model(sample: pd.DataFrame):
    lagged_controls = [f"L_{column}" for column in CONTROLS]
    terms = [f"Q('{IMS}')", *[f"Q('{column}')" for column in lagged_controls], f"C(Q('{YEAR}'))", f"C(Q('{INDUSTRY}'))"]
    formula = f"Q('{GW}') ~ " + " + ".join(terms)
    return smf.ols(formula, data=sample).fit(cov_type="cluster", cov_kwds={"groups": sample[FIRM]})


def tidy_results(result, sample: pd.DataFrame) -> pd.DataFrame:
    variables = ["Intercept", IMS, *[f"L_{column}" for column in CONTROLS]]
    confidence_intervals = result.conf_int()
    rows = []
    for variable in variables:
        parameter = "Intercept" if variable == "Intercept" else f"Q('{variable}')"
        if parameter not in result.params.index:
            continue
        p_value = float(result.pvalues[parameter])
        interval = confidence_intervals.loc[parameter]
        rows.append(
            {
                "规格": "C2_滞后完整控制",
                "变量": variable,
                "系数": result.params[parameter],
                "标准误": result.bse[parameter],
                "t值": result.tvalues[parameter],
                "p值": p_value,
                "显著性": significance(p_value),
                "置信区间下限": interval.iloc[0],
                "置信区间上限": interval.iloc[1],
                "样本数": int(result.nobs),
                "企业数": sample[FIRM].nunique(),
                "年度数": sample[YEAR].nunique(),
                "行业数": sample[INDUSTRY].nunique(),
                "调整R方": result.rsquared_adj,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    common = load_common_baseline_sample()
    sample = construct_lagged_sample(common)
    result = fit_model(sample)
    output = tidy_results(result, sample)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(REGRESSION_OUTPUT, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "基准共同样本数": len(common),
                "滞后回归样本数": len(sample),
                "企业数": sample[FIRM].nunique(),
                "年度数": sample[YEAR].nunique(),
                "行业数": sample[INDUSTRY].nunique(),
                "严格连续年度": "是",
                "缩尾规则": "与 Regression/scripts/reproduce_regression.py 一致；在基准共同样本的 1% 和 99% 分位缩尾",
                "控制变量": "、".join(CONTROLS),
            }
        ]
    ).to_csv(DIAGNOSTICS_OUTPUT, index=False, encoding="utf-8-sig")
    print(output.to_string(index=False))
    print(f"\n已写入: {REGRESSION_OUTPUT}\n已写入: {DIAGNOSTICS_OUTPUT}")


if __name__ == "__main__":
    main()
