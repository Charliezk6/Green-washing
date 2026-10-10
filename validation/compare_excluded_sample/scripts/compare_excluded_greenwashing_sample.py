"""Compare firm-years dropped because greenwashing cannot be constructed
with the final 15,381 regression sample.

Variable construction and the 1%/99% winsorization bounds follow
Regression/scripts/reproduce_regression.py. Bounds are calculated on the
retained sample and then applied to the excluded sample. IMS is not winsorized.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PACK = Path(__file__).resolve().parents[1]
PROJECT = PACK.parents[1]
sys.path.insert(0, str(PROJECT / "Regression" / "scripts"))
import reproduce_regression as regression

OUTPUT = PACK / "outputs"
DESCRIPTIVE = PROJECT / "Regression" / "outputs" / "paper_tables" / "descriptive_statistics_15381.csv"
EXPECTED_EXCLUDED = 1594
EXPECTED_RETAINED = 15381
MEAN_TOLERANCE = 1e-8
SMD_REFERENCE = 0.10
YEARS = (2021, 2022, 2023, 2024)

LABELS = {
    "IMS": "IMS",
    "IMS_D": "IMS_D",
    "Size": "Size",
    "Lev": "Lev",
    "ROA": "ROA",
    "CFO": "CFO",
    "Growth": "Growth",
    "FirmAge": "ListingAge",
    "SOE": "SOE",
    "Dual": "Dual",
    "BoardSize": "BoardSize",
    "IndependentDirectorRatio": "IndDirector",
    "Big4": "Big4",
    "LnAuditFee": "AuditFee",
    "AuditorTenure": "Tenure",
    "IsChangeFirm": "AuditorChange",
}
COMPARE_VARIABLES = [*regression.CORE, *regression.ALL_CONTROLS]


def format_p(p_value: float) -> str:
    if p_value < 0.0001:
        return "p < 0.0001"
    return f"p = {p_value:.4f}"


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def apply_retained_bounds(frame: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    result = frame.copy()
    for variable, (lower, upper) in bounds.items():
        result[variable] = result[variable].clip(lower=lower, upper=upper)
    return result


def retained_bounds(retained: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds = {}
    for variable in regression.WINSORIZED_CONTROLS:
        lower, upper = retained[variable].quantile([0.01, 0.99])
        bounds[variable] = (float(lower), float(upper))
    return bounds


def contrast(data: pd.DataFrame, outcome: str) -> dict[str, object]:
    sample = data.dropna(subset=[outcome, "证券代码", "dropped"]).copy()
    excluded = sample.loc[sample["dropped"].eq(1), outcome]
    retained = sample.loc[sample["dropped"].eq(0), outcome]
    mean_excluded = float(excluded.mean())
    mean_retained = float(retained.mean())
    sd_excluded = float(excluded.std(ddof=1))
    sd_retained = float(retained.std(ddof=1))
    difference = mean_excluded - mean_retained
    pooled = np.sqrt((sd_excluded**2 + sd_retained**2) / 2)
    if pooled == 0 or np.isnan(pooled):
        raise ValueError(f"{outcome} has no cross-group variation for a standardized difference.")
    fitted = smf.ols(f"Q('{outcome}') ~ dropped", data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["证券代码"]},
    )
    coefficient = float(fitted.params["dropped"])
    if abs(coefficient - difference) > 1e-8:
        raise ValueError(f"{outcome} clustered coefficient {coefficient} differs from the mean gap {difference}.")
    p_value = float(fitted.pvalues["dropped"])
    return {
        "排除组样本量": int(excluded.shape[0]),
        "保留组样本量": int(retained.shape[0]),
        "排除组均值": mean_excluded,
        "保留组均值": mean_retained,
        "排除组标准差": sd_excluded,
        "保留组标准差": sd_retained,
        "均值差": difference,
        "标准化均值差": float(difference / pooled),
        "聚类标准误": float(fitted.bse["dropped"]),
        "t值": float(fitted.tvalues["dropped"]),
        "p值": p_value,
        "显著性": stars(p_value),
    }


def check_retained_means(retained: pd.DataFrame) -> None:
    descriptive = pd.read_csv(DESCRIPTIVE, index_col=0)
    for variable, label in LABELS.items():
        actual = float(retained[variable].mean())
        expected = float(descriptive.loc[label, "mean"])
        if abs(actual - expected) > MEAN_TOLERANCE:
            raise ValueError(f"{label} retained mean {actual} does not match the published descriptive mean {expected}.")


def composition_frame(pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year in YEARS:
        column = f"year_{year}"
        pool[column] = pool["时间"].eq(year).astype(float)
        row = contrast(pool, column)
        row["项目"] = str(year)
        rows.append(row)
    labels = set(pool["样本类型"].dropna().unique())
    if pool["样本类型"].isna().any() or labels != {"制造业", "非制造业"}:
        raise ValueError("样本类型 must be 制造业 or 非制造业 for every firm-year in the comparison.")
    pool["manufacturing"] = pool["样本类型"].eq("制造业").astype(float)
    manufacturing = contrast(pool, "manufacturing")
    manufacturing["项目"] = "制造业"
    rows.append(manufacturing)
    columns = ["项目", "排除组样本量", "保留组样本量", "排除组均值", "保留组均值", "排除组标准差", "保留组标准差", "均值差", "标准化均值差", "聚类标准误", "t值", "p值", "显著性"]
    return pd.DataFrame(rows)[columns].rename(columns={"排除组均值": "排除组份额", "保留组均值": "保留组份额", "均值差": "份额差"})


def write_note(comparisons: pd.DataFrame, composition: pd.DataFrame, firm_counts: dict[str, int]) -> None:
    significant = comparisons.loc[comparisons["p值"].lt(0.10)].copy()
    lines = [
        "# 排除样本与最终样本的差异",
        "",
        f"排除组是连续漂绿缺失的 {EXPECTED_EXCLUDED:,} 个公司—年，保留组是最终回归样本 {EXPECTED_RETAINED:,} 个公司—年。",
        "连续控制变量使用在保留组上计算的 1% 与 99% 缩尾边界，同一边界应用到排除组。IMS 不缩尾。",
        "均值差等于排除组均值减去保留组均值。显著性来自该变量对排除组虚拟变量的回归，标准误按公司聚类。",
        f"排除组涉及 {firm_counts['excluded_firms']:,} 家公司，保留组涉及 {firm_counts['retained_firms']:,} 家公司，其中 {firm_counts['both_firms']:,} 家公司在不同年份分别出现在两组。",
        "",
        "## 公司特征",
        "",
    ]
    if significant.empty:
        lines.append("全部比较变量在 10% 水平上都不显著。")
    else:
        lines.append("在 10% 水平上显著的变量如下。标准化均值差的绝对值达到 0.10 时，记为达到参照。")
        lines.append("")
        for row in significant.to_dict("records"):
            direction = "更高" if row["均值差"] > 0 else "更低"
            reached = "达到" if abs(row["标准化均值差"]) >= SMD_REFERENCE else "未达到"
            coverage = ""
            if int(row["排除组样本量"]) < EXPECTED_EXCLUDED:
                coverage = f"该比较使用排除组中该变量非缺失的 {int(row['排除组样本量']):,} 条。"
            lines.append(
                f"- {row['变量']}：排除组均值 {row['排除组均值']:.4f}，保留组均值 {row['保留组均值']:.4f}，"
                f"排除组比保留组{direction} {abs(row['均值差']):.4f}（{row['显著性']}，{format_p(row['p值'])}）。"
                f"标准化均值差为 {row['标准化均值差']:.3f}，{reached} 0.10。{coverage}"
            )
    quiet = comparisons.loc[comparisons["p值"].ge(0.10), "变量"].tolist()
    if quiet:
        lines.extend(["", "在 10% 水平上不显著的变量：" + "、".join(quiet) + "。"])
    lines.extend(["", "## 年度与制造业构成", ""])
    for row in composition.to_dict("records"):
        direction = "更高" if row["份额差"] > 0 else "更低"
        mark = row["显著性"] if row["显著性"] else "不显著"
        lines.append(
            f"- {row['项目']}：排除组份额 {row['排除组份额']:.4f}，保留组份额 {row['保留组份额']:.4f}，"
            f"排除组比保留组{direction} {abs(row['份额差']):.4f}（{mark}，{format_p(row['p值'])}）。"
            f"标准化均值差为 {row['标准化均值差']:.3f}。"
        )
    lines.append("")
    (OUTPUT / "排除样本与最终样本差异说明.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = regression.load_data()
    data["时间"] = pd.to_numeric(data["时间"], errors="coerce")
    data["证券代码"] = data["证券代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    required = [regression.DEPENDENT, *COMPARE_VARIABLES, "证券代码", "时间", "ExpertiseIndustryCode"]
    retained = data.dropna(subset=required).copy()
    excluded = data.loc[data[regression.DEPENDENT].isna()].copy()
    if len(excluded) != EXPECTED_EXCLUDED or len(retained) != EXPECTED_RETAINED:
        raise ValueError(f"Expected {EXPECTED_EXCLUDED} excluded and {EXPECTED_RETAINED} retained observations, found {len(excluded)} and {len(retained)}.")
    excluded_keys = set(map(tuple, excluded[["证券代码", "时间"]].to_numpy()))
    retained_keys = set(map(tuple, retained[["证券代码", "时间"]].to_numpy()))
    if not excluded_keys.isdisjoint(retained_keys):
        raise ValueError("Excluded and retained firm-years overlap.")
    bounds = retained_bounds(retained)
    retained = apply_retained_bounds(retained, bounds)
    excluded = apply_retained_bounds(excluded, bounds)
    check_retained_means(retained)
    pool = pd.concat([retained.assign(dropped=0), excluded.assign(dropped=1)], ignore_index=True)
    rows = []
    for variable in COMPARE_VARIABLES:
        row = contrast(pool, variable)
        row["变量"] = LABELS[variable]
        rows.append(row)
    comparisons = pd.DataFrame(rows)
    comparisons = comparisons[["变量", "排除组样本量", "保留组样本量", "排除组均值", "保留组均值", "排除组标准差", "保留组标准差", "均值差", "标准化均值差", "聚类标准误", "t值", "p值", "显著性"]]
    composition = composition_frame(pool)
    excluded_firms = set(excluded["证券代码"])
    retained_firms = set(retained["证券代码"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(OUTPUT / "excluded_1594_vs_15381.csv", index=False, encoding="utf-8-sig")
    composition.to_csv(OUTPUT / "excluded_1594_vs_15381_composition.csv", index=False, encoding="utf-8-sig")
    write_note(comparisons, composition, {
        "excluded_firms": len(excluded_firms),
        "retained_firms": len(retained_firms),
        "both_firms": len(excluded_firms & retained_firms),
    })
    print(f"excluded={len(excluded)} retained={len(retained)} outputs={OUTPUT}")


if __name__ == "__main__":
    main()
