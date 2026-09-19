from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.nonparametric.smoothers_lowess import lowess


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "greenwashing_auditor_expertise_panel.csv"
OUTPUT_DIR = BASE_DIR / "相关关系检验结果"
TABLE_DIR = OUTPUT_DIR / "表格"
FIGURE_DIR = OUTPUT_DIR / "图形"

CORE_VARIABLES = [
    "Greenwashing连续_基准",
    "Greenwashing",
    "IMS",
    "IMS_D",
    "IPS",
    "IPS_D",
]
EXPERTISE_CONTINUOUS = ["IMS", "IPS"]
EXPERTISE_DUMMIES = ["IMS_D", "IPS_D"]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def significance(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def correlation_outputs(data: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = data[CORE_VARIABLES].corr(method=method)
    rows = []
    for index, left in enumerate(CORE_VARIABLES):
        for right in CORE_VARIABLES[index + 1 :]:
            pair = data[[left, right]].dropna()
            if method == "pearson":
                coefficient, p_value = stats.pearsonr(pair[left], pair[right])
            else:
                coefficient, p_value = stats.spearmanr(pair[left], pair[right])
            rows.append(
                {
                    "变量1": left,
                    "变量2": right,
                    "相关系数": coefficient,
                    "p值": p_value,
                    "显著性": significance(p_value),
                    "有效样本数": len(pair),
                }
            )
    return matrix, pd.DataFrame(rows)


def group_difference_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    outcome = "Greenwashing连续_基准"
    for dummy in EXPERTISE_DUMMIES:
        groups = [data.loc[data[dummy].eq(value), outcome].dropna() for value in (0, 1)]
        t_stat, t_p = stats.ttest_ind(groups[1], groups[0], equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(groups[1], groups[0], alternative="two-sided")
        rows.append(
            {
                "分组变量": dummy,
                "非专长组N": len(groups[0]),
                "非专长组均值": groups[0].mean(),
                "非专长组中位数": groups[0].median(),
                "专长组N": len(groups[1]),
                "专长组均值": groups[1].mean(),
                "专长组中位数": groups[1].median(),
                "均值差_专长减非专长": groups[1].mean() - groups[0].mean(),
                "Welch_t": t_stat,
                "Welch_p值": t_p,
                "Welch显著性": significance(t_p),
                "MannWhitney_U": u_stat,
                "MannWhitney_p值": u_p,
                "MannWhitney显著性": significance(u_p),
            }
        )
    return pd.DataFrame(rows)


def contingency_tests(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    tables = []
    for dummy in EXPERTISE_DUMMIES:
        table = pd.crosstab(data[dummy], data["Greenwashing"])
        table = table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        chi2, chi_p, _, _ = stats.chi2_contingency(table)
        odds_ratio, fisher_p = stats.fisher_exact(table)
        phi = np.sqrt(chi2 / table.to_numpy().sum())
        rates = table[1] / table.sum(axis=1)
        summary_rows.append(
            {
                "行业专长变量": dummy,
                "非专长组总数": int(table.loc[0].sum()),
                "非专长组漂绿数": int(table.loc[0, 1]),
                "非专长组漂绿率": rates.loc[0],
                "专长组总数": int(table.loc[1].sum()),
                "专长组漂绿数": int(table.loc[1, 1]),
                "专长组漂绿率": rates.loc[1],
                "漂绿率差_专长减非专长": rates.loc[1] - rates.loc[0],
                "卡方值": chi2,
                "卡方p值": chi_p,
                "Phi系数": phi,
                "优势比OR": odds_ratio,
                "Fisher_p值": fisher_p,
            }
        )
        formatted = table.reset_index().rename(columns={dummy: "行业专长取值", 0: "非漂绿", 1: "漂绿"})
        formatted.insert(0, "行业专长变量", dummy)
        tables.append(formatted)
    return pd.DataFrame(summary_rows), pd.concat(tables, ignore_index=True)


def subgroup_correlations(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group, subset in data.groupby(group_column):
        for expertise in EXPERTISE_CONTINUOUS:
            pair = subset[["Greenwashing连续_基准", expertise]].dropna()
            pearson_r, pearson_p = stats.pearsonr(pair[expertise], pair["Greenwashing连续_基准"])
            spearman_r, spearman_p = stats.spearmanr(pair[expertise], pair["Greenwashing连续_基准"])
            rows.append(
                {
                    "分组变量": group_column,
                    "组别": group,
                    "行业专长指标": expertise,
                    "有效样本数": len(pair),
                    "Pearson相关系数": pearson_r,
                    "Pearson_p值": pearson_p,
                    "Spearman相关系数": spearman_r,
                    "Spearman_p值": spearman_p,
                }
            )
    return pd.DataFrame(rows)


def expertise_consistency(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    continuous = data[["IMS", "IPS"]].corr().rename_axis("变量").reset_index()
    table = pd.crosstab(data["IMS_D"], data["IPS_D"]).reindex(
        index=[0, 1], columns=[0, 1], fill_value=0
    )
    total = table.to_numpy().sum()
    categories = pd.DataFrame(
        {
            "类别": ["两者均为0", "仅IMS_D为1", "仅IPS_D为1", "两者均为1", "两指标一致"],
            "样本数": [
                table.loc[0, 0], table.loc[1, 0], table.loc[0, 1],
                table.loc[1, 1], table.loc[0, 0] + table.loc[1, 1],
            ],
        }
    )
    categories["比例"] = categories["样本数"] / total
    return continuous, categories


def partial_correlations(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    working = data.dropna(subset=["Greenwashing连续_基准", "时间", "ExpertiseIndustryCode"]).copy()
    for expertise in EXPERTISE_CONTINUOUS:
        subset = working.dropna(subset=[expertise]).copy()
        y_residual = ols(
            "Q('Greenwashing连续_基准') ~ C(Q('时间')) + C(Q('ExpertiseIndustryCode'))",
            data=subset,
        ).fit().resid
        x_residual = ols(
            f"Q('{expertise}') ~ C(Q('时间')) + C(Q('ExpertiseIndustryCode'))",
            data=subset,
        ).fit().resid
        coefficient, p_value = stats.pearsonr(x_residual, y_residual)
        rows.append(
            {
                "漂绿指标": "Greenwashing连续_基准",
                "行业专长指标": expertise,
                "控制变量": "年度固定效应、行业固定效应",
                "偏相关系数": coefficient,
                "p值": p_value,
                "显著性": significance(p_value),
                "有效样本数": len(subset),
            }
        )
    return pd.DataFrame(rows)


def plot_scatter_lowess(data: pd.DataFrame, x: str) -> None:
    subset = data[[x, "Greenwashing连续_基准"]].dropna()
    sampled = subset.sample(min(5000, len(subset)), random_state=2026)
    smooth = lowess(
        subset["Greenwashing连续_基准"], subset[x], frac=0.25, return_sorted=True
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sampled[x], sampled["Greenwashing连续_基准"], alpha=0.16, s=12, color="#3B82F6")
    sns.regplot(data=subset, x=x, y="Greenwashing连续_基准", scatter=False, ci=95,
                line_kws={"color": "#DC2626", "label": "线性拟合"}, ax=ax)
    ax.plot(smooth[:, 0], smooth[:, 1], color="#059669", linewidth=2.2, label="LOWESS")
    ax.set(title=f"{x}与连续漂绿指标的关系", xlabel=x, ylabel="连续漂绿指标")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{x}与连续漂绿_散点及LOWESS.png", dpi=300)
    plt.close(fig)


def plot_binned(data: pd.DataFrame, x: str) -> None:
    subset = data[[x, "Greenwashing连续_基准"]].dropna().copy()
    subset["分位组"] = pd.qcut(subset[x], q=20, duplicates="drop")
    binned = subset.groupby("分位组", observed=True).agg(
        x均值=(x, "mean"), 漂绿均值=("Greenwashing连续_基准", "mean"),
        漂绿标准误=("Greenwashing连续_基准", "sem"), 样本数=(x, "size")
    ).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(binned["x均值"], binned["漂绿均值"], yerr=1.96 * binned["漂绿标准误"],
                fmt="o-", color="#2563EB", ecolor="#93C5FD", capsize=3)
    ax.axhline(0, color="#6B7280", linewidth=1, linestyle="--")
    ax.set(title=f"{x}与连续漂绿指标的分箱关系", xlabel=f"{x}分位组内均值", ylabel="连续漂绿指标组内均值")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{x}与连续漂绿_分箱散点图.png", dpi=300)
    plt.close(fig)


def plot_box(data: pd.DataFrame, dummy: str) -> None:
    subset = data[[dummy, "Greenwashing连续_基准"]].dropna()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=subset, x=dummy, y="Greenwashing连续_基准", showfliers=False,
                palette=["#94A3B8", "#2563EB"], hue=dummy, legend=False, ax=ax)
    ax.set(title=f"按{dummy}分组的连续漂绿指标", xlabel=f"{dummy}（0=非专长，1=专长）", ylabel="连续漂绿指标")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"按{dummy}分组的连续漂绿箱线图.png", dpi=300)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("核心变量Pearson相关系数")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "核心变量Pearson相关热力图.png", dpi=300)
    plt.close(fig)


def plot_annual_correlations(annual: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for indicator, group in annual.groupby("行业专长指标"):
        ax.plot(group["组别"], group["Pearson相关系数"], marker="o", linewidth=2, label=indicator)
    ax.axhline(0, color="#6B7280", linewidth=1, linestyle="--")
    ax.set(title="分年度行业专长与连续漂绿的Pearson相关系数", xlabel="年度", ylabel="Pearson相关系数")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "分年度IMS_IPS与漂绿相关关系图.png", dpi=300)
    plt.close(fig)


def main() -> None:
    data = pd.read_csv(INPUT_FILE, dtype={"证券代码": str}, encoding="utf-8-sig")
    missing = sorted(set(CORE_VARIABLES + ["时间", "样本类型", "ExpertiseIndustryCode"]) - set(data.columns))
    if missing:
        raise ValueError(f"输入文件缺少字段：{', '.join(missing)}")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    pearson_matrix, pearson_detail = correlation_outputs(data, "pearson")
    spearman_matrix, spearman_detail = correlation_outputs(data, "spearman")
    group_tests = group_difference_tests(data)
    contingency_summary, contingency_tables = contingency_tests(data)
    annual = subgroup_correlations(data, "时间")
    sample_type = subgroup_correlations(data, "样本类型")
    continuous_consistency, dummy_consistency = expertise_consistency(data)
    partial = partial_correlations(data)

    outputs = {
        "01_Pearson相关矩阵.csv": pearson_matrix,
        "02_Pearson相关系数_p值_样本量明细.csv": pearson_detail,
        "03_Spearman相关矩阵.csv": spearman_matrix,
        "04_Spearman相关系数_p值_样本量明细.csv": spearman_detail,
        "05_行业专长分组漂绿均值与中位数检验.csv": group_tests,
        "06_漂绿与行业专长虚拟变量比例检验.csv": contingency_summary,
        "07_漂绿与行业专长虚拟变量列联表.csv": contingency_tables,
        "08_分年度核心相关关系.csv": annual,
        "09_制造业与非制造业分组相关关系.csv": sample_type,
        "10_IMS与IPS连续指标相关性.csv": continuous_consistency,
        "11_IMS_D与IPS_D一致性诊断.csv": dummy_consistency,
        "12_控制年度行业后的偏相关.csv": partial,
    }
    for filename, table in outputs.items():
        table.to_csv(TABLE_DIR / filename, index=True if "矩阵" in filename else False, encoding="utf-8-sig")

    for indicator in EXPERTISE_CONTINUOUS:
        plot_scatter_lowess(data, indicator)
        plot_binned(data, indicator)
    for dummy in EXPERTISE_DUMMIES:
        plot_box(data, dummy)
    plot_heatmap(pearson_matrix)
    plot_annual_correlations(annual)

    print(f"输入样本数：{len(data)}")
    print(f"已生成表格：{len(outputs)}份")
    print("已生成图形：8份")
    print(f"结果目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
