from pathlib import Path

import pandas as pd

PANEL = Path(__file__).with_name("greenwashing_panel_all_industries_2021_2024.csv")

data = pd.read_csv(PANEL, dtype={"证券代码": str})
groups = data.groupby(["时间", "证监会行业新"], dropna=False)


def zscore(values: pd.Series) -> pd.Series:
    standard_deviation = values.std(ddof=0)
    if values.notna().sum() <= 1 or pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(float("nan"), index=values.index)
    return (values - values.mean()) / standard_deviation


data["GI_z_行业年度"] = groups["GI"].transform(zscore)
data["ESGE_z_行业年度"] = groups["E得分"].transform(zscore)
data["Greenwashing连续_基准"] = data["GI_z_行业年度"] - data["ESGE_z_行业年度"]
data["连续变量构造状态"] = data["Greenwashing连续_基准"].notna().map(
    {True: "complete", False: "insufficient_industry_year_variation"}
)
data.to_csv(PANEL, index=False, encoding="utf-8-sig", float_format="%.10f")

complete = int(data["Greenwashing连续_基准"].notna().sum())
print(f"rows={len(data)} complete={complete} incomplete={len(data) - complete} groups={groups.ngroups}")
