from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "greenwashing_auditor_regression_panel_2021_2024.csv"
OUT = ROOT / "结果" / "第二轮判断"
OUT.mkdir(parents=True, exist_ok=True)
ID, YEAR, INDUSTRY = "证券代码", "时间", "证监会行业新"
Y = "Greenwashing连续_基准"
AUDITOR = "AccountingFirmName"


def winsorize(s, lo=0.01, hi=0.99):
    x = s.dropna()
    return s if x.empty else s.clip(x.quantile(lo), x.quantile(hi))


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def estimate(df, y, xvars, name, firm_fe=False):
    needed = [y, ID, YEAR] + xvars + ([] if firm_fe else [INDUSTRY])
    s = df.dropna(subset=needed).copy()
    fixed = f'C(Q("{ID}")) + C(Q("{YEAR}"))' if firm_fe else f'C(Q("{INDUSTRY}")) + C(Q("{YEAR}"))'
    formula = f'Q("{y}") ~ ' + " + ".join(xvars) + " + " + fixed
    r = smf.ols(formula, data=s).fit(cov_type="cluster", cov_kwds={"groups": s[ID]})
    return name, r, s, formula, firm_fe


def tidy(models, focus):
    rows = []
    for name, r, s, formula, firm_fe in models:
        for term in focus:
            if term not in r.params:
                continue
            rows.append({
                "模型": name, "核心变量": term, "系数": r.params[term],
                "公司聚类标准误": r.bse[term], "t值": r.tvalues[term],
                "p值": r.pvalues[term], "显著性": stars(r.pvalues[term]),
                "样本量": int(r.nobs), "公司数": s[ID].nunique(),
                "调整R2": r.rsquared_adj, "公司固定效应": "是" if firm_fe else "否",
                "行业固定效应": "否" if firm_fe else "是", "年度固定效应": "是",
                "公式": formula,
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(INPUT, dtype={ID: str}, encoding="utf-8-sig")
    df[ID] = df[ID].str.extract(r"(\d+)", expand=False).str.zfill(6)
    num = [YEAR, Y, "TotalAssets", "TotalLiabilities", "NetProfit", "OperatingNetCashFlow",
           "S0304a", "SOE", "Dual", "IndependentDirectorRatio", "Big4", "FirmAge"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    valid = df[AUDITOR].notna() & df[INDUSTRY].notna() & df[YEAR].notna() & (df["TotalAssets"] > 0)
    b = df.loc[valid].copy()
    group_iy = [YEAR, INDUSTRY]
    group_aiy = [YEAR, INDUSTRY, AUDITOR]
    b["IndustryAssets"] = b.groupby(group_iy)["TotalAssets"].transform("sum")
    b["AuditorIndustryAssets"] = b.groupby(group_aiy)["TotalAssets"].transform("sum")
    b["IndustryClients"] = b.groupby(group_iy)[ID].transform("count")
    b["AuditorIndustryClients"] = b.groupby(group_aiy)[ID].transform("count")
    b["RelativeSize"] = b["TotalAssets"] / b["IndustryAssets"]
    b["AssetShare"] = b["AuditorIndustryAssets"] / b["IndustryAssets"]
    b["AssetShare_LOO"] = (b["AuditorIndustryAssets"] - b["TotalAssets"]) / (b["IndustryAssets"] - b["TotalAssets"])
    b["ClientShare_LOO"] = (b["AuditorIndustryClients"] - 1) / (b["IndustryClients"] - 1)
    b["Gap"] = b["AssetShare"] - b["AssetShare_LOO"]
    b.loc[b["IndustryClients"] <= 1, "ClientShare_LOO"] = np.nan
    b.loc[b["IndustryAssets"] <= b["TotalAssets"], "AssetShare_LOO"] = np.nan
    df = df.merge(b[[ID, YEAR, "AssetShare", "AssetShare_LOO", "ClientShare_LOO", "IndustryClients", "RelativeSize", "Gap"]],
                  on=[ID, YEAR], how="left", validate="one_to_one")

    df["Size"] = np.log(df["TotalAssets"].where(df["TotalAssets"] > 0))
    df["Lev"] = df["TotalLiabilities"] / df["TotalAssets"]
    df["ROA"] = df["NetProfit"] / df["TotalAssets"]
    df["CFO"] = df["OperatingNetCashFlow"] / df["TotalAssets"]
    df["Top1"] = df["S0304a"] / 100
    df["Age"] = np.log1p(df["FirmAge"].where(df["FirmAge"] >= 0))
    continuous = [Y, "AssetShare", "AssetShare_LOO", "ClientShare_LOO", "Size", "Lev", "ROA", "CFO", "Top1", "IndependentDirectorRatio", "Age"]
    for c in continuous:
        df[c + "_w"] = winsorize(df[c])

    financial = ["Size_w", "Lev_w", "ROA_w", "CFO_w"]
    governance = ["Top1_w", "SOE", "Dual", "IndependentDirectorRatio_w", "Age_w", "Big4"]
    controls = financial + governance
    complete_vars = [Y + "_w", "AssetShare_w", "AssetShare_LOO_w", "ClientShare_LOO_w"] + controls
    complete = df.dropna(subset=complete_vars).copy()

    models = []
    # Same-sample sequence separates control-variable effects from sample changes.
    models += [
        estimate(complete, Y + "_w", ["AssetShare_w"], "S1 原始资产份额：固定样本无控制"),
        estimate(complete, Y + "_w", ["AssetShare_w"] + financial, "S2 原始资产份额：加入财务控制"),
        estimate(complete, Y + "_w", ["AssetShare_w"] + controls, "S3 原始资产份额：完整控制"),
        estimate(complete, Y + "_w", ["AssetShare_LOO_w"], "L1 LOO资产份额：固定样本无控制"),
        estimate(complete, Y + "_w", ["AssetShare_LOO_w"] + financial, "L2 LOO资产份额：加入财务控制"),
        estimate(complete, Y + "_w", ["AssetShare_LOO_w"] + controls, "L3 LOO资产份额：完整控制"),
        estimate(complete, Y + "_w", ["ClientShare_LOO_w"] + controls, "C1 LOO客户数量份额：完整控制"),
        estimate(complete, Y + "_w", ["AssetShare_LOO_w"] + controls, "F1 LOO资产份额：公司固定效应", firm_fe=True),
        estimate(complete, Y + "_w", ["ClientShare_LOO_w"] + controls, "F2 LOO客户份额：公司固定效应", firm_fe=True),
    ]
    focus = ["AssetShare_w", "AssetShare_LOO_w", "ClientShare_LOO_w"]
    results = tidy(models, focus)
    results.to_csv(OUT / "01_key_diagnostic_regressions.csv", index=False, encoding="utf-8-sig")

    full = []
    for name, r, s, formula, _ in models:
        full.append(f"\n{'='*100}\n{name}\n{formula}\n{'='*100}\n{r.summary().as_text()}\n")
    (OUT / "02_full_diagnostic_output.txt").write_text("".join(full), encoding="utf-8")

    variation = []
    for c in ["AssetShare", "AssetShare_LOO", "ClientShare_LOO"]:
        x = complete[[ID, c]].dropna()
        means = x.groupby(ID)[c].transform("mean")
        within = x[c] - means
        nunique = x.groupby(ID)[c].nunique()
        variation.append({
            "指标": c, "总体标准差": x[c].std(), "公司内标准差": within.std(),
            "有至少两个不同取值的公司数": int((nunique > 1).sum()),
            "可用公司数": int(nunique.size), "可用观测数": len(x),
        })
    pd.DataFrame(variation).to_csv(OUT / "03_within_firm_variation.csv", index=False, encoding="utf-8-sig")

    switching = complete.sort_values([ID, YEAR]).copy()
    switching["AuditorChanged"] = switching.groupby(ID)[AUDITOR].transform(lambda x: x.ne(x.shift()) & x.shift().notna())
    diagnostic = pd.DataFrame([{
        "完整样本观测数": len(complete), "完整样本公司数": complete[ID].nunique(),
        "发生过事务所更换的公司数": switching.loc[switching["AuditorChanged"], ID].nunique(),
        "事务所更换事件数": int(switching["AuditorChanged"].sum()),
        "行业年度最小公司数": int(b["IndustryClients"].min()),
        "行业年度中位公司数": float(b["IndustryClients"].median()),
        "行业年度不足10家公司观测数": int((b["IndustryClients"] < 10).sum()),
        "原始份额与LOO资产份额相关系数": complete[["AssetShare", "AssetShare_LOO"]].corr().iloc[0, 1],
        "LOO资产与LOO客户份额相关系数": complete[["AssetShare_LOO", "ClientShare_LOO"]].corr().iloc[0, 1],
    }])
    diagnostic.to_csv(OUT / "04_sample_and_switching_diagnostics.csv", index=False, encoding="utf-8-sig")
    complete.to_csv(OUT / "05_complete_diagnostic_sample.csv", index=False, encoding="utf-8-sig")

    mechanical = complete.dropna(subset=["Gap", "RelativeSize", ID, YEAR, INDUSTRY]).copy()
    gap_model = smf.ols(
        f'Gap ~ RelativeSize + C(Q("{INDUSTRY}")) + C(Q("{YEAR}"))', data=mechanical
    ).fit(cov_type="cluster", cov_kwds={"groups": mechanical[ID]})
    mechanical_summary = pd.DataFrame([{
        "样本量": int(gap_model.nobs),
        "公司数": mechanical[ID].nunique(),
        "Gap与行业内相对规模相关系数": mechanical[["Gap", "RelativeSize"]].corr().iloc[0, 1],
        "相对规模系数": gap_model.params["RelativeSize"],
        "公司聚类标准误": gap_model.bse["RelativeSize"],
        "p值": gap_model.pvalues["RelativeSize"],
        "95%置信区间下限": gap_model.conf_int().loc["RelativeSize", 0],
        "95%置信区间上限": gap_model.conf_int().loc["RelativeSize", 1],
        "调整R2": gap_model.rsquared_adj,
    }])
    mechanical_summary.to_csv(OUT / "06_mechanical_component_test.csv", index=False, encoding="utf-8-sig")

    mechanical["相对规模组"] = pd.qcut(
        mechanical["RelativeSize"], q=4, labels=["Q1 最小", "Q2", "Q3", "Q4 最大"], duplicates="drop"
    )
    group_summary = mechanical.groupby("相对规模组", observed=True).agg(
        观测数=(ID, "size"), 公司数=(ID, "nunique"),
        相对规模均值=("RelativeSize", "mean"), 原始资产份额均值=("AssetShare", "mean"),
        LOO资产份额均值=("AssetShare_LOO", "mean"), Gap均值=("Gap", "mean"), Gap中位数=("Gap", "median")
    ).reset_index()
    group_summary.to_csv(OUT / "07_mechanical_component_by_size_quartile.csv", index=False, encoding="utf-8-sig")
    (OUT / "08_mechanical_component_full_output.txt").write_text(gap_model.summary().as_text(), encoding="utf-8")

    print(results[["模型", "核心变量", "系数", "公司聚类标准误", "p值", "显著性", "样本量", "调整R2"]].to_string(index=False))
    print("\nWithin-firm variation:\n", pd.DataFrame(variation).to_string(index=False))
    print("\nSample diagnostics:\n", diagnostic.to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
