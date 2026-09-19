from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "regression_audit_expertise" / "结果" / "第二轮判断" / "05_complete_diagnostic_sample.csv"
OUT = Path(__file__).resolve().parent
ID, YEAR, INDUSTRY = "证券代码", "时间", "证监会行业新"
AUDITOR, Y = "AccountingFirmName", "Greenwashing连续_基准_w"
CONTROLS = ["Size_w", "Lev_w", "ROA_w", "CFO_w", "Top1_w", "SOE", "Dual", "IndependentDirectorRatio_w", "Age_w", "Big4"]
EXPERTISE = ["AssetShare_w", "AssetShare_LOO_w", "ClientShare_LOO_w"]


def estimate(df, terms, label, include_big4=True):
    controls = [x for x in CONTROLS if include_big4 or x != "Big4"]
    term_columns = []
    for term in terms:
        term_columns.extend(c for c in ["AssetShare_w", "AssetShare_LOO_w", "ClientShare_LOO_w", "RelativeSize", "Gap", "Big4", "RelativeSizeQuartile"] if c in term)
    needed = list(dict.fromkeys([Y, ID, YEAR, INDUSTRY] + term_columns + controls))
    s = df.dropna(subset=needed).copy()
    formula = f'Q("{Y}") ~ ' + " + ".join(terms + controls) + f' + C(Q("{INDUSTRY}")) + C(Q("{YEAR}"))'
    r = smf.ols(formula, data=s).fit(cov_type="cluster", cov_kwds={"groups": s[ID]})
    return label, r, s, formula


def row(label, r, s, term, extra=None):
    ci = r.conf_int().loc[term]
    result = {
        "模型": label, "核心项": term, "系数": r.params[term], "公司聚类标准误": r.bse[term],
        "p值": r.pvalues[term], "95%置信区间下限": ci.iloc[0], "95%置信区间上限": ci.iloc[1],
        "样本量": int(r.nobs), "公司数": s[ID].nunique(), "调整R2": r.rsquared_adj,
    }
    if extra:
        result.update(extra)
    return result


def main():
    df = pd.read_csv(INPUT, dtype={ID: str}, encoding="utf-8-sig")
    numeric = [YEAR, Y, "TotalAssets", "OperatingRevenue", "RelativeSize", "Gap", "Big4", "IndustryClients"] + EXPERTISE + CONTROLS
    for c in set(numeric):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Industry-year quartiles ensure relative size is compared within the relevant market.
    df["RelativeSizeQuartile"] = df.groupby([YEAR, INDUSTRY])["RelativeSize"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=False, duplicates="drop") + 1 if x.notna().sum() >= 4 else np.nan
    )
    quartile_rows = []
    for q in range(1, 5):
        sub = df[df["RelativeSizeQuartile"] == q]
        for x in EXPERTISE:
            label, r, s, _ = estimate(sub, [x], f"Q{q} {x}")
            quartile_rows.append(row(label, r, s, x, {"相对规模组": f"Q{q}"}))
    pd.DataFrame(quartile_rows).to_csv(OUT / "相对规模分组回归.csv", index=False, encoding="utf-8-sig")

    interaction_rows = []
    for x in EXPERTISE:
        terms = [x, f'{x}:C(RelativeSizeQuartile)']
        label, r, s, _ = estimate(df, terms, f"{x}×相对规模组")
        for term in r.params.index:
            if term == x or term.startswith(f"{x}:C(RelativeSizeQuartile)"):
                interaction_rows.append(row(label, r, s, term))
        interaction_terms = [t for t in r.params.index if t.startswith(f"{x}:C(RelativeSizeQuartile)")]
        if interaction_terms:
            test = r.wald_test(" = 0, ".join(interaction_terms) + " = 0", scalar=True)
            interaction_rows.append({"模型": label, "核心项": "交互项联合检验", "p值": float(test.pvalue), "样本量": int(r.nobs), "公司数": s[ID].nunique(), "调整R2": r.rsquared_adj})
    pd.DataFrame(interaction_rows).to_csv(OUT / "相对规模交互检验.csv", index=False, encoding="utf-8-sig")

    # Remove focal firms by industry-year asset rank to test large-client sensitivity.
    df["AssetPctRank"] = df.groupby([YEAR, INDUSTRY])["TotalAssets"].rank(pct=True, method="average")
    df["AssetRankDesc"] = df.groupby([YEAR, INDUSTRY])["TotalAssets"].rank(ascending=False, method="first")
    deletion_specs = {
        "完整样本": pd.Series(True, index=df.index),
        "剔除行业年度第一大企业": df["AssetRankDesc"] > 1,
        "剔除行业年度前5%": df["AssetPctRank"] <= 0.95,
        "剔除行业年度前10%": df["AssetPctRank"] <= 0.90,
        "仅保留行业年度至少10家": df["IndustryClients"] >= 10,
    }
    deletion_rows = []
    for spec, mask in deletion_specs.items():
        sub = df[mask.fillna(False)]
        for x in EXPERTISE:
            label, r, s, _ = estimate(sub, [x], f"{spec}：{x}")
            deletion_rows.append(row(label, r, s, x, {"样本处理": spec}))
    pd.DataFrame(deletion_rows).to_csv(OUT / "大客户剔除稳健性.csv", index=False, encoding="utf-8-sig")

    # Decomposition diagnostic and collinearity statistics.
    decomp_rows = []
    for terms, label in [
        (["AssetShare_LOO_w", "RelativeSize"], "LOO资产份额+相对规模"),
        (["AssetShare_LOO_w", "Gap"], "LOO资产份额+Gap"),
    ]:
        name, r, s, _ = estimate(df, terms, label)
        for term in terms:
            decomp_rows.append(row(name, r, s, term))
        X = s[terms].astype(float).dropna()
        for i, term in enumerate(terms):
            decomp_rows.append({"模型": label, "核心项": f"VIF:{term}", "系数": variance_inflation_factor(X.values, i), "样本量": len(X)})
        decomp_rows.append({"模型": label, "核心项": "条件数（仅核心变量）", "系数": float(np.linalg.cond(np.column_stack([np.ones(len(X)), X.values]))), "样本量": len(X)})
    pd.DataFrame(decomp_rows).to_csv(OUT / "指标分解与共线性诊断.csv", index=False, encoding="utf-8-sig")

    # Distribution diagnostics.
    dist_rows = []
    for x in ["AssetShare", "AssetShare_LOO", "ClientShare_LOO", "RelativeSize", "Gap"]:
        s = pd.to_numeric(df[x], errors="coerce").dropna()
        dist_rows.append({
            "变量": x, "观测数": len(s), "均值": s.mean(), "标准差": s.std(), "最小值": s.min(),
            "P1": s.quantile(.01), "P5": s.quantile(.05), "中位数": s.median(), "P95": s.quantile(.95),
            "P99": s.quantile(.99), "最大值": s.max(), "零值比例": (s == 0).mean(),
        })
    pd.DataFrame(dist_rows).to_csv(OUT / "指标分布诊断.csv", index=False, encoding="utf-8-sig")

    # Big Four subgroup and formal interaction tests.
    big4_rows = []
    for group in [0, 1]:
        sub = df[df["Big4"] == group]
        for x in ["AssetShare_LOO_w", "ClientShare_LOO_w"]:
            label, r, s, _ = estimate(sub, [x], f"Big4={group}：{x}", include_big4=False)
            big4_rows.append(row(label, r, s, x, {"BigFour组": group}))
    for x in ["AssetShare_LOO_w", "ClientShare_LOO_w"]:
        label, r, s, _ = estimate(df, [x, f"{x}:Big4"], f"{x}×Big4")
        big4_rows.append(row(label, r, s, x))
        big4_rows.append(row(label, r, s, f"{x}:Big4"))
    pd.DataFrame(big4_rows).to_csv(OUT / "BigFour异质性检验.csv", index=False, encoding="utf-8-sig")

    # Auditor-switch feasibility. Expertise change is continuous; threshold definitions are descriptive only.
    panel = df.sort_values([ID, YEAR]).copy()
    panel["AuditorChanged"] = panel.groupby(ID)[AUDITOR].transform(lambda x: x.ne(x.shift()) & x.shift().notna())
    for x in ["AssetShare_LOO", "ClientShare_LOO"]:
        panel[f"Delta_{x}"] = panel.groupby(ID)[x].diff()
        panel[f"Prev_{x}"] = panel.groupby(ID)[x].shift()
    thresholds = panel.groupby([YEAR, INDUSTRY])["AssetShare_LOO"].transform(lambda x: x.quantile(.75))
    panel["Expert"] = (panel["AssetShare_LOO"] >= thresholds).astype(float)
    panel["PrevExpert"] = panel.groupby(ID)["Expert"].shift()
    panel["SwitchType"] = "非变更年度"
    changed = panel["AuditorChanged"]
    panel.loc[changed & (panel["PrevExpert"] == 0) & (panel["Expert"] == 1), "SwitchType"] = "非专家转专家"
    panel.loc[changed & (panel["PrevExpert"] == 1) & (panel["Expert"] == 0), "SwitchType"] = "专家转非专家"
    panel.loc[changed & (panel["PrevExpert"] == 1) & (panel["Expert"] == 1), "SwitchType"] = "专家转专家"
    panel.loc[changed & (panel["PrevExpert"] == 0) & (panel["Expert"] == 0), "SwitchType"] = "非专家转非专家"
    panel["FirmObservationNumber"] = panel.groupby(ID).cumcount() + 1
    panel["FirmTotalYears"] = panel.groupby(ID)[YEAR].transform("count")
    panel["AtLeastTwoPreOnePost"] = changed & (panel["FirmObservationNumber"] >= 3) & ((panel["FirmTotalYears"] - panel["FirmObservationNumber"]) >= 1)
    events = panel[changed].copy()
    feasibility = events.groupby("SwitchType").agg(
        事件数=(ID, "size"), 公司数=(ID, "nunique"),
        LOO资产份额平均变化=("Delta_AssetShare_LOO", "mean"),
        LOO客户份额平均变化=("Delta_ClientShare_LOO", "mean"),
        至少两个处理前且一个处理后事件数=("AtLeastTwoPreOnePost", "sum"),
    ).reset_index()
    overall = pd.DataFrame([{
        "SwitchType": "全部变更", "事件数": len(events), "公司数": events[ID].nunique(),
        "LOO资产份额平均变化": events["Delta_AssetShare_LOO"].mean(),
        "LOO客户份额平均变化": events["Delta_ClientShare_LOO"].mean(),
        "至少两个处理前且一个处理后事件数": int(events["AtLeastTwoPreOnePost"].sum()),
    }])
    pd.concat([overall, feasibility], ignore_index=True).to_csv(OUT / "事务所变更DID可行性.csv", index=False, encoding="utf-8-sig")
    events[[ID, YEAR, AUDITOR, "SwitchType", "Delta_AssetShare_LOO", "Delta_ClientShare_LOO", "AtLeastTwoPreOnePost"]].to_csv(
        OUT / "事务所变更事件明细.csv", index=False, encoding="utf-8-sig"
    )

    print("第二阶段检验完成，结果保存至：", OUT)


if __name__ == "__main__":
    main()
