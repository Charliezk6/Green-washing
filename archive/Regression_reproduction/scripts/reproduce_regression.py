from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config/regression_config.json").read_text(encoding="utf-8"))
DEPENDENT = CONFIG["dependent"]
CORE = CONFIG["core_variables"]
FINANCIAL = CONFIG["financial_controls"]
GOVERNANCE = CONFIG["governance_controls"]
AUDIT = CONFIG["audit_controls"]
ALL_CONTROLS = FINANCIAL + GOVERNANCE + AUDIT
LABELS = {"IMS":"IMS", "IMS_D":"IMS_D", "Size":"公司规模", "Lev":"资产负债率", "ROA":"总资产收益率", "CFO":"经营现金流", "Growth":"营业收入增长率", "FirmAgeLog":"公司年龄", "SOE":"产权性质", "Dual":"两职合一", "BoardSize":"董事会规模", "IndependentDirectorRatio":"独立董事比例", "Big4":"国际四大", "LnAuditFee":"审计收费", "AuditorTenure":"审计任期", "IsChangeFirm":"事务所变更", "Intercept":"常数项"}


def significance(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def load_data() -> pd.DataFrame:
    panel = pd.read_csv(ROOT / CONFIG["input_panel"], dtype={"证券代码": str}, encoding="utf-8-sig")
    revenue = pd.read_csv(ROOT / CONFIG["input_revenue"], dtype={"Symbol": str}, encoding="utf-8-sig")
    revenue["时间"] = pd.to_numeric(revenue["时间"], errors="coerce")
    revenue = revenue.sort_values(["Symbol", "时间"]).drop_duplicates(["Symbol", "时间"], keep="first")
    revenue["LagRevenue_FullMarket"] = revenue.groupby("Symbol")["OperatingRevenue_FullMarket"].shift(1)
    previous = revenue.groupby("Symbol")["时间"].shift(1)
    revenue.loc[revenue["时间"].sub(previous).ne(1), "LagRevenue_FullMarket"] = np.nan
    revenue["Growth"] = revenue["OperatingRevenue_FullMarket"] / revenue["LagRevenue_FullMarket"] - 1
    revenue.loc[revenue["LagRevenue_FullMarket"].le(0), "Growth"] = np.nan
    revenue.replace([np.inf, -np.inf], np.nan, inplace=True)
    revenue = revenue.rename(columns={"Symbol": "证券代码"})[["证券代码", "时间", "OperatingRevenue_FullMarket", "LagRevenue_FullMarket", "Growth"]]
    data = panel.merge(revenue, on=["证券代码", "时间"], how="left", validate="many_to_one")
    numeric = [DEPENDENT, *CORE, "TotalAssets", "TotalLiabilities", "OperatingRevenue", "NetProfit", "OperatingNetCashFlow", "FirmAge", "SOE", "Dual", "DirectorNumber", "IndependentDirectorRatio", "Big4", "TotalAuditFee", "AuditorTenure", "IsChangeFirm"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Size"] = np.log(data["TotalAssets"].where(data["TotalAssets"] > 0))
    data["Lev"] = data["TotalLiabilities"] / data["TotalAssets"]
    data["ROA"] = data["NetProfit"] / data["TotalAssets"]
    data["CFO"] = data["OperatingNetCashFlow"] / data["TotalAssets"]
    data["FirmAgeLog"] = np.log1p(data["FirmAge"].where(data["FirmAge"] >= 0))
    data["BoardSize"] = np.log(data["DirectorNumber"].where(data["DirectorNumber"] > 0))
    data["LnAuditFee"] = np.log(data["TotalAuditFee"].where(data["TotalAuditFee"] > 0))
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data.sort_values(["证券代码", "时间"])


def fit(data: pd.DataFrame, core: str, controls: list[str], fe: bool, cov: str):
    terms = [core, *controls]
    if fe:
        terms += ["C(时间)", "C(ExpertiseIndustryCode)"]
    required = [DEPENDENT, core, "证券代码", *controls]
    if fe:
        required += ["时间", "ExpertiseIndustryCode"]
    sample = data.dropna(subset=required).copy()
    formula = f"Q('{DEPENDENT}') ~ " + " + ".join(terms)
    model = smf.ols(formula, data=sample)
    if cov == "cluster":
        result = model.fit(cov_type="cluster", cov_kwds={"groups": sample["证券代码"]})
    elif cov == "industry_cluster":
        result = model.fit(cov_type="cluster", cov_kwds={"groups": sample["ExpertiseIndustryCode"]})
    elif cov == "HC3":
        result = model.fit(cov_type="HC3")
    else:
        result = model.fit()
    return result, sample


def tidy(result, name: str, core: str, cov_label: str, fe: bool) -> pd.DataFrame:
    rows = []
    for variable in ["Intercept", core, *ALL_CONTROLS]:
        if variable not in result.params.index:
            continue
        interval = result.conf_int().loc[variable]
        rows.append({"模型": name, "核心变量": core, "变量": variable, "变量名称": LABELS.get(variable, variable), "系数": result.params[variable], "标准误": result.bse[variable], "t值": result.tvalues[variable], "p值": result.pvalues[variable], "显著性": significance(result.pvalues[variable]), "置信区间下限": interval[0], "置信区间上限": interval[1], "标准误类型": cov_label, "年度固定效应": "是" if fe else "否", "行业固定效应": "是" if fe else "否", "样本数": int(result.nobs), "调整R方": result.rsquared_adj})
    return pd.DataFrame(rows)


def main() -> None:
    data = load_data()
    output = ROOT / CONFIG["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    specifications = [("(1) 一元回归", [], False), ("(2) 财务控制", FINANCIAL, False), ("(3) 财务与治理控制", FINANCIAL + GOVERNANCE, False), ("(4) 全部控制", ALL_CONTROLS, False), ("(5) 全部控制+双向固定效应", ALL_CONTROLS, True)]
    common_required = [DEPENDENT, *CORE, *ALL_CONTROLS, "证券代码", "时间", "ExpertiseIndustryCode"]
    common = data.dropna(subset=common_required).copy()
    models, tidy_tables, sample_rows = [], [], []
    for core in CORE:
        for name, controls, fe in specifications:
            result, sample = fit(common, core, controls, fe, "cluster")
            full_name = f"{core} {name}"
            models.append({"name": full_name, "result": result, "core": core, "fe": fe, "cov": "cluster"})
            tidy_tables.append(tidy(result, full_name, core, "公司聚类", fe))
            sample_rows.append({"模型": full_name, "核心变量": core, "样本数": len(sample), "公司数": sample["证券代码"].nunique(), "年度数": sample["时间"].nunique(), "行业数": sample["ExpertiseIndustryCode"].nunique()})
        for cov, label, suffix in [("HC3", "HC3稳健", "_HC3"), ("industry_cluster", "行业聚类", "_行业聚类")]:
            result, sample = fit(common, core, ALL_CONTROLS, True, cov)
            full_name = f"{core} (6) 全部控制+双向固定效应{suffix}" if cov == "HC3" else f"{core} (7) 全部控制+双向固定效应{suffix}"
            models.append({"name": full_name, "result": result, "core": core, "fe": True, "cov": cov})
            tidy_tables.append(tidy(result, full_name, core, label, True))
            sample_rows.append({"模型": full_name, "核心变量": core, "样本数": len(sample), "公司数": sample["证券代码"].nunique(), "年度数": sample["时间"].nunique(), "行业数": sample["ExpertiseIndustryCode"].nunique()})
    tidy_table = pd.concat(tidy_tables, ignore_index=True)
    samples = pd.DataFrame(sample_rows)
    diagnostics = data[[DEPENDENT, *CORE, *ALL_CONTROLS]].describe().T.join(data[[DEPENDENT, *CORE, *ALL_CONTROLS]].isna().sum().rename("缺失数"))
    data.to_csv(output / "已构造基准回归变量的面板.csv", index=False, encoding="utf-8-sig")
    tidy_table.to_csv(output / "基准回归完整系数.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(output / "基准回归样本信息.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output / "回归变量诊断.csv", encoding="utf-8-sig")
    print(f"raw={len(data)} common_sample={len(common)}")
    print(f"outputs={output}")


if __name__ == "__main__":
    main()
