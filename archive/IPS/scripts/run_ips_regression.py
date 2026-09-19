from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/regression_config.json").read_text(encoding="utf-8"))
DEPENDENT = CONFIG["dependent"]
CORE = CONFIG["core_variables"]
FINANCIAL = CONFIG["financial_controls"]
GOVERNANCE = CONFIG["governance_controls"]
AUDIT = CONFIG["audit_controls"]
ALL_CONTROLS = FINANCIAL + GOVERNANCE + AUDIT
WINSORIZED_CONTROLS = ["Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "BoardSize", "IndependentDirectorRatio", "LnAuditFee", "AuditorTenure"]
module = sys.modules[__name__]
LABELS = {"IPS": "IPS", "IPS_D": "IPS_D", "Size": "公司规模", "Lev": "资产负债率", "ROA": "总资产收益率", "CFO": "经营现金流", "Growth": "营业收入增长率", "FirmAge": "上市年限", "SOE": "产权性质", "Dual": "两职合一", "BoardSize": "董事会规模", "IndependentDirectorRatio": "独立董事比例", "Big4": "国际四大", "LnAuditFee": "审计收费", "AuditorTenure": "审计任期", "IsChangeFirm": "事务所变更", "Intercept": "常数项"}


def significance(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def load_data() -> pd.DataFrame:
    panel = pd.read_csv((ROOT / CONFIG["input_panel"]).resolve(), dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    revenue = pd.read_csv((ROOT / CONFIG["input_revenue"]).resolve(), dtype={"Symbol": str}, encoding="utf-8-sig")
    revenue["时间"] = pd.to_numeric(revenue["时间"], errors="coerce")
    revenue = revenue.sort_values(["Symbol", "时间"]).drop_duplicates(["Symbol", "时间"], keep="first")
    revenue["LagRevenue_FullMarket"] = revenue.groupby("Symbol")["OperatingRevenue_FullMarket"].shift(1)
    previous = revenue.groupby("Symbol")["时间"].shift(1)
    revenue.loc[revenue["时间"].sub(previous).ne(1), "LagRevenue_FullMarket"] = np.nan
    revenue["Growth"] = revenue["OperatingRevenue_FullMarket"] / revenue["LagRevenue_FullMarket"] - 1
    revenue.loc[revenue["LagRevenue_FullMarket"].le(0), "Growth"] = np.nan
    revenue = revenue.replace([np.inf, -np.inf], np.nan).rename(columns={"Symbol": "证券代码"})[["证券代码", "时间", "OperatingRevenue_FullMarket", "LagRevenue_FullMarket", "Growth"]]
    data = panel.merge(revenue, on=["证券代码", "时间"], how="left", validate="many_to_one")
    numeric = [DEPENDENT, *CORE, "TotalAssets", "TotalLiabilities", "OperatingRevenue", "NetProfit", "OperatingNetCashFlow", "FirmAge", "SOE", "Dual", "DirectorNumber", "IndependentDirectorRatio", "Big4", "TotalAuditFee", "AuditorTenure", "IsChangeFirm"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Size"] = np.log(data["TotalAssets"].where(data["TotalAssets"] > 0))
    data["Lev"] = data["TotalLiabilities"] / data["TotalAssets"]
    data["ROA"] = data["NetProfit"] / data["TotalAssets"]
    data["CFO"] = data["OperatingNetCashFlow"] / data["TotalAssets"]
    data["FirmAge"] = data["FirmAge"].where(data["FirmAge"] >= 0)
    data["BoardSize"] = np.log(data["DirectorNumber"].where(data["DirectorNumber"] > 0))
    data["LnAuditFee"] = np.log(data["TotalAuditFee"].where(data["TotalAuditFee"] > 0))
    return data.replace([np.inf, -np.inf], np.nan).sort_values(["证券代码", "时间"])


def winsorize_controls(data: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 <= lower < upper <= 1:
        raise ValueError("Winsorization quantiles must satisfy 0 <= lower < upper <= 1.")
    result = data.copy()
    records = []
    for variable in WINSORIZED_CONTROLS:
        lower_bound, upper_bound = result[variable].quantile([lower, upper])
        original = result[variable]
        result[variable] = original.clip(lower=lower_bound, upper=upper_bound)
        records.append({"变量": variable, "下分位数": lower, "上分位数": upper, "缩尾下限": lower_bound, "缩尾上限": upper_bound, "下尾调整观测数": int(original.lt(lower_bound).sum()), "上尾调整观测数": int(original.gt(upper_bound).sum())})
    return result, pd.DataFrame(records)


def fit(data: pd.DataFrame, core: str, controls: list[str]):
    required = [DEPENDENT, core, "证券代码", *controls, "时间", "ExpertiseIndustryCode"]
    sample = data.dropna(subset=required).copy()
    terms = [core, *controls, "C(时间)", "C(ExpertiseIndustryCode)"]
    model = smf.ols(f"Q('{DEPENDENT}') ~ " + " + ".join(terms), data=sample)
    return model.fit(cov_type="cluster", cov_kwds={"groups": sample["证券代码"]}), sample


def tidy(result, name: str, core: str) -> pd.DataFrame:
    output = []
    for variable in ["Intercept", core, *ALL_CONTROLS]:
        if variable not in result.params.index:
            continue
        interval = result.conf_int().loc[variable]
        output.append({"模型": name, "核心变量": core, "变量": variable, "变量名称": LABELS.get(variable, variable), "系数": result.params[variable], "标准误": result.bse[variable], "t值": result.tvalues[variable], "p值": result.pvalues[variable], "显著性": significance(result.pvalues[variable]), "置信区间下限": interval[0], "置信区间上限": interval[1], "标准误类型": "公司聚类", "年度固定效应": "是", "行业固定效应": "是", "企业固定效应": "否", "样本数": int(result.nobs), "调整R方": result.rsquared_adj})
    return pd.DataFrame(output)


def main() -> None:
    data = load_data()
    output = ROOT / CONFIG["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    specs = [("(1) Univariate", []), ("(2) Financial Controls", FINANCIAL), ("(3) Financial and Governance Controls", FINANCIAL + GOVERNANCE), ("(4) Full Controls", ALL_CONTROLS)]
    required = [DEPENDENT, *CORE, *ALL_CONTROLS, "证券代码", "时间", "ExpertiseIndustryCode"]
    common = data.dropna(subset=required).copy()
    if len(common) != CONFIG["expected_common_sample"]:
        raise ValueError(f"Expected common sample of {CONFIG['expected_common_sample']}, got {len(common)}.")
    common, winsorization = winsorize_controls(common)
    winsorization.to_csv(output / "continuous_control_winsorization_bounds.csv", index=False, encoding="utf-8-sig")
    tables, sample_rows = [], []
    for core in CORE:
        for name, controls in specs:
            result, sample = fit(common, core, controls)
            full_name = f"{core} {name}"
            tables.append(tidy(result, full_name, core))
            sample_rows.append({"模型": full_name, "核心变量": core, "样本数": len(sample), "公司数": sample["证券代码"].nunique(), "年度数": sample["时间"].nunique(), "行业数": sample["ExpertiseIndustryCode"].nunique(), "企业固定效应": "否"})
    pd.concat(tables, ignore_index=True).to_csv(output / "ips_regression_coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sample_rows).to_csv(output / "ips_regression_sample_info.csv", index=False, encoding="utf-8-sig")
    data[[DEPENDENT, *CORE, *ALL_CONTROLS]].describe().T.join(data[[DEPENDENT, *CORE, *ALL_CONTROLS]].isna().sum().rename("缺失数")).to_csv(output / "ips_regression_variable_diagnostics.csv", encoding="utf-8-sig")
    common.to_csv(output / "ips_common_sample_14778.csv", index=False, encoding="utf-8-sig")
    print(f"raw={len(data)} common_sample={len(common)}\noutputs={output}")


if __name__ == "__main__":
    main()
