from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
REGRESSION_ROOT = ROOT / "Regression"
CONFIG = json.loads((REGRESSION_ROOT / "config/regression_config.json").read_text(encoding="utf-8"))
INPUT = REGRESSION_ROOT / CONFIG["input_panel"]
REVENUE = REGRESSION_ROOT / CONFIG["input_revenue"]
OUTPUT = Path(__file__).resolve().parent / "outputs"
DEPENDENT = CONFIG["dependent"]
CORE = CONFIG["core_variables"]
FINANCIAL = CONFIG["financial_controls"]
GOVERNANCE = CONFIG["governance_controls"]
AUDIT = CONFIG["audit_controls"]
ALL_CONTROLS = FINANCIAL + GOVERNANCE + AUDIT
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"


def load_data() -> pd.DataFrame:
    panel = pd.read_csv(INPUT, dtype={FIRM: str}, encoding="utf-8-sig", low_memory=False)
    revenue = pd.read_csv(REVENUE, dtype={"Symbol": str}, encoding="utf-8")
    revenue[YEAR] = pd.to_numeric(revenue[YEAR], errors="coerce")
    revenue = revenue.sort_values(["Symbol", YEAR]).drop_duplicates(["Symbol", YEAR], keep="first")
    revenue["LagRevenue_FullMarket"] = revenue.groupby("Symbol")["OperatingRevenue_FullMarket"].shift(1)
    previous = revenue.groupby("Symbol")[YEAR].shift(1)
    revenue.loc[revenue[YEAR].sub(previous).ne(1), "LagRevenue_FullMarket"] = np.nan
    revenue["Growth"] = revenue["OperatingRevenue_FullMarket"] / revenue["LagRevenue_FullMarket"] - 1
    revenue.loc[revenue["LagRevenue_FullMarket"].le(0), "Growth"] = np.nan
    revenue = revenue.replace([np.inf, -np.inf], np.nan).rename(columns={"Symbol": FIRM})
    revenue = revenue[[FIRM, YEAR, "Growth"]]
    data = panel.merge(revenue, on=[FIRM, YEAR], how="left", validate="many_to_one")
    numeric = [DEPENDENT, *CORE, "TotalAssets", "TotalLiabilities", "NetProfit", "OperatingNetCashFlow", "FirmAge", "SOE", "Dual", "DirectorNumber", "IndependentDirectorRatio", "Big4", "TotalAuditFee", "AuditorTenure", "IsChangeFirm"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["Size"] = np.log(data["TotalAssets"].where(data["TotalAssets"] > 0))
    data["Lev"] = data["TotalLiabilities"] / data["TotalAssets"]
    data["ROA"] = data["NetProfit"] / data["TotalAssets"]
    data["CFO"] = data["OperatingNetCashFlow"] / data["TotalAssets"]
    data["FirmAgeLog"] = np.log1p(data["FirmAge"].where(data["FirmAge"] >= 0))
    data["BoardSize"] = np.log(data["DirectorNumber"].where(data["DirectorNumber"] > 0))
    data["LnAuditFee"] = np.log(data["TotalAuditFee"].where(data["TotalAuditFee"] > 0))
    return data.replace([np.inf, -np.inf], np.nan)


def fit(data: pd.DataFrame, core: str, controls: list[str]):
    terms = [f"Q('{core}')", *[f"Q('{variable}')" for variable in controls], f"C(Q('{YEAR}'))", f"C(Q('{INDUSTRY}'))"]
    formula = f"Q('{DEPENDENT}') ~ " + " + ".join(terms)
    result = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[FIRM]})
    return result


def oster_delta(beta_restricted: float, beta_full: float, r_restricted: float, r_full: float, r_max: float) -> float:
    denominator = (beta_restricted - beta_full) * (r_max - r_full)
    if abs(denominator) < 1e-12 or abs(r_full - r_restricted) < 1e-12:
        return np.nan
    return beta_full * (r_full - r_restricted) / denominator


def make_result(data: pd.DataFrame, core: str, restricted_controls: list[str], restricted_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    restricted = fit(data, core, restricted_controls)
    full = fit(data, core, ALL_CONTROLS)
    beta_restricted = float(restricted.params[f"Q('{core}')"])
    beta_full = float(full.params[f"Q('{core}')"])
    r_restricted = float(restricted.rsquared)
    r_full = float(full.rsquared)
    rows = []
    for rmax_name, rmax in [("Rmax=R_full", r_full), ("Rmax=1.3R_full", min(1.3 * r_full, 1.0)), ("Rmax=min(2R_full,1)", min(2.0 * r_full, 1.0))]:
        delta_star = oster_delta(beta_restricted, beta_full, r_restricted, r_full, rmax)
        beta_delta1 = beta_full - (beta_full - beta_restricted) * (rmax - r_full) / (r_full - r_restricted) if r_full != r_restricted else np.nan
        rows.append({"核心变量": core, "限制模型": restricted_name, "限制模型控制": " | ".join(restricted_controls) if restricted_controls else "无", "完整模型控制": " | ".join(ALL_CONTROLS), "限制模型系数": beta_restricted, "完整模型系数": beta_full, "限制模型R2": r_restricted, "完整模型R2": r_full, "Rmax设定": rmax_name, "Rmax": rmax, "Delta使校正系数为零": delta_star, "Delta=1校正系数": beta_delta1, "限制模型样本数": int(restricted.nobs), "完整模型样本数": int(full.nobs), "企业数": data[FIRM].nunique(), "完整模型p值": float(full.pvalues[f"Q('{core}')"]), "完整模型标准误": float(full.bse[f"Q('{core}')"])})
    summary = pd.DataFrame([{"核心变量": core, "限制模型": restricted_name, "限制模型系数": beta_restricted, "完整模型系数": beta_full, "限制模型R2": r_restricted, "完整模型R2": r_full, "系数变化": beta_full - beta_restricted, "完整模型p值": float(full.pvalues[f"Q('{core}')"]), "样本数": int(full.nobs), "企业数": data[FIRM].nunique()}])
    return pd.DataFrame(rows), summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    required = [DEPENDENT, *CORE, *ALL_CONTROLS, FIRM, YEAR, INDUSTRY]
    data = load_data().dropna(subset=required).copy()
    all_results, summaries = [], []
    for core in CORE:
        for restricted_controls, restricted_name in [([], "仅核心变量+年份行业固定效应"), (FINANCIAL, "核心变量+财务控制+年份行业固定效应"), (FINANCIAL + GOVERNANCE, "核心变量+财务治理控制+年份行业固定效应")]:
            result, summary = make_result(data, core, restricted_controls, restricted_name)
            all_results.append(result)
            summaries.append(summary)
    pd.concat(all_results, ignore_index=True).to_csv(OUTPUT / "oster_sensitivity_results.csv", index=False, encoding="utf-8-sig")
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT / "oster_model_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"样本数": len(data), "企业数": data[FIRM].nunique(), "年份数": data[YEAR].nunique(), "行业数": data[INDUSTRY].nunique(), "Rmax设定": "R_full; min(1.3R_full,1); min(2R_full,1)", "方法说明": "Oster (2019)系数稳定性敏感性分析"}]).to_csv(OUTPUT / "oster_sample_diagnostics.csv", index=False, encoding="utf-8-sig")
    print(pd.concat(all_results, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
