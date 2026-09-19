from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACK = Path(__file__).resolve().parents[1]
PROJECT = PACK.parent
PANEL = PACK / "data/input/combined_greenwashing_auditor_expertise_panel.csv"
sys.path.insert(0, str(PROJECT / "validation/panel_version_audit/scripts"))
from soe_impute import reconstruct_soe

SOE_RULE = (
    "any same-year S0702b in {1100,2000,2100,2120}; no actual controller coded 0; "
    "constant within-firm carry if gap<=2; residual missing coded 0"
)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    standard_deviation = values.std(ddof=1)
    if values.notna().sum() <= 1 or pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / standard_deviation


def main() -> None:
    data = pd.read_csv(PANEL, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    data["证券代码"] = data["证券代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    previous_soe = pd.to_numeric(data["SOE"], errors="coerce")
    previous_y = pd.to_numeric(data["Greenwashing连续_基准"], errors="coerce")
    data = reconstruct_soe(data)
    gi = pd.to_numeric(data["GI"], errors="coerce")
    e_score = pd.to_numeric(data["E得分"], errors="coerce")
    industry = data["ExpertiseIndustryCode"].astype(str).str.strip()
    eligible = gi.notna() & e_score.notna() & industry.ne("") & industry.ne("nan")
    data["GI"] = gi
    data["E得分"] = e_score
    data.loc[~eligible, ["GI_z_行业年度", "ESGE_z_行业年度", "Greenwashing连续_基准"]] = np.nan
    groups = data.loc[eligible].groupby(["时间", "ExpertiseIndustryCode"], dropna=False)
    data.loc[eligible, "GI_z_行业年度"] = groups["GI"].transform(zscore)
    data.loc[eligible, "ESGE_z_行业年度"] = groups["E得分"].transform(zscore)
    data["Greenwashing连续_基准"] = data["GI_z_行业年度"] - data["ESGE_z_行业年度"]
    data["连续变量构造状态"] = np.where(data["Greenwashing连续_基准"].notna(), "complete", "insufficient_industry_year_variation")
    data.to_csv(PANEL, index=False, encoding="utf-8-sig")
    new_y = pd.to_numeric(data["Greenwashing连续_基准"], errors="coerce")
    both = previous_y.notna() & new_y.notna()
    summary = {
        "rows": int(len(data)),
        "soe_mean_before": float(previous_soe.mean()) if previous_soe.notna().any() else None,
        "soe_mean_after": float(data["SOE"].mean()),
        "soe_ones_after": int(data["SOE"].eq(1).sum()),
        "soe_source_counts": {str(key): int(value) for key, value in data["SOE_source"].value_counts().items()},
        "y_complete_before": int(previous_y.notna().sum()),
        "y_complete_after": int(new_y.notna().sum()),
        "y_corr_overlapping": float(np.corrcoef(previous_y[both], new_y[both])[0, 1]) if both.sum() > 2 else None,
        "soe_rule": SOE_RULE,
        "y_rule": "industry-year z-score of GI minus E-score using ExpertiseIndustryCode and ddof=1",
    }
    report = PACK / "outputs/validation"
    report.mkdir(parents=True, exist_ok=True)
    (report / "panel_correction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
