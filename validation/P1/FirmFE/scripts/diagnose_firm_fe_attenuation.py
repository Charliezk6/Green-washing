"""Diagnose why firm-FE coefficients attenuate relative to industry FE.

Outputs variance decomposition, IMS_D within-switch stats, PanelOLS cross-check
against the demean implementation, and switchers-only firm-FE regressions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from run_firm_fe_regression import (  # noqa: E402
    ALL_CONTROLS,
    CORE,
    DEPENDENT,
    load_data,
    fit,
    significance,
    winsorize_controls,
)

DIAG = ROOT / "outputs" / "diagnostics"
FIRM = "证券代码"
YEAR = "时间"


def prepare_common() -> pd.DataFrame:
    data = load_data()
    required = [DEPENDENT, *CORE, *ALL_CONTROLS, FIRM, YEAR, "ExpertiseIndustryCode"]
    common = data.dropna(subset=required).copy()
    common, _ = winsorize_controls(common)
    return common


def within_sd(series: pd.Series, groups: pd.Series) -> float:
    demeaned = series - series.groupby(groups).transform("mean")
    return float(demeaned.std(ddof=1))


def variance_decomposition(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in [DEPENDENT, "IMS", "IMS_D", *ALL_CONTROLS]:
        y = common[variable]
        firm_mean = y.groupby(common[FIRM]).transform("mean")
        within = y - firm_mean
        # Overall variance (observation-level)
        overall_var = float(y.var(ddof=1))
        within_var = float(within.var(ddof=1))
        # Between: variance of firm means, weighted by firm size via firm-level unique means
        firm_means = y.groupby(common[FIRM]).mean()
        between_var = float(firm_means.var(ddof=1))
        # Share of total (overall) attributed to within vs residual between approximation
        # Classic: overall ≈ within + between (not exact with unbalanced panels)
        total = within_var + between_var
        rows.append({
            "变量": variable,
            "总体方差": overall_var,
            "组内方差": within_var,
            "组间方差_公司均值": between_var,
            "组内方差占比": within_var / total if total > 0 else np.nan,
            "组间方差占比": between_var / total if total > 0 else np.nan,
            "组内标准差": float(within.std(ddof=1)),
            "组间标准差": float(firm_means.std(ddof=1)),
            "近似时不变_组内SD小于1e-8": within.std(ddof=1) < 1e-8,
        })
    return pd.DataFrame(rows)


def correlation_decomposition(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    gw = common[DEPENDENT]
    for core in CORE:
        x = common[core]
        overall = float(gw.corr(x))
        # Between: firm means
        firm = common.groupby(FIRM)[[DEPENDENT, core]].mean()
        between = float(firm[DEPENDENT].corr(firm[core]))
        # Within: demeaned
        gw_w = gw - gw.groupby(common[FIRM]).transform("mean")
        x_w = x - x.groupby(common[FIRM]).transform("mean")
        within = float(gw_w.corr(x_w))
        rows.append({
            "核心变量": core,
            "总体相关": overall,
            "组间相关_公司均值": between,
            "组内相关_去均值": within,
        })
    return pd.DataFrame(rows)


def ims_d_switch_stats(common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = common.groupby(FIRM)["IMS_D"]
    summary = g.agg(
        n_obs="count",
        mean="mean",
        std="std",
        n_unique="nunique",
        min="min",
        max="max",
    ).reset_index()
    years = common.groupby(FIRM)[YEAR].nunique().rename("n_years")
    summary = summary.merge(years, on=FIRM, how="left")

    summary["never_switches"] = summary["n_unique"] <= 1
    summary["always_zero"] = (summary["max"] == 0) & (summary["n_unique"] == 1)
    summary["always_one"] = (summary["min"] == 1) & (summary["n_unique"] == 1)
    summary["switcher"] = summary["n_unique"] > 1

    ordered = common.sort_values([FIRM, YEAR])
    ordered["IMS_D_lag"] = ordered.groupby(FIRM)["IMS_D"].shift(1)
    ordered["year_lag"] = ordered.groupby(FIRM)[YEAR].shift(1)
    consecutive = ordered[YEAR].sub(ordered["year_lag"]).eq(1)
    flips = ordered.loc[consecutive & ordered["IMS_D_lag"].notna()].copy()
    flips["flip"] = flips["IMS_D"].ne(flips["IMS_D_lag"]).astype(int)
    flip_counts = flips.groupby(FIRM)["flip"].sum().rename("n_flips").reset_index()
    summary = summary.merge(flip_counts, on=FIRM, how="left")
    summary["n_flips"] = summary["n_flips"].fillna(0).astype(int)

    ims_within = common["IMS"] - common.groupby(FIRM)["IMS"].transform("mean")
    firms_ims_move = (
        pd.DataFrame({FIRM: common[FIRM], "_w": ims_within})
        .groupby(FIRM)["_w"]
        .apply(lambda s: float(s.abs().max()) > 1e-12)
    )

    firm_years = common.groupby(FIRM)[YEAR].nunique()
    n_firms = len(summary)
    aggregate = pd.DataFrame([{
        "公司数": n_firms,
        "观测数": len(common),
        "仅1年观测公司数": int((summary["n_years"] == 1).sum()),
        "平均每公司年数": float(firm_years.mean()),
        "中位每公司年数": float(firm_years.median()),
        "IMS_D从不切换公司数": int(summary["never_switches"].sum()),
        "IMS_D从不切换公司占比": float(summary["never_switches"].mean()),
        "IMS_D始终为0公司数": int(summary["always_zero"].sum()),
        "IMS_D始终为1公司数": int(summary["always_one"].sum()),
        "IMS_D有切换公司数": int(summary["switcher"].sum()),
        "IMS_D有切换公司占比": float(summary["switcher"].mean()),
        "平均组内切换次数_全部公司": float(summary["n_flips"].mean()),
        "平均组内切换次数_仅切换公司": float(summary.loc[summary["switcher"], "n_flips"].mean())
        if summary["switcher"].any() else np.nan,
        "IMS组内有变动公司数": int(firms_ims_move.sum()),
        "IMS组内有变动公司占比": float(firms_ims_move.mean()),
    }])

    return aggregate, summary


def panelols_fit(common: pd.DataFrame, core: str) -> dict:
    controls = ALL_CONTROLS
    cols = [DEPENDENT, core, *controls, FIRM, YEAR]
    sample = common[cols].dropna().copy()
    sample = sample.set_index([FIRM, YEAR])
    # Drop firms with a single observation (PanelOLS entity effects)
    n_per = sample.groupby(level=0).size()
    keep = n_per[n_per >= 2].index
    sample = sample.loc[sample.index.get_level_values(0).isin(keep)]
    y = sample[DEPENDENT]
    x = sample[[core, *controls]]
    model = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)
    result = model.fit(cov_type="clustered", cluster_entity=True)
    return {
        "核心变量": core,
        "方法": "PanelOLS_EntityTime",
        "系数": float(result.params[core]),
        "标准误": float(result.std_errors[core]),
        "t值": float(result.tstats[core]),
        "p值": float(result.pvalues[core]),
        "显著性": significance(float(result.pvalues[core])),
        "样本数": int(result.nobs),
        "公司数_用于估计": int(sample.index.get_level_values(0).nunique()),
    }


def demean_fit_row(common: pd.DataFrame, core: str) -> dict:
    result, sample = fit(common, core, ALL_CONTROLS)
    return {
        "核心变量": core,
        "方法": "Demean_YearFE_statsmodels",
        "系数": float(result.params[core]),
        "标准误": float(result.bse[core]),
        "t值": float(result.tvalues[core]),
        "p值": float(result.pvalues[core]),
        "显著性": significance(float(result.pvalues[core])),
        "样本数": int(result.nobs),
        "公司数_用于估计": int(sample[FIRM].nunique()),
    }


def crosscheck(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for core in CORE:
        d = demean_fit_row(common, core)
        p = panelols_fit(common, core)
        rows.append(d)
        rows.append(p)
        rows.append({
            "核心变量": core,
            "方法": "系数差_PanelOLS减Demean",
            "系数": p["系数"] - d["系数"],
            "标准误": np.nan,
            "t值": np.nan,
            "p值": np.nan,
            "显著性": "",
            "样本数": np.nan,
            "公司数_用于估计": np.nan,
        })
    return pd.DataFrame(rows)


def control_within_sd(common: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ALL_CONTROLS:
        rows.append({
            "变量": variable,
            "组内标准差": within_sd(common[variable], common[FIRM]),
            "总体标准差": float(common[variable].std(ddof=1)),
        })
    return pd.DataFrame(rows)


def switchers_only_firm_fe(common: pd.DataFrame) -> pd.DataFrame:
    """Restrict to firms with within-firm variation in the core regressor."""
    rows = []
    for core in CORE:
        if core == "IMS_D":
            movers = common.groupby(FIRM)[core].nunique().gt(1)
        else:
            w = common[core] - common.groupby(FIRM)[core].transform("mean")
            movers = w.groupby(common[FIRM]).apply(lambda s: float(s.abs().max()) > 1e-12)
        mover_ids = movers[movers].index
        subsample = common[common[FIRM].isin(mover_ids)].copy()
        result, sample = fit(subsample, core, ALL_CONTROLS)
        rows.append({
            "核心变量": core,
            "子样本": f"{core}_组内有变动",
            "公司数": int(sample[FIRM].nunique()),
            "样本数": int(result.nobs),
            "系数": float(result.params[core]),
            "标准误": float(result.bse[core]),
            "t值": float(result.tvalues[core]),
            "p值": float(result.pvalues[core]),
            "显著性": significance(float(result.pvalues[core])),
            "调整R方": float(result.rsquared_adj),
            "FE": "企业+年份",
        })
        # Also report full-sample firm FE for side-by-side
        full = fit(common, core, ALL_CONTROLS)[0]
        rows.append({
            "核心变量": core,
            "子样本": "全样本_对照",
            "公司数": int(common[FIRM].nunique()),
            "样本数": int(full.nobs),
            "系数": float(full.params[core]),
            "标准误": float(full.bse[core]),
            "t值": float(full.tvalues[core]),
            "p值": float(full.pvalues[core]),
            "显著性": significance(float(full.pvalues[core])),
            "调整R方": float(full.rsquared_adj),
            "FE": "企业+年份",
        })
    return pd.DataFrame(rows)


def main() -> None:
    DIAG.mkdir(parents=True, exist_ok=True)
    common = prepare_common()
    print(f"common_sample={len(common)} firms={common[FIRM].nunique()}")

    var_decomp = variance_decomposition(common)
    var_decomp.to_csv(DIAG / "variance_decomposition.csv", index=False, encoding="utf-8-sig")

    corr = correlation_decomposition(common)
    corr.to_csv(DIAG / "correlation_decomposition.csv", index=False, encoding="utf-8-sig")

    aggregate, firm_level = ims_d_switch_stats(common)
    aggregate.to_csv(DIAG / "ims_d_within_switches.csv", index=False, encoding="utf-8-sig")
    firm_level.to_csv(DIAG / "ims_d_within_switches_by_firm.csv", index=False, encoding="utf-8-sig")

    control_within_sd(common).to_csv(
        DIAG / "control_within_sd.csv", index=False, encoding="utf-8-sig"
    )

    cross = crosscheck(common)
    cross.to_csv(DIAG / "panelols_crosscheck.csv", index=False, encoding="utf-8-sig")

    switchers = switchers_only_firm_fe(common)
    switchers.to_csv(DIAG / "switchers_only_firm_fe.csv", index=False, encoding="utf-8-sig")

    print("Wrote diagnostics to", DIAG)
    print(aggregate.to_string(index=False))
    print(cross.to_string(index=False))
    print(switchers.to_string(index=False))


if __name__ == "__main__":
    main()
