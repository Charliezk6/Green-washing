from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

V1_PANEL = ROOT / "archive/Regression_reproduction/data/input/greenwashing_auditor_expertise_panel.csv"
V1_COMMON = ROOT / "archive/Regression_reproduction/outputs/paper_tables/common_sample_8129.csv"
V1_REVENUE = ROOT / "archive/Regression_reproduction/data/input/full_market_financial_revenue_2020_2024.csv"
V2_PANEL = ROOT / "Regression/data/input/combined_greenwashing_auditor_expertise_panel.csv"
V2_COMMON = ROOT / "Regression/outputs/paper_tables/common_sample_14778.csv"
V2_REVENUE = ROOT / "Regression/data/input/full_market_financial_revenue_2020_2024.csv"
Y_EXPANSION = ROOT / "Y_variable_reproduction/outputs/combined/greenwashing_panel_expansion.csv"
Y_ORIG_REPRO = ROOT / "Y_variable_reproduction/outputs/combined/greenwashing_panel_all_industries_2021_2024.csv"
Y_ORIG_ARCHIVE = ROOT / "archive/Y_variable_origional/greenwashing_panel_all_industries_2021_2024.csv"

DEPENDENT = "Greenwashing连续_基准"
WINSORIZED = [
    "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "BoardSize",
    "IndependentDirectorRatio", "LnAuditFee", "AuditorTenure",
]
ALL_CONTROLS_V1 = [
    "Size", "Lev", "ROA", "CFO", "Growth", "FirmAgeLog", "SOE", "Dual",
    "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee",
    "AuditorTenure", "IsChangeFirm",
]
ALL_CONTROLS_V2 = [
    "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "SOE", "Dual",
    "BoardSize", "IndependentDirectorRatio", "Big4", "LnAuditFee",
    "AuditorTenure", "IsChangeFirm",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)


def pad_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)


def keys(frame: pd.DataFrame) -> pd.Series:
    return pad_code(frame["证券代码"]) + "_" + pd.to_numeric(frame["时间"], errors="coerce").astype("Int64").astype(str)


def attach_growth(panel: pd.DataFrame, revenue_path: Path) -> pd.DataFrame:
    revenue = pd.read_csv(revenue_path, dtype={"Symbol": str}, encoding="utf-8-sig")
    revenue["时间"] = pd.to_numeric(revenue["时间"], errors="coerce")
    revenue["证券代码"] = pad_code(revenue["Symbol"])
    revenue = revenue.sort_values(["证券代码", "时间"]).drop_duplicates(["证券代码", "时间"], keep="first")
    revenue["LagRevenue_FullMarket"] = revenue.groupby("证券代码")["OperatingRevenue_FullMarket"].shift(1)
    previous = revenue.groupby("证券代码")["时间"].shift(1)
    revenue.loc[revenue["时间"].sub(previous).ne(1), "LagRevenue_FullMarket"] = np.nan
    revenue["Growth"] = revenue["OperatingRevenue_FullMarket"] / revenue["LagRevenue_FullMarket"] - 1
    revenue.loc[revenue["LagRevenue_FullMarket"].le(0), "Growth"] = np.nan
    revenue = revenue.replace([np.inf, -np.inf], np.nan)
    keep = ["证券代码", "时间", "OperatingRevenue_FullMarket", "LagRevenue_FullMarket", "Growth"]
    data = panel.copy()
    data["证券代码"] = pad_code(data["证券代码"])
    data["时间"] = pd.to_numeric(data["时间"], errors="coerce")
    return data.merge(revenue[keep], on=["证券代码", "时间"], how="left", validate="many_to_one")


def construct_controls(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    numeric = [
        DEPENDENT, "IMS", "IMS_D", "TotalAssets", "TotalLiabilities", "OperatingRevenue",
        "NetProfit", "OperatingNetCashFlow", "FirmAge", "SOE", "Dual", "DirectorNumber",
        "IndependentDirectorRatio", "Big4", "TotalAuditFee", "AuditorTenure", "IsChangeFirm",
        "GI", "E得分", "Y1001b", "InternationalBig4",
    ]
    for column in numeric:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["Size"] = np.log(out["TotalAssets"].where(out["TotalAssets"] > 0))
    out["Lev"] = out["TotalLiabilities"] / out["TotalAssets"]
    out["ROA"] = out["NetProfit"] / out["TotalAssets"]
    out["CFO"] = out["OperatingNetCashFlow"] / out["TotalAssets"]
    out["FirmAge"] = out["FirmAge"].where(out["FirmAge"] >= 0)
    out["FirmAgeLog"] = np.log1p(out["FirmAge"])
    out["BoardSize"] = np.log(out["DirectorNumber"].where(out["DirectorNumber"] > 0))
    out["LnAuditFee"] = np.log(out["TotalAuditFee"].where(out["TotalAuditFee"] > 0))
    return out.replace([np.inf, -np.inf], np.nan)


def winsorize(frame: pd.DataFrame, variables: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = frame.copy()
    records = []
    for variable in variables:
        lower_bound, upper_bound = result[variable].quantile([0.01, 0.99])
        original = result[variable]
        result[variable] = original.clip(lower=lower_bound, upper=upper_bound)
        records.append({
            "变量": variable,
            "缩尾下限": float(lower_bound),
            "缩尾上限": float(upper_bound),
            "下尾调整观测数": int(original.lt(lower_bound).sum()),
            "上尾调整观测数": int(original.gt(upper_bound).sum()),
        })
    return result, pd.DataFrame(records)


def zscore_group(frame: pd.DataFrame, group_col: str, ddof: int, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    gi = pd.to_numeric(out["GI"], errors="coerce")
    e = pd.to_numeric(out["E得分"], errors="coerce")
    groups = out.groupby(["时间", group_col], dropna=False)

    def z(series: pd.Series) -> pd.Series:
        sd = series.std(ddof=ddof)
        n = series.notna().sum()
        if n <= 1 or pd.isna(sd) or sd == 0:
            return pd.Series(np.nan, index=series.index)
        return (series - series.mean()) / sd

    out[f"{prefix}_GI_z"] = groups["GI"].transform(lambda s: z(pd.to_numeric(s, errors="coerce")))
    out[f"{prefix}_E_z"] = groups["E得分"].transform(lambda s: z(pd.to_numeric(s, errors="coerce")))
    out[f"{prefix}_Y"] = out[f"{prefix}_GI_z"] - out[f"{prefix}_E_z"]
    out[f"{prefix}_n"] = groups["GI"].transform(lambda s: pd.to_numeric(s, errors="coerce").notna().sum())
    out.loc[gi.isna() | e.isna() | out[group_col].isna() | (out[group_col].astype(str) == ""), f"{prefix}_Y"] = np.nan
    return out


def compare_y(stored: pd.Series, rebuilt: pd.Series) -> dict[str, float]:
    mask = stored.notna() & rebuilt.notna()
    if mask.sum() < 3:
        return {"n": int(mask.sum()), "corr": np.nan, "rmse": np.nan, "mean_diff": np.nan, "max_abs": np.nan}
    a, b = stored[mask], rebuilt[mask]
    return {
        "n": int(mask.sum()),
        "corr": float(np.corrcoef(a, b)[0, 1]),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "mean_diff": float((a - b).mean()),
        "max_abs": float((a - b).abs().max()),
    }


def fit_core(sample: pd.DataFrame, y: str, core: str, controls: list[str], cluster: str = "证券代码") -> dict:
    use = sample.dropna(subset=[y, core, *controls, "时间", "ExpertiseIndustryCode", cluster]).copy()
    terms = [core, *controls, "C(时间)", "C(ExpertiseIndustryCode)"]
    model = smf.ols(f"Q('{y}') ~ " + " + ".join(terms), data=use)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": use[cluster]})
    return {
        "core": core,
        "n": int(result.nobs),
        "coef": float(result.params[core]),
        "se": float(result.bse[core]),
        "t": float(result.tvalues[core]),
        "p": float(result.pvalues[core]),
        "r2_adj": float(result.rsquared_adj),
        "stars": "***" if result.pvalues[core] < 0.01 else "**" if result.pvalues[core] < 0.05 else "*" if result.pvalues[core] < 0.10 else "",
    }


def main() -> None:
    v1_panel = construct_controls(attach_growth(read_csv(V1_PANEL), V1_REVENUE))
    v2_panel = construct_controls(attach_growth(read_csv(V2_PANEL), V2_REVENUE))
    v1_common_raw = read_csv(V1_COMMON)
    v2_common_raw = read_csv(V2_COMMON)
    for frame in (v1_panel, v2_panel, v1_common_raw, v2_common_raw):
        frame["证券代码"] = pad_code(frame["证券代码"])
        frame["时间"] = pd.to_numeric(frame["时间"], errors="coerce")
        frame["key"] = keys(frame)

    orig_keys = set(v1_panel["key"])
    v2_panel["source"] = np.where(v2_panel["key"].isin(orig_keys), "original", "expansion")
    v1_common_keys = set(v1_common_raw["key"])
    v2_common_keys = set(v2_common_raw["key"])

    required_v2 = [DEPENDENT, "IMS", "IMS_D", *ALL_CONTROLS_V2, "证券代码", "时间", "ExpertiseIndustryCode"]
    v2_rebuilt_common = v2_panel.dropna(subset=required_v2).copy()

    overlap = v1_common_keys & v2_common_keys
    only_v1 = v1_common_keys - v2_common_keys
    only_v2 = v2_common_keys - v1_common_keys
    v2_from_orig = set(v2_panel.loc[v2_panel["source"].eq("original") & v2_panel["key"].isin(v2_common_keys), "key"])
    v2_from_exp = set(v2_panel.loc[v2_panel["source"].eq("expansion") & v2_panel["key"].isin(v2_common_keys), "key"])

    sample_flow = pd.DataFrame([
        {"stage": "v1_input_panel", "n": len(v1_panel), "firms": v1_panel["证券代码"].nunique()},
        {"stage": "v1_common_8129", "n": len(v1_common_keys), "firms": v1_common_raw["证券代码"].nunique()},
        {"stage": "v2_input_combined", "n": len(v2_panel), "firms": v2_panel["证券代码"].nunique()},
        {"stage": "v2_input_original_rows", "n": int(v2_panel["source"].eq("original").sum()), "firms": v2_panel.loc[v2_panel["source"].eq("original"), "证券代码"].nunique()},
        {"stage": "v2_input_expansion_rows", "n": int(v2_panel["source"].eq("expansion").sum()), "firms": v2_panel.loc[v2_panel["source"].eq("expansion"), "证券代码"].nunique()},
        {"stage": "v2_common_14778_file", "n": len(v2_common_keys), "firms": v2_common_raw["证券代码"].nunique()},
        {"stage": "v2_common_rebuilt_dropna", "n": len(v2_rebuilt_common), "firms": v2_rebuilt_common["证券代码"].nunique()},
        {"stage": "overlap_v1_v2_common", "n": len(overlap), "firms": pd.Series([k.split("_")[0] for k in overlap]).nunique()},
        {"stage": "only_v1_common", "n": len(only_v1), "firms": pd.Series([k.split("_")[0] for k in only_v1]).nunique() if only_v1 else 0},
        {"stage": "only_v2_common", "n": len(only_v2), "firms": pd.Series([k.split("_")[0] for k in only_v2]).nunique()},
        {"stage": "v2_common_from_original", "n": len(v2_from_orig), "firms": pd.Series([k.split("_")[0] for k in v2_from_orig]).nunique()},
        {"stage": "v2_common_from_expansion", "n": len(v2_from_exp), "firms": pd.Series([k.split("_")[0] for k in v2_from_exp]).nunique()},
    ])
    sample_flow.to_csv(OUT / "sample_flow.csv", index=False, encoding="utf-8-sig")

    missing_steps = []
    remaining = v2_panel.copy()
    for col in [DEPENDENT, "Growth", "SOE", "Dual", "LnAuditFee", "AuditorTenure", "IMS", "Size", "Lev", "ROA", "CFO", "FirmAge", "BoardSize", "IndependentDirectorRatio", "Big4", "IsChangeFirm", "ExpertiseIndustryCode"]:
        before = len(remaining)
        remaining = remaining.dropna(subset=[col])
        missing_steps.append({"drop": col, "before": before, "after": len(remaining), "lost": before - len(remaining)})
    pd.DataFrame(missing_steps).to_csv(OUT / "v2_missingness_flow.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame({"key": sorted(only_v1)}).to_csv(OUT / "keys_only_v1_common.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"key": sorted(only_v2)}).to_csv(OUT / "keys_only_v2_common.csv", index=False, encoding="utf-8-sig")

    # --- SOE ---
    soe = v2_panel.copy()
    soe["S0701b_nonempty"] = soe["S0701b"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}).notna()
    soe["S0702b_code"] = pd.to_numeric(soe["S0702b"], errors="coerce")
    orig_soe = soe.loc[soe["source"].eq("original") & soe["SOE"].notna() & soe["S0702b_code"].notna()].copy()
    code_means = orig_soe.groupby("S0702b_code")["SOE"].agg(["mean", "count"]).reset_index().sort_values("count", ascending=False)
    code_means.to_csv(OUT / "soe_s0702b_original_mapping.csv", index=False, encoding="utf-8-sig")
    soe_codes = set(orig_soe.loc[orig_soe["SOE"].eq(1), "S0702b_code"].round(0).astype(int).unique())
    nonsoe_codes = set(orig_soe.loc[orig_soe["SOE"].eq(0), "S0702b_code"].round(0).astype(int).unique())
    exclusive_soe = sorted(soe_codes - nonsoe_codes)
    soe["SOE_recode_s0702b"] = np.where(
        soe["S0702b_code"].isna(),
        np.nan,
        soe["S0702b_code"].round(0).astype("Int64").isin(exclusive_soe).astype(float),
    )
    soe["SOE_name_rule"] = soe["S0701b_nonempty"].astype(float)
    soe["first_digit"] = soe["S0702b"].astype(str).str.replace(r"\.0$", "", regex=True).str[0]
    soe["SOE_firstdigit_12"] = np.where(soe["S0702b_code"].isna(), np.nan, soe["first_digit"].isin(["1", "2"]).astype(float))

    soe_rows = []
    scopes = {
        "v2_input_original": soe["source"].eq("original"),
        "v2_input_expansion": soe["source"].eq("expansion"),
        "v2_input_all": pd.Series(True, index=soe.index),
        "v2_common": soe["key"].isin(v2_common_keys),
        "v2_common_original": soe["key"].isin(v2_from_orig),
        "v2_common_expansion": soe["key"].isin(v2_from_exp),
        "v1_common_overlap": soe["key"].isin(overlap),
    }
    for name, mask in scopes.items():
        sub = soe.loc[mask]
        soe_rows.append({
            "scope": name,
            "n": len(sub),
            "SOE_mean_stored": float(sub["SOE"].mean()) if sub["SOE"].notna().any() else np.nan,
            "SOE_n": int(sub["SOE"].notna().sum()),
            "SOE1_stored": int(sub["SOE"].eq(1).sum()),
            "name_nonempty_mean": float(sub["S0701b_nonempty"].mean()),
            "SOE_eq_name_rule": float((sub["SOE"] == sub["SOE_name_rule"]).mean()) if len(sub) else np.nan,
            "recode_s0702b_mean": float(sub["SOE_recode_s0702b"].mean()) if sub["SOE_recode_s0702b"].notna().any() else np.nan,
            "recode_firstdigit12_mean": float(sub["SOE_firstdigit_12"].mean()) if sub["SOE_firstdigit_12"].notna().any() else np.nan,
            "stored_vs_recode_agree": float((sub["SOE"] == sub["SOE_recode_s0702b"]).mean()) if sub["SOE"].notna().any() else np.nan,
        })
    pd.DataFrame(soe_rows).to_csv(OUT / "soe_audit_summary.csv", index=False, encoding="utf-8-sig")
    (OUT / "soe_rule.json").write_text(json.dumps({
        "exclusive_soe_s0702b_codes": [int(x) for x in exclusive_soe],
        "soe_codes_seen_with_SOE1": [int(x) for x in sorted(soe_codes)],
        "nonsoe_codes_seen_with_SOE0": [int(x) for x in sorted(nonsoe_codes)],
        "overlap_codes": [int(x) for x in sorted(soe_codes & nonsoe_codes)],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    exp_common = soe.loc[soe["key"].isin(v2_from_exp)]
    soe_expansion_flag = {
        "expansion_common_n": int(len(exp_common)),
        "expansion_SOE_all_one": bool(exp_common["SOE"].fillna(-1).eq(1).all()) if len(exp_common) else False,
        "expansion_SOE_mean": float(exp_common["SOE"].mean()) if len(exp_common) else np.nan,
        "expansion_name_nonempty_mean": float(exp_common["S0701b_nonempty"].mean()) if len(exp_common) else np.nan,
        "expansion_SOE_equals_name_nonempty": float((exp_common["SOE"] == exp_common["SOE_name_rule"]).mean()) if len(exp_common) else np.nan,
    }

    # --- Y reconstruction ---
    y_exp = read_csv(Y_EXPANSION)
    y_orig_repro = read_csv(Y_ORIG_REPRO)
    y_orig_arch = read_csv(Y_ORIG_ARCHIVE)
    for frame, ycol in ((y_exp, "Greenwashing连续"), (y_orig_repro, "Greenwashing连续_基准"), (y_orig_arch, "Greenwashing连续_基准")):
        frame["证券代码"] = pad_code(frame["证券代码"])
        frame["时间"] = pd.to_numeric(frame["时间"], errors="coerce")
        frame["key"] = keys(frame)
        frame["y_source"] = pd.to_numeric(frame[ycol], errors="coerce")
    y_lookup = pd.concat([
        y_orig_arch[["key", "证监会行业新", "y_source"]].assign(y_file="archive_y_origional"),
        y_orig_repro[["key", "证监会行业新", "y_source"]].assign(y_file="y_repro_original_universe"),
        y_exp[["key", "行业基准分组", "证监会行业新", "y_source"]].assign(y_file="y_repro_expansion"),
    ], ignore_index=True)

    y = v2_panel.copy()
    y = y.merge(y_exp[["key", "行业基准分组"]].drop_duplicates(), on="key", how="left")
    y["group_expertise"] = y["ExpertiseIndustryCode"].astype(str)
    y["group_industry_name"] = y["IndustryName"].astype(str)
    y["group_benchmark"] = y["行业基准分组"].fillna(y["ExpertiseIndustryCode"]).astype(str)
    y["Y_stored"] = pd.to_numeric(y[DEPENDENT], errors="coerce")
    for group_col, ddof, prefix in [
        ("group_industry_name", 0, "name_ddof0"),
        ("group_industry_name", 1, "name_ddof1"),
        ("group_expertise", 0, "exp_ddof0"),
        ("group_expertise", 1, "exp_ddof1"),
        ("group_benchmark", 0, "bench_ddof0"),
        ("group_benchmark", 1, "bench_ddof1"),
    ]:
        y = zscore_group(y, group_col, ddof, prefix)

    y_cmp_rows = []
    for source in ("original", "expansion", "all"):
        sub = y if source == "all" else y.loc[y["source"].eq(source)]
        for prefix in ["name_ddof0", "name_ddof1", "exp_ddof0", "exp_ddof1", "bench_ddof0", "bench_ddof1"]:
            stats = compare_y(sub["Y_stored"], sub[f"{prefix}_Y"])
            stats.update({"source": source, "rebuild": prefix, "scope": "v2_input"})
            y_cmp_rows.append(stats)
        common_sub = sub.loc[sub["key"].isin(v2_common_keys)]
        for prefix in ["name_ddof0", "exp_ddof1", "bench_ddof1"]:
            stats = compare_y(common_sub["Y_stored"], common_sub[f"{prefix}_Y"])
            stats.update({"source": source, "rebuild": prefix, "scope": "v2_common"})
            y_cmp_rows.append(stats)
    pd.DataFrame(y_cmp_rows).to_csv(OUT / "y_rebuild_vs_stored.csv", index=False, encoding="utf-8-sig")

    overlap_y = y.loc[y["key"].isin(overlap), ["key", "source", "Y_stored", "GI", "E得分", "name_ddof0_Y", "exp_ddof0_Y", "exp_ddof1_Y", "bench_ddof1_Y"]]
    v1_y = v1_common_raw[["key", DEPENDENT]].copy()
    v1_y["Y_v1"] = pd.to_numeric(v1_y[DEPENDENT], errors="coerce")
    overlap_y = overlap_y.merge(v1_y[["key", "Y_v1"]], on="key", how="left")
    overlap_y_stats = compare_y(overlap_y["Y_v1"], overlap_y["Y_stored"])
    overlap_y.to_csv(OUT / "y_overlap_values.csv", index=False, encoding="utf-8-sig")

    # --- controls / X ---
    v2_common = v2_panel.loc[v2_panel["key"].isin(v2_common_keys)].copy()
    v2_common["IMS_D_10"] = np.where(v2_common["IMS"].isna(), np.nan, (v2_common["IMS"] >= 0.10).astype(float))
    v2_common["IMS_D_20"] = np.where(v2_common["IMS"].isna(), np.nan, (v2_common["IMS"] >= 0.20).astype(float))
    v2_common_w, winsor_tbl = winsorize(v2_common, WINSORIZED)
    winsor_tbl.to_csv(OUT / "winsorization_bounds.csv", index=False, encoding="utf-8-sig")

    dual_check = v2_panel.copy()
    dual_check["Dual_from_Y1001b"] = np.where(dual_check["Y1001b"].isna(), np.nan, (dual_check["Y1001b"] == 1).astype(float))
    dual_check["Big4_from_flag"] = np.where(dual_check["InternationalBig4"].isna(), np.nan, (dual_check["InternationalBig4"] == 1).astype(float))
    control_rows = []
    for source in ("original", "expansion", "all"):
        sub = dual_check if source == "all" else dual_check.loc[dual_check["source"].eq(source)]
        common_sub = sub.loc[sub["key"].isin(v2_common_keys)]
        idx = slice(None) if source == "all" else v2_common["source"].eq(source)
        control_rows.append({
            "source": source,
            "n_common": len(common_sub),
            "Dual_agree_Y1001b": float((common_sub["Dual"] == common_sub["Dual_from_Y1001b"]).mean()) if len(common_sub) else np.nan,
            "Big4_agree_InternationalBig4": float((common_sub["Big4"] == common_sub["Big4_from_flag"]).mean()) if len(common_sub) else np.nan,
            "IMS_D10_mean": float(v2_common.loc[idx, "IMS_D_10"].mean()),
            "IMS_D20_mean": float(v2_common.loc[idx, "IMS_D_20"].mean()),
            "FirmAge_mean": float(v2_common.loc[idx, "FirmAge"].mean()),
            "FirmAgeLog_mean": float(v2_common.loc[idx, "FirmAgeLog"].mean()),
            "corr_FirmAge_log": float(v2_common.loc[idx, ["FirmAge", "FirmAgeLog"]].corr().iloc[0, 1]),
        })
    pd.DataFrame(control_rows).to_csv(OUT / "controls_x_audit.csv", index=False, encoding="utf-8-sig")

    overlap_ctrl = v1_common_raw.merge(
        v2_panel[["key", "IMS", "SOE", "Dual", "Big4", "FirmAge", DEPENDENT, "source"]],
        on="key", suffixes=("_v1", "_v2"),
    )
    overlap_cmp = []
    for col in ["IMS", "SOE", "Dual", "Big4", "FirmAge", DEPENDENT]:
        a = pd.to_numeric(overlap_ctrl[f"{col}_v1"], errors="coerce")
        b = pd.to_numeric(overlap_ctrl[f"{col}_v2"], errors="coerce")
        mask = a.notna() & b.notna()
        overlap_cmp.append({
            "variable": col,
            "n": int(mask.sum()),
            "corr": float(np.corrcoef(a[mask], b[mask])[0, 1]) if mask.sum() > 2 else np.nan,
            "exact_agree": float(np.isclose(a[mask], b[mask], equal_nan=False).mean()) if mask.sum() else np.nan,
            "mean_v1": float(a[mask].mean()) if mask.sum() else np.nan,
            "mean_v2": float(b[mask].mean()) if mask.sum() else np.nan,
        })
    pd.DataFrame(overlap_cmp).to_csv(OUT / "overlap_field_comparison.csv", index=False, encoding="utf-8-sig")

    # composition original vs expansion in v2 common
    v2_common = v2_common.merge(soe[["key", "SOE_recode_s0702b"]], on="key", how="left")
    comp_vars = ["Size", "Lev", "ROA", "CFO", "Growth", "FirmAge", "IMS", "SOE", "SOE_recode_s0702b", "Dual", "Big4", "IsChangeFirm", "BoardSize", "IndependentDirectorRatio", "LnAuditFee", "AuditorTenure", DEPENDENT]
    comp_rows = []
    for source in ("original", "expansion"):
        sub = v2_common.loc[v2_common["source"].eq(source)]
        row = {"source": source, "n": len(sub), "firms": sub["证券代码"].nunique()}
        for var in comp_vars:
            row[f"{var}_mean"] = float(sub[var].mean()) if var in sub and sub[var].notna().any() else np.nan
        for year, count in sub["时间"].value_counts().sort_index().items():
            row[f"year_{int(year)}"] = int(count)
        comp_rows.append(row)
    pd.DataFrame(comp_rows).to_csv(OUT / "composition_original_vs_expansion.csv", index=False, encoding="utf-8-sig")

    # --- nested regressions (diagnostic only) ---
    overlap_reg = v2_panel.loc[v2_panel["key"].isin(overlap)].copy()
    overlap_reg["Y_v1"] = overlap_reg["key"].map(v1_y.set_index("key")["Y_v1"])
    overlap_reg["IMS_D_10"] = (overlap_reg["IMS"] >= 0.10).astype(float)
    overlap_reg["IMS_D_20"] = (overlap_reg["IMS"] >= 0.20).astype(float)
    overlap_reg_w, _ = winsorize(overlap_reg, WINSORIZED)
    v2_full = v2_panel.loc[v2_panel["key"].isin(v2_common_keys)].copy()
    v2_full["IMS_D_10"] = (v2_full["IMS"] >= 0.10).astype(float)
    v2_full["IMS_D_20"] = (v2_full["IMS"] >= 0.20).astype(float)
    v2_full_w, _ = winsorize(v2_full, WINSORIZED)
    v2_full_recode = v2_full_w.merge(soe[["key", "SOE_recode_s0702b"]], on="key", how="left")

    specs = []
    specs.append({"spec": "1_overlap_v1_controls_ims", **fit_core(overlap_reg, "Y_v1", "IMS", [c for c in ALL_CONTROLS_V1 if c != "IMS"])})
    specs.append({"spec": "1_overlap_v1_controls_imsd10", **fit_core(overlap_reg.assign(IMS_D=overlap_reg["IMS_D_10"]), "Y_v1", "IMS_D", ALL_CONTROLS_V1)})
    specs.append({"spec": "2_overlap_imsd20", **fit_core(overlap_reg.assign(IMS_D=overlap_reg["IMS_D_20"]), "Y_v1", "IMS_D", ALL_CONTROLS_V1)})
    specs.append({"spec": "3_overlap_firmage_winsor_ims", **fit_core(overlap_reg_w, "Y_v1", "IMS", ALL_CONTROLS_V2)})
    specs.append({"spec": "4_v2full_storedY_ims", **fit_core(v2_full_w, DEPENDENT, "IMS", ALL_CONTROLS_V2)})
    specs.append({"spec": "4_v2full_storedY_imsd20", **fit_core(v2_full_w.assign(IMS_D=v2_full_w["IMS_D_20"]), DEPENDENT, "IMS_D", ALL_CONTROLS_V2)})
    recode_controls = [c if c != "SOE" else "SOE_recode_s0702b" for c in ALL_CONTROLS_V2]
    specs.append({"spec": "5_v2full_recodeSOE_ims", **fit_core(v2_full_recode, DEPENDENT, "IMS", recode_controls)})
    pd.DataFrame(specs).to_csv(OUT / "nested_regression_diagnostics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "sample_flow": sample_flow.to_dict(orient="records"),
        "overlap_n": len(overlap),
        "only_v1_n": len(only_v1),
        "only_v2_n": len(only_v2),
        "v2_common_from_original": len(v2_from_orig),
        "v2_common_from_expansion": len(v2_from_exp),
        "v2_rebuilt_common_n": len(v2_rebuilt_common),
        "soe_expansion_flag": soe_expansion_flag,
        "soe_exclusive_codes": exclusive_soe,
        "overlap_y_v1_vs_v2_stored": overlap_y_stats,
        "v2_common_soe_stored": float(v2_common["SOE"].mean()),
        "v2_common_soe_recode": float(v2_common["SOE_recode_s0702b"].mean()),
        "v2_common_ims_d10": float(v2_common["IMS_D_10"].mean()),
        "v2_common_ims_d20": float(v2_common["IMS_D_20"].mean()),
    }
    (OUT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["overlap_n", "only_v1_n", "only_v2_n", "v2_common_from_original", "v2_common_from_expansion", "v2_rebuilt_common_n", "v2_common_soe_stored", "v2_common_soe_recode"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
