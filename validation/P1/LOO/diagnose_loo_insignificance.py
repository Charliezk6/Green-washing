from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_loo_analysis import (  # noqa: E402
    AUDITOR_ID, CONFIG, CONTROLS, DEP, FIRM, INDUSTRY, OUTPUT, REVENUE, YEAR,
    construct_loo, load_data, load_full_market,
)

OUT = OUTPUT / "diagnostics/insignificance"
FIG = OUTPUT / "figures"
GI_Z = "GI_z_行业年度"
ESGE_Z = "ESGE_z_行业年度"


def prepare() -> pd.DataFrame:
    market = load_full_market()
    data = construct_loo(load_data(), market)
    for column in [GI_Z, ESGE_Z]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    cell = market.groupby([AUDITOR_ID, INDUSTRY, YEAR])[REVENUE].agg(["sum", "count"])
    industry = market.groupby([INDUSTRY, YEAR])[REVENUE].sum()
    cell_keys = pd.MultiIndex.from_frame(data[[AUDITOR_ID, INDUSTRY, YEAR]])
    data["CELL_REVENUE"] = cell_keys.map(cell["sum"])
    data["CELL_CLIENTS"] = cell_keys.map(cell["count"])
    data["INDUSTRY_REVENUE"] = pd.MultiIndex.from_frame(data[[INDUSTRY, YEAR]]).map(industry)
    data["IMS_FULL"] = data["CELL_REVENUE"] / data["INDUSTRY_REVENUE"]
    revenue = data[REVENUE].where(data[REVENUE].gt(0))
    data["OWN"] = data["IMS"] - data["IMS_LOO"]
    data["SHARE"] = revenue / data["INDUSTRY_REVENUE"]
    data["CELL_SHARE"] = revenue / data["CELL_REVENUE"]
    data["SINGLE_CLIENT"] = data["CELL_CLIENTS"].eq(1).astype(float).where(data["IMS_LOO"].notna())
    data["Size_sq"] = data["Size"] ** 2
    market_auditor = market.set_index([market["证券代码"].astype(int).astype(str).str.zfill(6), market[YEAR].astype(int)])[AUDITOR_ID]
    firm_year = pd.MultiIndex.from_arrays([data[FIRM].astype(str).str.zfill(6), data[YEAR].astype(int)])
    data["AUDITOR_MARKET"] = firm_year.map(market_auditor)
    data["AUDITOR_MISMATCH"] = data["AUDITOR_MARKET"].notna() & data["AUDITOR_MARKET"].ne(data[AUDITOR_ID])
    market_cell = pd.MultiIndex.from_frame(data[["AUDITOR_MARKET", INDUSTRY, YEAR]]).map(cell["sum"])
    numerator = market_cell - revenue
    denominator = data["INDUSTRY_REVENUE"] - revenue
    data["IMS_LOO_MK"] = (numerator / denominator).where(numerator.ge(0) & denominator.gt(0))
    data["IMS_LOO_MK_D"] = data["IMS_LOO_MK"].ge(CONFIG["ims_threshold"]).astype(float).where(data["IMS_LOO_MK"].notna())
    data["IMS_MK"] = market_cell / data["INDUSTRY_REVENUE"]
    lost = data["IMS_D"].eq(1) & data["IMS_LOO_D"].eq(0)
    both = data["IMS_D"].eq(1) & data["IMS_LOO_D"].eq(1)
    data["D_LOST"] = lost.astype(float).where(data["IMS_LOO_D"].notna())
    data["D_BOTH"] = both.astype(float).where(data["IMS_LOO_D"].notna())
    return data


def formula(xs: list[str], extra: list[str] | None = None, dep: str = DEP) -> str:
    terms = [*xs, *(extra or []), *CONTROLS]
    return f"Q('{dep}') ~ " + " + ".join(f"Q('{t}')" for t in terms) + f" + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"


def design(sample: pd.DataFrame, xs: list[str], extra: list[str] | None = None, dep: str = DEP):
    y, x = patsy.dmatrices(formula(xs, extra, dep), sample, return_type="dataframe", NA_action="raise")
    return y.iloc[:, 0], x


def fit(sample: pd.DataFrame, xs: list[str], extra: list[str] | None = None, dep: str = DEP):
    y, x = design(sample, xs, extra, dep)
    return sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sample[FIRM])[0]})


def spec_rows(spec: str, description: str, sample: pd.DataFrame, xs: list[str], extra: list[str] | None = None, dep: str = DEP) -> list[dict]:
    needed = [dep, FIRM, YEAR, INDUSTRY, *xs, *(extra or []), *CONTROLS]
    sample = sample.dropna(subset=needed)
    result = fit(sample, xs, extra, dep)
    rows = []
    for variable in [*xs, *(extra or [])]:
        name = f"Q('{variable}')"
        rows.append({
            "spec": spec, "description": description, "dependent": dep, "variable": variable,
            "coefficient": result.params[name], "std_error": result.bse[name],
            "t_value": result.tvalues[name], "p_value": result.pvalues[name],
            "n": int(result.nobs), "firms": sample[FIRM].nunique(),
            "x_sd": sample[variable].std(),
        })
    return rows


def run_specs(data: pd.DataFrame, loo: pd.DataFrame) -> pd.DataFrame:
    share_p95 = loo["SHARE"].quantile(0.95)
    share_p99 = loo["SHARE"].quantile(0.99)
    no_single = loo[loo["SINGLE_CLIENT"].eq(0)]
    no_top5 = loo[loo["SHARE"].le(share_p95)]
    no_top1 = loo[loo["SHARE"].le(share_p99)]
    not_dominant = loo[loo["CELL_SHARE"].le(0.5)]
    mk_sample = data[data["IMS_LOO_MK"].notna()].assign(OWN_MK=lambda f: f["IMS"] - f["IMS_LOO_MK"])
    specs = [
        ("A1", "Baseline IMS, full 15,381 sample", data, ["IMS"], None),
        ("A2", "Baseline IMS_D, full 15,381 sample", data, ["IMS_D"], None),
        ("A3", "Baseline IMS, LOO-valid sample", loo, ["IMS"], None),
        ("A4", "Baseline IMS_D, LOO-valid sample", loo, ["IMS_D"], None),
        ("A5", "Reconstructed full-market IMS, LOO-valid sample", loo, ["IMS_FULL"], None),
        ("A6", "IMS_LOO", loo, ["IMS_LOO"], None),
        ("A7", "IMS_LOO_D", loo, ["IMS_LOO_D"], None),
        ("A8", "Baseline IMS, market-auditor LOO sample", mk_sample, ["IMS"], None),
        ("A9", "IMS_LOO_MK (market auditor key)", mk_sample, ["IMS_LOO_MK"], None),
        ("A10", "IMS_LOO_MK_D (market auditor key)", mk_sample, ["IMS_LOO_MK_D"], None),
        ("A11", "IMS_LOO_MK and OWN_MK jointly", mk_sample, ["IMS_LOO_MK", "OWN_MK"], None),
        ("A12", "IMS_LOO_MK controlling SHARE", mk_sample, ["IMS_LOO_MK"], ["SHARE"]),
        ("B1", "Own-firm component OWN = IMS - IMS_LOO alone", loo, ["OWN"], None),
        ("B2", "IMS_LOO and OWN jointly", loo, ["IMS_LOO", "OWN"], None),
        ("B3", "Focal firm industry revenue share SHARE alone", loo, ["SHARE"], None),
        ("B4", "IMS controlling SHARE", loo, ["IMS"], ["SHARE"]),
        ("B5", "IMS_LOO controlling SHARE", loo, ["IMS_LOO"], ["SHARE"]),
        ("B6", "IMS controlling Size squared", loo, ["IMS"], ["Size_sq"]),
        ("B7", "IMS_LOO with single-client indicator", loo, ["IMS_LOO"], ["SINGLE_CLIENT"]),
        ("B8", "SHARE controlling Size squared", loo, ["SHARE"], ["Size_sq"]),
        ("B9", "IMS_LOO controlling SHARE and Size squared", loo, ["IMS_LOO"], ["SHARE", "Size_sq"]),
        ("C1", "IMS, excluding single-client cells", no_single, ["IMS"], None),
        ("C2", "IMS_LOO, excluding single-client cells", no_single, ["IMS_LOO"], None),
        ("C3", "IMS, excluding top 5% SHARE", no_top5, ["IMS"], None),
        ("C4", "IMS_LOO, excluding top 5% SHARE", no_top5, ["IMS_LOO"], None),
        ("C5", "IMS, excluding top 1% SHARE", no_top1, ["IMS"], None),
        ("C6", "IMS_LOO, excluding top 1% SHARE", no_top1, ["IMS_LOO"], None),
        ("C7", "IMS, excluding focal firms with CELL_SHARE > 50%", not_dominant, ["IMS"], None),
        ("C8", "IMS_LOO, excluding focal firms with CELL_SHARE > 50%", not_dominant, ["IMS_LOO"], None),
        ("D1", "IMS_D transition groups (reference: IMS_D = 0)", loo, ["D_BOTH", "D_LOST"], None),
    ]
    for dep, label in [(GI_Z, "disclosure GI_z"), (ESGE_Z, "E performance ESGE_z")]:
        specs += [
            (f"E_{label}_1", f"IMS on {label}", loo, ["IMS"], None, dep),
            (f"E_{label}_2", f"IMS_LOO on {label}", loo, ["IMS_LOO"], None, dep),
            (f"E_{label}_3", f"IMS_LOO and OWN on {label}", loo, ["IMS_LOO", "OWN"], None, dep),
        ]
    rows = []
    for spec in specs:
        rows.extend(spec_rows(*spec))
    table = pd.DataFrame(rows)
    table["beta_x_sd"] = table["coefficient"] * table["x_sd"]
    return table


def difference_test(loo: pd.DataFrame, first: str, second: str) -> dict:
    sample = loo.dropna(subset=[DEP, first, second, *CONTROLS])
    y1, x1 = design(sample, [first])
    y2, x2 = design(sample, [second])
    k1, k2 = x1.shape[1], x2.shape[1]
    stacked_x = np.block([[x1.to_numpy(), np.zeros((len(x1), k2))], [np.zeros((len(x2), k1)), x2.to_numpy()]])
    stacked_y = np.r_[y1.to_numpy(), y2.to_numpy()]
    groups = np.tile(pd.factorize(sample[FIRM])[0], 2)
    result = sm.OLS(stacked_y, stacked_x).fit(cov_type="cluster", cov_kwds={"groups": groups})
    i, j = list(x1.columns).index(f"Q('{first}')"), k1 + list(x2.columns).index(f"Q('{second}')")
    contrast = np.zeros(k1 + k2)
    contrast[i], contrast[j] = 1, -1
    test = result.t_test(contrast)
    cov = result.cov_params()
    return {
        "first": first, "second": second, "beta_first": result.params[i], "beta_second": result.params[j],
        "difference": float(np.squeeze(test.effect)), "std_error": float(np.squeeze(test.sd)),
        "p_value": float(np.squeeze(test.pvalue)),
        "corr_of_estimates": cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]), "n": len(sample),
    }


def residualize(sample: pd.DataFrame, variable: str) -> np.ndarray:
    rhs = " + ".join(f"Q('{c}')" for c in CONTROLS) + f" + C(Q('{YEAR}')) + C(Q('{INDUSTRY}'))"
    y, x = patsy.dmatrices(f"Q('{variable}') ~ {rhs}", sample, return_type="dataframe")
    return (y.iloc[:, 0] - x @ np.linalg.lstsq(x.to_numpy(), y.to_numpy(), rcond=None)[0][:, 0]).to_numpy()


def covariance_decomposition(loo: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    sample = loo.dropna(subset=[DEP, "IMS", "IMS_LOO", *CONTROLS])
    res = {v: residualize(sample, v) for v in [DEP, "IMS", "IMS_LOO", "OWN", "SHARE"]}
    y = res[DEP]
    cov_ims = np.mean(y * res["IMS"])
    cov_loo = np.mean(y * res["IMS_LOO"])
    cov_own = np.mean(y * res["OWN"])
    single = sample["SINGLE_CLIENT"].eq(1).to_numpy()
    rows = [
        {"item": "var(res_IMS)", "value": np.var(res["IMS"])},
        {"item": "var(res_IMS_LOO)", "value": np.var(res["IMS_LOO"])},
        {"item": "var(res_OWN)", "value": np.var(res["OWN"])},
        {"item": "2cov(res_IMS_LOO,res_OWN)", "value": 2 * np.mean(res["IMS_LOO"] * res["OWN"])},
        {"item": "corr(res_IMS,res_IMS_LOO)", "value": np.corrcoef(res["IMS"], res["IMS_LOO"])[0, 1]},
        {"item": "corr(res_OWN,res_SHARE)", "value": np.corrcoef(res["OWN"], res["SHARE"])[0, 1]},
        {"item": "cov(res_y,res_IMS)", "value": cov_ims},
        {"item": "cov(res_y,res_IMS_LOO)", "value": cov_loo},
        {"item": "cov(res_y,res_OWN)", "value": cov_own},
        {"item": "share_of_cov(y,IMS)_from_OWN", "value": cov_own / cov_ims},
        {"item": "share_of_cov(y,IMS)_from_single_client_rows", "value": np.sum(y[single] * res["IMS"][single]) / np.sum(y * res["IMS"])},
        {"item": "share_of_var(res_OWN)_from_single_client_rows", "value": np.sum(res["OWN"][single] ** 2) / np.sum(res["OWN"] ** 2)},
        {"item": "beta_IMS_FWL", "value": cov_ims / np.var(res["IMS"])},
        {"item": "beta_IMS_LOO_FWL", "value": cov_loo / np.var(res["IMS_LOO"])},
    ]
    return pd.DataFrame(rows), res


def group_profiles(loo: pd.DataFrame) -> pd.DataFrame:
    frame = loo.dropna(subset=["IMS_LOO"]).copy()
    frame["group"] = np.where(frame["SINGLE_CLIENT"].eq(1), "single_client_cell", "multi_client_cell")
    frame["transition"] = np.select(
        [frame["D_BOTH"].eq(1), frame["D_LOST"].eq(1)],
        ["IMS_D=1,LOO_D=1", "IMS_D=1,LOO_D=0 (lost)"], default="IMS_D=0,LOO_D=0",
    )
    frame["share_bin"] = pd.qcut(frame["SHARE"], [0, 0.5, 0.9, 0.95, 0.99, 1.0], labels=["p0-50", "p50-90", "p90-95", "p95-99", "p99-100"])
    columns = {"n": (DEP, "size"), "GW_mean": (DEP, "mean"), "IMS_mean": ("IMS", "mean"), "IMS_LOO_mean": ("IMS_LOO", "mean"),
               "OWN_mean": ("OWN", "mean"), "SHARE_mean": ("SHARE", "mean"), "CELL_SHARE_mean": ("CELL_SHARE", "mean"),
               "Size_mean": ("Size", "mean"), "Big4_mean": ("Big4", "mean"), "SOE_mean": ("SOE", "mean")}
    parts = []
    for key in ["group", "transition", "share_bin"]:
        part = frame.groupby(key, observed=True).agg(**columns).reset_index().rename(columns={key: "level"})
        part.insert(0, "grouping", key)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def binscatter(loo: pd.DataFrame, res: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, variable in zip(axes, ["IMS", "IMS_LOO", "OWN"]):
        x = res[variable] + loo.dropna(subset=[DEP, "IMS", "IMS_LOO", *CONTROLS])[variable].mean()
        bins = pd.qcut(pd.Series(x).rank(method="first"), 20, labels=False)
        grouped = pd.DataFrame({"x": x, "y": res[DEP], "bin": bins}).groupby("bin").mean()
        slope = np.cov(res[DEP], res[variable])[0, 1] / np.var(res[variable], ddof=1)
        ax.scatter(grouped["x"], grouped["y"], s=22)
        grid = np.linspace(grouped["x"].min(), grouped["x"].max(), 50)
        ax.plot(grid, slope * (grid - x.mean()), color="firebrick", lw=1.2)
        ax.set_title(f"{variable}  (slope = {slope:.3f})")
        ax.set_xlabel(f"{variable} (residualized + mean)")
        for b, r in grouped.iterrows():
            rows.append({"variable": variable, "bin": int(b), "x_mean": r["x"], "residual_gw_mean": r["y"]})
    axes[0].set_ylabel("Greenwashing (residualized)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "binscatter_ims_vs_loo.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(rows)


def construction_check(data: pd.DataFrame) -> pd.DataFrame:
    valid = data["IMS_LOO"].notna()
    diff = (data["IMS"] - data["IMS_FULL"]).abs()
    own = data.loc[valid, "OWN"]
    mismatch = data["AUDITOR_MISMATCH"]
    return pd.DataFrame([
        {"item": "rows", "value": len(data)},
        {"item": "loo_valid_rows", "value": int(valid.sum())},
        {"item": "panel_vs_market_auditor_mismatch_rows", "value": int(mismatch.sum())},
        {"item": "mismatch_rows_IMS_equals_market_auditor_IMS_share", "value": float((data.loc[mismatch, "IMS"] - data.loc[mismatch, "IMS_MK"]).abs().le(1e-9).mean())},
        {"item": "mismatch_rows_GW_mean", "value": data.loc[mismatch, DEP].mean()},
        {"item": "other_rows_GW_mean", "value": data.loc[~mismatch, DEP].mean()},
        {"item": "mismatch_rows_IMS_mean", "value": data.loc[mismatch, "IMS"].mean()},
        {"item": "mismatch_rows_IMS_D_mean", "value": data.loc[mismatch, "IMS_D"].mean()},
        {"item": "loo_market_key_valid_rows", "value": int(data["IMS_LOO_MK"].notna().sum())},
        {"item": "IMS_LOO_MK_equals_IMS_LOO_share_(both_valid)", "value": float((data["IMS_LOO_MK"] - data["IMS_LOO"]).abs().le(1e-12)[valid].mean())},
        {"item": "IMS_equals_full_market_recompute_share_(valid_rows)", "value": float(diff[valid].le(1e-9).mean())},
        {"item": "IMS_equals_full_market_recompute_share_(invalid_rows)", "value": float(diff[~valid].le(1e-9).mean())},
        {"item": "single_client_cells_rows", "value": int(data["SINGLE_CLIENT"].eq(1).sum())},
        {"item": "IMS_LOO_exactly_zero_rows", "value": int(data["IMS_LOO"].eq(0).sum())},
        {"item": "OWN_mean", "value": own.mean()},
        {"item": "OWN_sd", "value": own.std()},
        {"item": "OWN_p99", "value": own.quantile(0.99)},
        {"item": "OWN_max", "value": own.max()},
        {"item": "OWN_negative_share", "value": float(own.lt(-1e-12).mean())},
        {"item": "corr(OWN,Size)", "value": data.loc[valid, ["OWN", "Size"]].corr().iloc[0, 1]},
        {"item": "corr(OWN,SHARE)", "value": data.loc[valid, ["OWN", "SHARE"]].corr().iloc[0, 1]},
        {"item": "corr(IMS,Size)", "value": data.loc[valid, ["IMS", "Size"]].corr().iloc[0, 1]},
        {"item": "corr(IMS_LOO,Size)", "value": data.loc[valid, ["IMS_LOO", "Size"]].corr().iloc[0, 1]},
        {"item": "SHARE_p95", "value": data.loc[valid, "SHARE"].quantile(0.95)},
        {"item": "SHARE_p99", "value": data.loc[valid, "SHARE"].quantile(0.99)},
        {"item": "CELL_SHARE_gt_0.5_rows", "value": int(data.loc[valid, "CELL_SHARE"].gt(0.5).sum())},
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = prepare()
    loo = data[data["IMS_LOO"].notna()].copy()
    construction_check(data).to_csv(OUT / "construction_check.csv", index=False, encoding="utf-8-sig")
    run_specs(data, loo).to_csv(OUT / "diagnostic_regressions.csv", index=False, encoding="utf-8-sig")
    mk_sample = data[data["IMS_LOO_MK"].notna()]
    pd.DataFrame([
        difference_test(loo, "IMS", "IMS_LOO"), difference_test(loo, "IMS_D", "IMS_LOO_D"),
        difference_test(mk_sample, "IMS", "IMS_LOO_MK"), difference_test(mk_sample, "IMS_D", "IMS_LOO_MK_D"),
    ]).to_csv(
        OUT / "coefficient_difference_test.csv", index=False, encoding="utf-8-sig")
    decomposition, res = covariance_decomposition(loo)
    decomposition.to_csv(OUT / "fwl_covariance_decomposition.csv", index=False, encoding="utf-8-sig")
    group_profiles(loo).to_csv(OUT / "group_profiles.csv", index=False, encoding="utf-8-sig")
    binscatter(loo, res).to_csv(OUT / "binscatter_points.csv", index=False, encoding="utf-8-sig")
    pd.crosstab(loo["IMS_D"], loo["IMS_LOO_D"], margins=True).to_csv(OUT / "binary_transition_crosstab.csv", encoding="utf-8-sig")
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
