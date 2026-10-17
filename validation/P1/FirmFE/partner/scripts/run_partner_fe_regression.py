"""Partner-level specialization regressions and within-firm diagnostics.

The main comparison is four columns: industry-year FE with firm IMS, industry-year
FE with partner IMS, firm-year FE with partner IMS, and firm-year FE with both.
Alternative aggregations, binary rules, auditor clustering, and a restricted wild
cluster bootstrap are reported alongside, not selected after the fact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
FIRMFE = Path(__file__).resolve().parents[2]
PROJECT_TABLES = FIRMFE / "outputs" / "tables"
sys.path.insert(0, str(FIRMFE / "scripts"))

from run_firm_fe_regression import (  # noqa: E402
    ALL_CONTROLS,
    DEPENDENT,
    load_data,
    significance,
    winsorize_controls,
)

FIRM = "证券代码"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"
MEASURES = ["PIMS_max", "PIMS_mean", "PIMS_LOO_max", "PIMS_LOO_mean", "PIMS_D_leader", "PIMS_D_p90"]
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
BOOTSTRAP_REPS = 999
BOOTSTRAP_SEED = 20240922


def prepare() -> pd.DataFrame:
    data = load_data()
    partner = pd.read_csv(ROOT / "data" / "partner_ims_company_year.csv", dtype={FIRM: str}, encoding="utf-8-sig")
    partner[FIRM] = partner[FIRM].astype(str).str.zfill(6)
    partner[YEAR] = pd.to_numeric(partner[YEAR], errors="coerce")
    data[FIRM] = data[FIRM].astype(str).str.zfill(6)
    merged = data.merge(partner[[FIRM, YEAR, *MEASURES]], on=[FIRM, YEAR], how="left")
    required = [DEPENDENT, "IMS", "IMS_D", *MEASURES, *ALL_CONTROLS, FIRM, YEAR, INDUSTRY, "InstitutionID"]
    sample = merged.dropna(subset=required).copy()
    sample, _ = winsorize_controls(sample)
    sample["InstitutionID"] = sample["InstitutionID"].map(
        lambda v: str(int(float(v))) if pd.notna(v) and str(v).strip() else np.nan
    )
    return sample.sort_values([FIRM, YEAR]).reset_index(drop=True), merged


def fit_model(data: pd.DataFrame, cores: list[str], controls: list[str], firm_fe: bool, cluster: str,
              two_way: bool = False):
    sample = data.dropna(subset=[DEPENDENT, *cores, *controls, FIRM, YEAR, INDUSTRY, cluster]).copy()
    if firm_fe:
        year_columns = []
        if two_way:
            years = sorted(int(year) for year in sample[YEAR].dropna().unique())
            for year in years[1:]:
                name = f"year_{year}"
                sample[name] = (sample[YEAR].astype(int) == year).astype(float)
                year_columns.append(name)
        for column in [DEPENDENT, *cores, *controls, *year_columns]:
            sample[column] = sample[column] - sample.groupby(FIRM)[column].transform("mean")
        terms = [*cores, *controls, *(year_columns if two_way else [f"C({YEAR})"])]
    else:
        terms = [*cores, *controls, f"C({YEAR})", f"C({INDUSTRY})"]
    model = smf.ols(f"Q('{DEPENDENT}') ~ " + " + ".join(terms), data=sample)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": sample[cluster]})
    return result, sample


def coefficient_row(result, sample: pd.DataFrame, spec: str, variable: str, firm_fe: bool, cluster: str) -> dict:
    ci = result.conf_int().loc[variable]
    return {
        "规格": spec,
        "变量": variable,
        "系数": float(result.params[variable]),
        "标准误": float(result.bse[variable]),
        "t值": float(result.tvalues[variable]),
        "p值": float(result.pvalues[variable]),
        "显著性": significance(float(result.pvalues[variable])),
        "置信区间下限": float(ci[0]),
        "置信区间上限": float(ci[1]),
        "样本数": int(result.nobs),
        "公司数": int(sample[FIRM].nunique()),
        "聚类数": int(sample[cluster].nunique()),
        "企业固定效应": "是" if firm_fe else "否",
        "行业固定效应": "否" if firm_fe else "是",
        "年度固定效应": "是",
        "聚类": "事务所" if cluster == "InstitutionID" else "公司",
    }


def within_variation(sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable in ["IMS", "IMS_D", *MEASURES]:
        within = sample[variable] - sample.groupby(FIRM)[variable].transform("mean")
        movers = within.groupby(sample[FIRM]).apply(lambda s: float(np.nanmax(np.abs(s))) > 1e-12)
        switches = sample.groupby(FIRM)[variable].nunique().gt(1)
        rows.append({
            "变量": variable,
            "组内标准差": float(within.std(ddof=1)),
            "总体标准差": float(sample[variable].std(ddof=1)),
            "组内有变动公司数": int(movers.sum()),
            "组内有变动公司占比": float(movers.mean()),
            "取值发生变化公司数": int(switches.sum()),
            "取值发生变化公司占比": float(switches.mean()),
        })
    return pd.DataFrame(rows)


def wild_cluster_bootstrap(sample: pd.DataFrame, core: str) -> dict:
    used = sample.dropna(subset=[DEPENDENT, core, *ALL_CONTROLS, FIRM, "InstitutionID"]).copy()
    for column in [DEPENDENT, core, *ALL_CONTROLS]:
        used[column] = used[column] - used.groupby(FIRM)[column].transform("mean")
    year = pd.get_dummies(used[YEAR].astype(int), prefix="year", drop_first=True, dtype=float)
    controls = np.column_stack([used[ALL_CONTROLS].to_numpy(dtype=float), year.to_numpy(dtype=float)])
    x = used[core].to_numpy(dtype=float)
    y = used[DEPENDENT].to_numpy(dtype=float)
    design = np.column_stack([x, controls])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    beta_restricted = np.linalg.lstsq(controls, y, rcond=None)[0]
    fitted_restricted = controls @ beta_restricted
    residual_restricted = y - fitted_restricted
    residual_full = y - design @ beta
    clusters = used["InstitutionID"].astype(str).to_numpy()
    unique = pd.unique(clusters)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    restricted_draws = np.empty(BOOTSTRAP_REPS)
    unrestricted_draws = np.empty(BOOTSTRAP_REPS)
    for draw in range(BOOTSTRAP_REPS):
        weights = rng.choice(WEBB, size=len(unique))
        weight_map = dict(zip(unique, weights))
        cluster_weight = np.fromiter((weight_map[item] for item in clusters), dtype=float, count=len(clusters))
        y_restricted = fitted_restricted + residual_restricted * cluster_weight
        y_unrestricted = design @ beta + residual_full * cluster_weight
        restricted_draws[draw] = np.linalg.lstsq(design, y_restricted, rcond=None)[0][0]
        unrestricted_draws[draw] = np.linalg.lstsq(design, y_unrestricted, rcond=None)[0][0]
    estimate = float(beta[0])
    p_value = (1 + np.sum(np.abs(restricted_draws) >= abs(estimate) - 1e-10)) / (BOOTSTRAP_REPS + 1)
    return {
        "规格": "企业+年份FE",
        "变量": core,
        "系数": estimate,
        "受限wild_cluster_bootstrap_p": float(p_value),
        "无约束bootstrap_95CI下限": float(np.quantile(unrestricted_draws, 0.025)),
        "无约束bootstrap_95CI上限": float(np.quantile(unrestricted_draws, 0.975)),
        "bootstrap次数": BOOTSTRAP_REPS,
        "权重": "Webb六点",
        "原假设": "系数为0",
        "聚类": "事务所",
        "聚类数": int(len(unique)),
        "样本数": int(len(used)),
    }


def main() -> None:
    out = ROOT / "outputs"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "diagnostics").mkdir(parents=True, exist_ok=True)
    PROJECT_TABLES.mkdir(parents=True, exist_ok=True)
    sample, merged = prepare()
    print(f"panel={len(merged)} complete={len(sample)} firms={sample[FIRM].nunique()}")

    rows = []
    specs = [
        ("(1) 行业+年份FE，IMS", ["IMS"], False),
        ("(2) 行业+年份FE，PIMS_max", ["PIMS_max"], False),
        ("(3) 企业+年份FE，PIMS_max", ["PIMS_max"], True),
        ("(4) 企业+年份FE，PIMS_max与IMS", ["PIMS_max", "IMS"], True),
    ]
    for name, cores, firm_fe in specs:
        result, used = fit_model(sample, cores, ALL_CONTROLS, firm_fe, FIRM)
        for core in cores:
            rows.append(coefficient_row(result, used, name, core, firm_fe, FIRM))

    extra = []
    for measure in MEASURES:
        for firm_fe, label in ((False, "行业+年份FE"), (True, "企业+年份FE")):
            result, used = fit_model(sample, [measure], ALL_CONTROLS, firm_fe, FIRM)
            extra.append(coefficient_row(result, used, label, measure, firm_fe, FIRM))
            if firm_fe:
                auditor_result, auditor_sample = fit_model(sample, [measure], ALL_CONTROLS, True, "InstitutionID")
                extra.append(coefficient_row(
                    auditor_result, auditor_sample, "企业+年份FE", measure, True, "InstitutionID"
                ))
        horse_core = "IMS_D" if measure.startswith("PIMS_D") else "IMS"
        result, used = fit_model(sample, [measure, horse_core], ALL_CONTROLS, True, FIRM)
        for core in (measure, horse_core):
            extra.append(coefficient_row(
                result, used, f"企业+年份FE，同时放入{horse_core}", core, True, FIRM
            ))

    main_table = pd.DataFrame(rows)
    extra_table = pd.DataFrame(extra)
    variation = within_variation(sample)
    bootstrap_rows = [
        wild_cluster_bootstrap(sample, core)
        for core in ("PIMS_max", "PIMS_LOO_max", "PIMS_D_leader", "PIMS_D_p90")
    ]
    bootstrap = pd.DataFrame(bootstrap_rows)

    main_table.to_csv(out / "tables" / "partner_vs_firm_specialization.csv", index=False, encoding="utf-8-sig")
    main_table.to_csv(PROJECT_TABLES / "partner_vs_firm_specialization.csv", index=False, encoding="utf-8-sig")
    extra_table.to_csv(out / "tables" / "partner_fe_all_measures.csv", index=False, encoding="utf-8-sig")
    variation.to_csv(out / "diagnostics" / "partner_within_variation.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(out / "tables" / "partner_wild_cluster_bootstrap.csv", index=False, encoding="utf-8-sig")
    print(main_table[["规格", "变量", "系数", "标准误", "p值", "显著性", "样本数"]].to_string(index=False))
    print(variation.to_string(index=False))
    print(bootstrap[["变量", "系数", "受限wild_cluster_bootstrap_p"]].to_string(index=False))


def write_twoway_sensitivity() -> None:
    """Two-way within estimates for the firm-FE columns already in the memo.

    Does not overwrite partner_fe_all_measures.csv. The published firm-FE
    columns remain the estimator that demeans covariates and then adds year
    dummies in levels.
    """
    sample, _ = prepare()
    specs = [
        ("企业+年份，双向组内", ["PIMS_max"]),
        ("企业+年份，双向组内，同时放入IMS", ["PIMS_max", "IMS"]),
        ("企业+年份，双向组内", ["PIMS_D_p90"]),
        ("企业+年份，双向组内，同时放入IMS_D", ["PIMS_D_p90", "IMS_D"]),
        ("企业+年份，双向组内", ["PIMS_D_leader"]),
    ]
    rows = []
    for name, cores in specs:
        result, used = fit_model(sample, cores, ALL_CONTROLS, True, FIRM, two_way=True)
        for core in cores:
            row = coefficient_row(result, used, name, core, True, FIRM)
            row["组内变换"] = "因变量、协变量和年份哑变量一并公司内去均值"
            rows.append(row)
    table = pd.DataFrame(rows)
    destination = ROOT / "outputs" / "tables" / "partner_fe_twoway_sensitivity.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False, encoding="utf-8-sig")
    print(table[["规格", "变量", "系数", "标准误", "p值", "显著性"]].to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--twoway-sensitivity":
        write_twoway_sensitivity()
    else:
        main()
