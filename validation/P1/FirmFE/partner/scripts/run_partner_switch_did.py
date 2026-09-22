"""CPA-switch difference-in-differences with firm and year fixed effects.

Treated firms replace the signing team and move from a non-specialist partner to a
specialist partner. Control firms also replace the signing team, but the incoming
team is not a specialist. The specialist definition is run twice, once for the
industry-year leader and once for the pre-specified 90th-percentile cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
FIRMFE = Path(__file__).resolve().parents[2]
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
MEASURES = ["PIMS_D_leader", "PIMS_D_p90"]


def partner_key(value: object) -> str | None:
    if pd.isna(value):
        return None
    names = []
    for part in str(value).replace("，", ",").replace("、", ",").split(","):
        name = part.strip()
        if name:
            names.append(name)
    return "|".join(sorted(names)) if names else None


def prepare() -> pd.DataFrame:
    data = load_data()
    partner = pd.read_csv(ROOT / "data" / "partner_ims_company_year.csv", dtype={FIRM: str}, encoding="utf-8-sig")
    partner[FIRM] = partner[FIRM].astype(str).str.zfill(6)
    partner[YEAR] = pd.to_numeric(partner[YEAR], errors="coerce")
    data[FIRM] = data[FIRM].astype(str).str.zfill(6)
    merged = data.merge(partner[[FIRM, YEAR, *MEASURES]], on=[FIRM, YEAR], how="left")
    required = [DEPENDENT, *MEASURES, *ALL_CONTROLS, FIRM, YEAR, "Auditor"]
    sample = merged.dropna(subset=required).copy()
    sample, _ = winsorize_controls(sample)
    sample["partner_key"] = sample["Auditor"].map(partner_key)
    sample = sample.sort_values([FIRM, YEAR])
    sample["prev_key"] = sample.groupby(FIRM)["partner_key"].shift(1)
    sample["prev_year"] = sample.groupby(FIRM)[YEAR].shift(1)
    for measure in MEASURES:
        sample[f"prev_{measure}"] = sample.groupby(FIRM)[measure].shift(1)
    consecutive = sample[YEAR].sub(sample["prev_year"]).eq(1)
    sample["cpa_change"] = consecutive & sample["prev_key"].notna() & sample["partner_key"].ne(sample["prev_key"])
    return sample.reset_index(drop=True)


def classify(sample: pd.DataFrame, measure: str) -> pd.DataFrame:
    changes = sample[sample["cpa_change"]].copy()
    previous = changes[f"prev_{measure}"]
    current = changes[measure]
    changes["upgrade"] = previous.eq(0) & current.eq(1)
    changes["into_nonspecialist"] = current.eq(0)
    eligible = changes[changes["upgrade"] | changes["into_nonspecialist"]].copy()
    first = eligible.sort_values([FIRM, YEAR]).drop_duplicates(FIRM, keep="first")
    events = first[[FIRM, YEAR, "upgrade"]].rename(columns={YEAR: "event_year", "upgrade": "treat"})
    events["treat"] = events["treat"].astype(float)
    panel = sample.merge(events, on=FIRM, how="inner")
    panel["post"] = panel[YEAR].ge(panel["event_year"]).astype(float)
    panel["treat_post"] = panel["treat"] * panel["post"]
    panel["rel"] = panel[YEAR] - panel["event_year"]
    return panel


def fit_firm_year(sample: pd.DataFrame, regressors: list[str], controls: list[str]) -> tuple[object, pd.DataFrame]:
    used = sample.dropna(subset=[DEPENDENT, *regressors, *controls, FIRM, YEAR]).copy()
    for column in [DEPENDENT, *regressors, *controls]:
        used[column] = used[column] - used.groupby(FIRM)[column].transform("mean")
    terms = [*regressors, *controls, f"C({YEAR})"]
    model = smf.ols(f"Q('{DEPENDENT}') ~ " + " + ".join(terms), data=used)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": used[FIRM]})
    return result, used


def row(result, sample: pd.DataFrame, measure: str, spec: str, variable: str) -> dict:
    if variable not in result.params.index:
        return {}
    return {
        "专长定义": measure,
        "规格": spec,
        "变量": variable,
        "系数": float(result.params[variable]),
        "标准误": float(result.bse[variable]),
        "t值": float(result.tvalues[variable]),
        "p值": float(result.pvalues[variable]),
        "显著性": significance(float(result.pvalues[variable])),
        "样本数": int(result.nobs),
        "公司数": int(sample[FIRM].nunique()),
        "处理公司数": int(sample.loc[sample["treat"].eq(1), FIRM].nunique()),
        "对照公司数": int(sample.loc[sample["treat"].eq(0), FIRM].nunique()),
    }


def sun_abraham(panel: pd.DataFrame, measure: str) -> list[dict]:
    """Cohort-specific firm-year estimates, weighted by treated cohort size."""
    estimates = []
    weights = []
    rows = []
    controls = panel[panel["treat"].eq(0)]
    for cohort, treated in panel[panel["treat"].eq(1)].groupby("event_year"):
        piece = pd.concat([treated, controls], ignore_index=True)
        piece["event_time_0"] = ((piece["rel"].eq(0)) & piece["treat"].eq(1)).astype(float)
        result, used = fit_firm_year(piece, ["event_time_0"], [])
        if "event_time_0" not in result.params.index:
            continue
        estimate = float(result.params["event_time_0"])
        weight = int(treated[FIRM].nunique())
        estimates.append(estimate)
        weights.append(weight)
        rows.append(row(result, used, measure, f"Sun-Abraham队列{int(cohort)}", "event_time_0"))
    if weights:
        weighted = float(np.average(estimates, weights=weights))
        rows.append({
            "专长定义": measure,
            "规格": "Sun-Abraham加权平均_事件当期",
            "变量": "event_time_0",
            "系数": weighted,
            "标准误": np.nan,
            "t值": np.nan,
            "p值": np.nan,
            "显著性": "",
            "样本数": int(len(panel)),
            "公司数": int(panel[FIRM].nunique()),
            "处理公司数": int(sum(weights)),
            "对照公司数": int(controls[FIRM].nunique()),
            "说明": "各队列事件当期系数按处理公司数加权；标准误未做队列间协方差调整",
        })
    return rows


def main() -> None:
    out = ROOT / "outputs" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    sample = prepare()
    rows = []
    counts = []
    for measure in MEASURES:
        panel = classify(sample, measure)
        counts.append({
            "专长定义": measure,
            "换签字人公司年": int(sample["cpa_change"].sum()),
            "进入回归的公司数": int(panel[FIRM].nunique()),
            "处理公司数": int(panel.loc[panel["treat"].eq(1), FIRM].nunique()),
            "对照公司数": int(panel.loc[panel["treat"].eq(0), FIRM].nunique()),
        })
        for controls, label in (([], "企业+年份FE"), (ALL_CONTROLS, "企业+年份FE+控制变量")):
            result, used = fit_firm_year(panel, ["treat_post"], controls)
            rows.append(row(result, used, measure, label, "treat_post"))
        event_terms = []
        for lag in (-2, 0, 1, 2):
            name = f"rel_{lag}".replace("-", "m")
            panel[name] = (panel["rel"].eq(lag) & panel["treat"].eq(1)).astype(float)
            event_terms.append(name)
        result, used = fit_firm_year(panel, event_terms, [])
        for term in event_terms:
            rows.append(row(result, used, measure, "事件研究_省略-1期", term))
        rows.extend(sun_abraham(panel, measure))
    table = pd.DataFrame(rows)
    summary = pd.DataFrame(counts)
    table.to_csv(out / "partner_switch_did.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(ROOT / "outputs" / "diagnostics" / "partner_switch_counts.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(table[["专长定义", "规格", "变量", "系数", "标准误", "p值", "处理公司数"]].to_string(index=False))


if __name__ == "__main__":
    main()
