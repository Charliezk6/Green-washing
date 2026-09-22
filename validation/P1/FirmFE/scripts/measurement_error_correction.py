"""Measurement-error diagnostics for within-firm IMS.

Builds two alternative market-share measures of the same audit-firm specialization
concept (audit-fee share and square-root-asset share), estimates the within-firm
reliability ratio, and reports both the classical errors-in-variables correction
and instrumental-variable estimates that use the alternative measures as
instruments for one another.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SCRIPTS))

from run_firm_fe_regression import (  # noqa: E402
    ALL_CONTROLS,
    DEPENDENT,
    fit,
    load_data,
    significance,
    winsorize_controls,
)

TABLES = ROOT / "outputs" / "tables"
DIAG = ROOT / "outputs" / "diagnostics"
FIRM = "证券代码"
YEAR = "时间"
INDUSTRY = "ExpertiseIndustryCode"
MARKET = PROJECT / "X_reproduction" / "data" / "input" / "full_a_share_audit_market_2021_2024.csv"
FEE_CANDIDATES = list(PROJECT.glob("appendix/**/AIQ_FinAuditOpinY.csv"))


def expertise_code(industry_code: object) -> str:
    code = str(industry_code).strip()
    return code if code.startswith("C") else code[:1]


def normalize_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def read_fees() -> pd.DataFrame:
    if not FEE_CANDIDATES:
        raise FileNotFoundError("AIQ_FinAuditOpinY.csv was not found under appendix/.")
    path = FEE_CANDIDATES[0]
    last_error: Exception | None = None
    frame = None
    for encoding in ("utf-8-sig", "gbk", "utf-8"):
        try:
            frame = pd.read_csv(path, dtype={"Symbol": str}, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    if frame is None:
        raise RuntimeError(f"Could not read {path}") from last_error
    frame[FIRM] = frame["Symbol"].str.zfill(6)
    frame[YEAR] = pd.to_numeric(frame["EndDate"].astype(str).str[:4], errors="coerce")
    frame["TotalAuditFee"] = pd.to_numeric(frame["TotalAuditFee"], errors="coerce")
    frame = frame.dropna(subset=[FIRM, YEAR]).sort_values("TotalAuditFee", ascending=False)
    frame = frame.drop_duplicates([FIRM, YEAR], keep="first")
    return frame[[FIRM, YEAR, "TotalAuditFee"]]


def alternative_shares() -> pd.DataFrame:
    market = pd.read_csv(
        MARKET,
        dtype={"InstitutionID": str, FIRM: str},
        encoding="utf-8-sig",
        low_memory=False,
    )
    market[YEAR] = pd.to_numeric(market[YEAR], errors="coerce")
    market["OperatingRevenue"] = pd.to_numeric(market["OperatingRevenue"], errors="coerce")
    market["TotalAssets"] = pd.to_numeric(market["TotalAssets"], errors="coerce")
    market["InstitutionID"] = market["InstitutionID"].map(normalize_id)
    market[FIRM] = market[FIRM].astype(str).str.zfill(6)
    market[INDUSTRY] = market["IndustryCode"].map(expertise_code)
    market = market[market[YEAR].isin([2021, 2022, 2023, 2024])]
    market = market[market[INDUSTRY].ne("O")].dropna(subset=["InstitutionID", INDUSTRY, YEAR])
    market = market.merge(read_fees(), on=[FIRM, YEAR], how="left")

    rows = []
    for scale, column in (("IMS_fee", "TotalAuditFee"), ("IMS_asset", "AssetRoot")):
        piece = market.copy()
        if column == "AssetRoot":
            piece[column] = np.sqrt(piece["TotalAssets"].where(piece["TotalAssets"] > 0))
        else:
            piece[column] = piece[column].where(piece[column] > 0)
        piece = piece.dropna(subset=[column])
        numerator = piece.groupby(["InstitutionID", INDUSTRY, YEAR])[column].sum()
        denominator = piece.groupby([INDUSTRY, YEAR])[column].sum()
        share = (numerator / denominator).rename(scale).reset_index()
        rows.append(share)
    shares = rows[0].merge(rows[1], on=["InstitutionID", INDUSTRY, YEAR], how="outer")
    return shares


def prepare() -> pd.DataFrame:
    data = load_data()
    required = [DEPENDENT, "IMS", *ALL_CONTROLS, FIRM, YEAR, INDUSTRY, "InstitutionID"]
    common = data.dropna(subset=required).copy()
    common, _ = winsorize_controls(common)
    common["InstitutionID"] = common["InstitutionID"].map(normalize_id)
    shares = alternative_shares()
    merged = common.merge(shares, on=["InstitutionID", INDUSTRY, YEAR], how="left")
    return merged.sort_values([FIRM, YEAR]).reset_index(drop=True)


def within_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = data[columns + [FIRM]].copy()
    for column in columns:
        out[column] = out[column] - out.groupby(FIRM)[column].transform("mean")
    return out


def reliability_rows(sample: pd.DataFrame) -> list[dict]:
    rows = []
    pairs = [("IMS", "IMS_fee"), ("IMS", "IMS_asset"), ("IMS_fee", "IMS_asset")]
    for scope, frame in (
        ("总体", sample),
        ("组内", within_frame(sample, ["IMS", "IMS_fee", "IMS_asset"])),
    ):
        for left, right in pairs:
            pair = frame[[left, right]].dropna()
            covariance = float(pair[left].cov(pair[right]))
            variance = float(pair[left].var(ddof=1))
            reliability = covariance / variance if variance > 0 else np.nan
            rows.append({
                "记录类型": "信度比",
                "范围": scope,
                "被度量变量": left,
                "参照度量": right,
                "协方差": covariance,
                "被度量变量方差": variance,
                "相关系数": float(pair[left].corr(pair[right])),
                "信度比lambda": reliability,
                "观测数": int(len(pair)),
                "公司数": int(sample[FIRM].nunique()) if scope == "总体" else int(frame[FIRM].nunique()),
            })
    return rows


def corrected_row(beta: float, se: float, reliability: float, reference: str, n_obs: int, n_firms: int) -> dict:
    usable = np.isfinite(reliability) and 0 < reliability <= 1
    corrected = beta / reliability if usable else np.nan
    corrected_se = se / reliability if usable else np.nan
    return {
        "记录类型": "EIV修正",
        "范围": "组内",
        "被度量变量": "IMS",
        "参照度量": reference,
        "FirmFE系数": beta,
        "FirmFE标准误": se,
        "信度比lambda": reliability,
        "修正后系数": corrected,
        "修正后标准误": corrected_se,
        "修正后p值": np.nan,
        "可用": usable,
        "样本数": n_obs,
        "公司数": n_firms,
        "说明": "古典EIV修正beta/lambda；lambda超出(0,1]时不修正" if not usable else "古典EIV修正beta/lambda，标准误按同一比例放大",
    }


def design_matrices(sample: pd.DataFrame, endog: str, instrument: str):
    within_cols = [DEPENDENT, endog, instrument, *ALL_CONTROLS]
    used = sample.dropna(subset=within_cols + [FIRM, YEAR]).copy()
    for column in within_cols:
        used[column] = used[column] - used.groupby(FIRM)[column].transform("mean")
    year = pd.get_dummies(used[YEAR].astype(int), prefix="year", drop_first=True, dtype=float)
    exog = pd.concat([used[ALL_CONTROLS].reset_index(drop=True), year.reset_index(drop=True)], axis=1)
    exog.insert(0, "const", 1.0)
    dependent = used[DEPENDENT].reset_index(drop=True)
    endogenous = used[[endog]].reset_index(drop=True)
    instruments = used[[instrument]].reset_index(drop=True)
    clusters = used[FIRM].reset_index(drop=True)
    return dependent, exog, endogenous, instruments, clusters, used


def first_stage_f(used: pd.DataFrame, endog: str, instrument: str) -> float:
    terms = [instrument, *ALL_CONTROLS, f"C({YEAR})"]
    model = smf.ols(f"Q('{endog}') ~ " + " + ".join(terms), data=used)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": used[FIRM]})
    return float(result.tvalues[instrument] ** 2)


def iv_row(sample: pd.DataFrame, endog: str, instrument: str) -> dict:
    dependent, exog, endogenous, instruments, clusters, used = design_matrices(sample, endog, instrument)
    result = IV2SLS(dependent, exog, endogenous, instruments).fit(cov_type="clustered", clusters=clusters)
    coef = float(result.params[endog])
    se = float(result.std_errors[endog])
    p_value = float(result.pvalues[endog])
    f_stat = first_stage_f(used, endog, instrument)
    return {
        "记录类型": "IV",
        "范围": "组内",
        "内生变量": endog,
        "工具变量": instrument,
        "系数": coef,
        "标准误": se,
        "t值": float(result.tstats[endog]),
        "p值": p_value,
        "显著性": significance(p_value),
        "一阶段F": f_stat,
        "弱工具_F小于10": bool(f_stat < 10),
        "样本数": int(result.nobs),
        "公司数": int(clusters.nunique()),
        "说明": "企业内去均值后，以另一份额口径作为工具变量；公司聚类",
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    data = prepare()
    sample = data.dropna(subset=["IMS", "IMS_fee", "IMS_asset", DEPENDENT, *ALL_CONTROLS]).copy()
    print(
        f"common_with_alt={len(sample)} firms={sample[FIRM].nunique()} "
        f"fee_missing={data['IMS_fee'].isna().mean():.3f} asset_missing={data['IMS_asset'].isna().mean():.3f}"
    )
    firm_result, firm_sample = fit(sample, "IMS", ALL_CONTROLS)
    beta = float(firm_result.params["IMS"])
    se = float(firm_result.bse["IMS"])
    within = within_frame(sample, ["IMS", "IMS_fee", "IMS_asset"])
    rows: list[dict] = []
    rows.extend(reliability_rows(sample))
    for reference in ("IMS_fee", "IMS_asset"):
        pair = within[["IMS", reference]].dropna()
        reliability = float(pair["IMS"].cov(pair[reference]) / pair["IMS"].var(ddof=1))
        rows.append(corrected_row(beta, se, reliability, reference, int(firm_result.nobs), int(firm_sample[FIRM].nunique())))
    for endog, instrument in (("IMS", "IMS_fee"), ("IMS", "IMS_asset"), ("IMS_fee", "IMS"), ("IMS_asset", "IMS")):
        rows.append(iv_row(sample, endog, instrument))
    output = pd.DataFrame(rows)
    output.to_csv(TABLES / "measurement_error_correction.csv", index=False, encoding="utf-8-sig")
    print(output[["记录类型", "范围"]].head(20).to_string(index=False))
    iv = output[output["记录类型"] == "IV"][["内生变量", "工具变量", "系数", "标准误", "p值", "一阶段F"]]
    print(iv.to_string(index=False))


if __name__ == "__main__":
    main()
