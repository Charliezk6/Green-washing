"""Shared paths, industry-year z-scores, and the full baseline specification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PACK = Path(__file__).resolve().parents[1]
ROOT = PACK.parent
COMMON_PATH = ROOT / "Regression/outputs/paper_tables/common_sample_15381.csv"
LAYER_B_PATHS = [
    ROOT / "Y_variable_reproduction/data/input/manufacturing/layer_b_results.jsonl",
    ROOT / "Y_variable_reproduction/data/input/non_manufacturing/layer_b_results.jsonl",
    ROOT / "Y_variable_reproduction/data/input/expansion/layer_b_results.jsonl",
]
PENALTY_PATH = ROOT / "appendix/上市公司环保处罚总表.xlsx"
DERIVED = PACK / "results/derived"
TABLES = PACK / "results/tables"
TEX = PACK / "tex"
FIRM_YEAR_COUNTS = DERIVED / "layer_b_substantive_flags.csv"

CONTROLS = [
    "Size", "Lev", "ROA", "CFO", "Growth", "FirmAge",
    "SOE", "Dual", "BoardSize", "IndependentDirectorRatio",
    "Big4", "LnAuditFee", "AuditorTenure", "IsChangeFirm",
]
ADVERSE_TERMS = (
    "处罚", "罚款", "罚金", "超标", "违规", "违法", "责令", "立案", "通报",
    "行政处罚", "停产", "限产", "未按时", "未完成", "逾期",
)


def ensure_dirs() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)


def pad_code(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    try:
        return str(int(float(text))).zfill(6)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(6) if digits else text.zfill(6)


def year_from_enddate(value: object) -> float:
    text = str(value).strip()
    if not text or text == "nan":
        return np.nan
    if len(text) >= 4 and text[:4].isdigit():
        return float(text[:4])
    try:
        return float(int(float(text)))
    except (TypeError, ValueError):
        return np.nan


NEGATION_MARKERS = ("没有", "不存在", "未发生", "未出现", "未受到", "并未", "并不", "未", "无")


def adverse_hit(sentence: object, drop_negation: bool) -> bool:
    text = "" if sentence is None else str(sentence)
    for term in ADVERSE_TERMS:
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            window = text[max(0, index - 8):index]
            negated = window.endswith("不") or any(marker in window for marker in NEGATION_MARKERS)
            if not drop_negation or not negated:
                return True
            start = index + len(term)
    return False


def is_adverse(sentence: object) -> bool:
    return adverse_hit(sentence, drop_negation=True)


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    standard_deviation = values.std(ddof=1)
    if values.notna().sum() <= 1 or pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / standard_deviation


def industry_year_z(data: pd.DataFrame, source: str, target: str) -> pd.Series:
    values = pd.to_numeric(data[source], errors="coerce")
    industry = data["ExpertiseIndustryCode"].astype(str).str.strip()
    eligible = values.notna() & industry.ne("") & industry.ne("nan")
    out = pd.Series(np.nan, index=data.index)
    if not eligible.any():
        return out
    grouped = data.loc[eligible].groupby(["时间", "ExpertiseIndustryCode"], dropna=False)[source]
    out.loc[eligible] = grouped.transform(zscore)
    out.name = target
    return out


def load_common() -> pd.DataFrame:
    data = pd.read_csv(COMMON_PATH, dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    data["证券代码"] = data["证券代码"].map(pad_code)
    data["时间"] = pd.to_numeric(data["时间"], errors="coerce")
    data["ExpertiseIndustryCode"] = data["ExpertiseIndustryCode"].astype(str).str.strip()
    for column in ["GI", "GI_z_行业年度", "ESGE_z_行业年度", "Greenwashing连续_基准", "IMS", "IMS_D", *CONTROLS]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def significance(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def fit_full(data: pd.DataFrame, dependent: str, core: str, extra: list[str] | None = None):
    extra = extra or []
    controls = [*CONTROLS, *extra]
    required = [dependent, core, "证券代码", "时间", "ExpertiseIndustryCode", *controls]
    sample = data.dropna(subset=required).copy()
    formula = (
        f"Q('{dependent}') ~ Q('{core}') + "
        + " + ".join(f"Q('{name}')" for name in controls)
        + " + C(时间) + C(ExpertiseIndustryCode)"
    )
    result = smf.ols(formula, data=sample).fit(
        cov_type="cluster",
        cov_kwds={"groups": sample["证券代码"]},
    )
    return result, sample


def coef_row(result, model: str, dependent: str, core: str, extra: str) -> dict:
    name = f"Q('{core}')"
    return {
        "模型": model,
        "因变量": dependent,
        "核心变量": core,
        "额外控制": extra,
        "系数": float(result.params[name]),
        "标准误": float(result.bse[name]),
        "t值": float(result.tvalues[name]),
        "p值": float(result.pvalues[name]),
        "显著性": significance(float(result.pvalues[name])),
        "样本数": int(result.nobs),
        "调整R方": float(result.rsquared_adj),
    }


def load_penalty() -> pd.DataFrame:
    raw = pd.read_excel(PENALTY_PATH, dtype=str)
    code_col = "证券代码" if "证券代码" in raw.columns else raw.columns[0]
    date_col = "处罚日期" if "处罚日期" in raw.columns else "处罚时间"
    amount_col = "处罚金额（万元）" if "处罚金额（万元）" in raw.columns else None
    data = pd.DataFrame(
        {
            "证券代码": raw[code_col].map(pad_code),
            "时间": raw[date_col].map(year_from_enddate),
            "PenaltyAmt": pd.to_numeric(raw[amount_col], errors="coerce") if amount_col else np.nan,
        }
    )
    data = data[(data["时间"] >= 2021) & (data["时间"] <= 2024)]
    grouped = data.groupby(["证券代码", "时间"], as_index=False).agg(
        Penalty_Count=("证券代码", "size"),
        PenaltyAmt=("PenaltyAmt", "sum"),
    )
    grouped["Penalty_D"] = 1.0
    grouped["LnPenaltyAmt"] = np.log1p(grouped["PenaltyAmt"].clip(lower=0))
    return grouped
