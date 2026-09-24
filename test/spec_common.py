from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
TEST = Path(__file__).resolve().parent
PROJECT = TEST.parent
DATA = TEST / "data"
sys.path.insert(0, str(PROJECT / "Regression/scripts"))

import reproduce_regression as rr  # noqa: E402

SPEC_COMPANY_YEAR = DATA / "spec_company_year_full_market.csv"
SPEC_COLUMNS = ["SPEC_AuditorID", "SPEC_Fee", "FeeShare", "SPEC", "SPEC_LOO"]
CORE = "SPEC"
FIRM, YEAR, INDUSTRY = "证券代码", "时间", "ExpertiseIndustryCode"


def significance(p: float) -> str:
    return "***" if p < .01 else "**" if p < .05 else "*" if p < .10 else ""


def load_spec() -> pd.DataFrame:
    spec = pd.read_csv(SPEC_COMPANY_YEAR, dtype={FIRM: str, "SPEC_AuditorID": str}, encoding="utf-8-sig")
    spec[FIRM] = spec[FIRM].str.zfill(6)
    return spec[[FIRM, YEAR, *SPEC_COLUMNS]]


def load_data_with_spec() -> pd.DataFrame:
    data = rr.load_data()
    data[FIRM] = data[FIRM].astype(str).str.zfill(6)
    return data.merge(load_spec(), on=[FIRM, YEAR], how="left", validate="many_to_one")


def common_sample(data: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same common sample and winsorization as the baseline, with SPEC added to the required set."""
    data = load_data_with_spec() if data is None else data
    required = [rr.DEPENDENT, *rr.CORE, CORE, *rr.ALL_CONTROLS, FIRM, YEAR, INDUSTRY]
    common = data.dropna(subset=required).copy()
    return rr.winsorize_controls(common)
