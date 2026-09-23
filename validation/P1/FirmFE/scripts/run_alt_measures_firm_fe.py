"""Main-specification firm FE for baseline IMS and every alternative specialization measure.

Each measure is estimated once, alone, with firm and year fixed effects, the full
baseline control set, and firm-clustered standard errors. Samples follow the
pipeline that originally built each measure, so coefficients match the existing
tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "partner" / "scripts"))

from diagnose_firm_fe_power import add_ims_loo, prepare_common  # noqa: E402
from measurement_error_correction import prepare as prepare_fee_asset  # noqa: E402
from run_firm_fe_regression import ALL_CONTROLS, DEPENDENT, fit, significance  # noqa: E402
from run_partner_fe_regression import prepare as prepare_partner  # noqa: E402

TABLES = ROOT / "outputs" / "tables"
FIRM = "证券代码"
THRESHOLD = 0.20
TOLERANCE = 1e-8

EXPECTED_N = {
    "IMS": 15381,
    "IMS_fee": 15279,
    "IMS_asset": 15279,
    "IMS_LOO": 15277,
    "IMS_D": 15381,
    "IMS_fee_D": 15279,
    "IMS_asset_D": 15279,
    "IMS_LOO_D": 15277,
    "PIMS_max": 15380,
    "PIMS_mean": 15380,
    "PIMS_LOO_max": 15380,
    "PIMS_LOO_mean": 15380,
    "PIMS_D_leader": 15380,
    "PIMS_D_p90": 15380,
}


def binary_from(data: pd.DataFrame, continuous: str, binary: str) -> None:
    share = data[continuous]
    data[binary] = (share >= THRESHOLD).astype(float)
    data.loc[share.isna(), binary] = np.nan


def within_stats(sample: pd.DataFrame, core: str) -> tuple[float, float]:
    within = sample[core] - sample.groupby(FIRM)[core].transform("mean")
    changed = sample.groupby(FIRM)[core].nunique().gt(1)
    return float(within.std(ddof=1)), float(changed.mean())


def estimate(data: pd.DataFrame, level: str, kind: str, core: str) -> dict:
    required = [DEPENDENT, core, FIRM, "时间", *ALL_CONTROLS]
    sample = data.dropna(subset=required).copy()
    within_sd, changed_share = within_stats(sample, core)
    result, used = fit(sample, core, ALL_CONTROLS)
    p_value = float(result.pvalues[core])
    n_obs = int(result.nobs)
    if n_obs != EXPECTED_N[core]:
        raise ValueError(f"{core}: expected N={EXPECTED_N[core]}, got {n_obs}.")
    return {
        "层级": level,
        "类型": kind,
        "变量": core,
        "系数": float(result.params[core]),
        "标准误": float(result.bse[core]),
        "t值": float(result.tvalues[core]),
        "p值": p_value,
        "显著性": significance(p_value),
        "N": n_obs,
        "公司数": int(used[FIRM].nunique()),
        "组内标准差": within_sd,
        "取值有变化的公司占比": changed_share,
    }


def reference_coefficients() -> dict[str, float]:
    firm = pd.read_csv(ROOT / "outputs" / "coefficients" / "firm_fe_coefficients.csv", encoding="utf-8-sig")
    firm = firm[firm["模型"].str.contains(r"\(4\) 全部控制", regex=True)]
    specs = pd.read_csv(ROOT / "outputs" / "diagnostics" / "firm_fe_spec_extensions.csv", encoding="utf-8-sig")
    specs = specs[specs["规格"] == "企业+年份FE_LOO处理变量"]
    partner = pd.read_csv(
        ROOT / "partner" / "outputs" / "tables" / "partner_fe_all_measures.csv",
        encoding="utf-8-sig",
    )
    partner = partner[(partner["规格"] == "企业+年份FE") & (partner["聚类"] == "公司")]
    expected = {
        "IMS": float(firm.loc[firm["变量"] == "IMS", "系数"].iloc[0]),
        "IMS_D": float(firm.loc[firm["变量"] == "IMS_D", "系数"].iloc[0]),
        "IMS_LOO": float(specs.loc[specs["核心变量"] == "IMS_LOO", "系数"].iloc[0]),
        "IMS_LOO_D": float(specs.loc[specs["核心变量"] == "IMS_LOO_D", "系数"].iloc[0]),
    }
    for variable in ("PIMS_max", "PIMS_mean", "PIMS_LOO_max", "PIMS_LOO_mean", "PIMS_D_leader", "PIMS_D_p90"):
        expected[variable] = float(partner.loc[partner["变量"] == variable, "系数"].iloc[0])
    return expected


def check_against_existing(table: pd.DataFrame) -> None:
    expected = reference_coefficients()
    for variable, target in expected.items():
        got = float(table.loc[table["变量"] == variable, "系数"].iloc[0])
        if abs(got - target) > TOLERANCE:
            raise AssertionError(f"{variable}: {got} != {target}")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    fee_asset = prepare_fee_asset()
    binary_from(fee_asset, "IMS_fee", "IMS_fee_D")
    binary_from(fee_asset, "IMS_asset", "IMS_asset_D")
    baseline = prepare_common()
    loo = add_ims_loo(baseline)
    partner, _ = prepare_partner()

    jobs = [
        ("事务所", "连续", "IMS", baseline),
        ("事务所", "连续", "IMS_fee", fee_asset),
        ("事务所", "连续", "IMS_asset", fee_asset),
        ("事务所", "连续", "IMS_LOO", loo),
        ("事务所", "二元", "IMS_D", baseline),
        ("事务所", "二元", "IMS_fee_D", fee_asset),
        ("事务所", "二元", "IMS_asset_D", fee_asset),
        ("事务所", "二元", "IMS_LOO_D", loo),
        ("签字合伙人", "连续", "PIMS_max", partner),
        ("签字合伙人", "连续", "PIMS_mean", partner),
        ("签字合伙人", "连续", "PIMS_LOO_max", partner),
        ("签字合伙人", "连续", "PIMS_LOO_mean", partner),
        ("签字合伙人", "二元", "PIMS_D_leader", partner),
        ("签字合伙人", "二元", "PIMS_D_p90", partner),
    ]
    table = pd.DataFrame(estimate(data, level, kind, core) for level, kind, core, data in jobs)
    check_against_existing(table)
    table.to_csv(TABLES / "firm_fe_alt_measures.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
