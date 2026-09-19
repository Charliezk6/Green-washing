"""Literature-standard heterogeneity using the existing 15,381 panel only."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from hetero_common import (
    DERIVED,
    TABLES,
    load_analysis_panel,
    run_split,
    run_three_layer,
)

# 环保部《上市公司环保核查行业分类管理名录》对照 2012 证监会代码，
# 并并入 CAS 1631/1521 应用指南点名的高环境风险行业（公用事业 D45/D46）。
HEAVY_POLL_CODES = {
    "B06",  # 煤炭开采和洗选业
    "B07",  # 石油和天然气开采业
    "B08",  # 黑色金属矿采选业
    "B09",  # 有色金属矿采选业
    "B10",  # 非金属矿采选业
    "C13",  # 农副食品加工业（发酵）
    "C15",  # 酒、饮料和精制茶制造业（酿造）
    "C17",  # 纺织业（含印染）
    "C19",  # 皮革、毛皮、羽毛及其制品和制鞋业（制革）
    "C22",  # 造纸和纸制品业
    "C25",  # 石油、煤炭及其他燃料加工业（石化）
    "C26",  # 化学原料和化学制品制造业
    "C27",  # 医药制造业
    "C28",  # 化学纤维制造业
    "C30",  # 非金属矿物制品业（水泥/建材）
    "C31",  # 黑色金属冶炼和压延加工业（钢铁/冶金）
    "C32",  # 有色金属冶炼和压延加工业（电解铝/冶金）
    "D44",  # 电力、热力生产和供应业（火电/公用事业）
    "D45",  # 燃气生产和供应业（公用事业）
    "D46",  # 水的生产和供应业（公用事业）
}


def add_lit_moderators(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    industry = out["IndustryCode"].astype(str).str.strip()
    out["HeavyPoll"] = industry.isin(HEAVY_POLL_CODES).astype(int)

    out["SOE"] = pd.to_numeric(out["SOE"], errors="coerce")
    tenure = pd.to_numeric(out["AuditorTenure"], errors="coerce")
    out["ShortTenure"] = (tenure <= 3).astype(float)
    out.loc[tenure.isna(), "ShortTenure"] = np.nan

    fee = pd.to_numeric(out["TotalAuditFee"], errors="coerce")
    auditor = out["InstitutionID"].astype(str).str.strip()
    year = out["时间"]
    auditor_year_sum = fee.groupby([auditor, year]).transform("sum")
    out["CI"] = fee / auditor_year_sum.where(auditor_year_sum > 0)
    lower, upper = out["CI"].quantile([0.01, 0.99])
    out["CI"] = out["CI"].clip(lower=lower, upper=upper)
    out["HighCI"] = (out["CI"] > out["CI"].median()).astype(float)
    out.loc[out["CI"].isna(), "HighCI"] = np.nan
    return out


def coverage_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, column in [
        ("HeavyPoll", "HeavyPoll"),
        ("SOE", "SOE"),
        ("ShortTenure", "ShortTenure"),
        ("HighCI", "HighCI"),
    ]:
        series = pd.to_numeric(data[column], errors="coerce")
        rows.append(
            {
                "variable": name,
                "n": int(series.notna().sum()),
                "n1": int(series.eq(1).sum()),
                "share1": float(series.eq(1).mean()),
            }
        )
    ci = pd.to_numeric(data["CI"], errors="coerce")
    rows.append(
        {
            "variable": "CI",
            "n": int(ci.notna().sum()),
            "n1": np.nan,
            "share1": np.nan,
            "mean": float(ci.mean()),
            "p50": float(ci.median()),
            "p90": float(ci.quantile(0.9)),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    data = add_lit_moderators(load_analysis_panel())
    coverage = coverage_table(data)
    coverage.to_csv(TABLES / "lit_hetero_coverage.csv", index=False, encoding="utf-8-sig")
    data[["证券代码", "时间", "IndustryCode", "IndustryName", "HeavyPoll", "SOE", "ShortTenure", "CI", "HighCI"]].to_csv(
        DERIVED / "lit_moderators_panel.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    dummy_specs = [
        ("HeavyPoll", "HeavyPoll", "Heavy polluting industry"),
        ("SOE", "SOE", "State ownership"),
        ("ShortTenure", "ShortTenure", "Short auditor tenure"),
        ("HighCI", "HighCI", "Client importance high/low"),
    ]
    for core in ["IMS", "IMS_D"]:
        for moderator, splitter, label in dummy_specs:
            table, _, _ = run_three_layer(data, core, moderator, label)
            rows.append(table)
            rows.append(run_split(data, core, splitter, label))
        table, _, _ = run_three_layer(data, core, "CI", "Client importance continuous")
        rows.append(table)

    output = pd.concat(rows, ignore_index=True)
    output.to_csv(TABLES / "lit_hetero_coefficients.csv", index=False, encoding="utf-8-sig")
    print(coverage.to_string(index=False))
    focus = output[output["variable"].str.contains(r"IMS|:")]
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
