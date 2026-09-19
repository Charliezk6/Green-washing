from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config/regression_config.json").read_text(encoding="utf-8"))


def validate() -> dict:
    actual = pd.read_csv(ROOT / CONFIG["output_dir"] / "基准回归完整系数.csv", encoding="utf-8-sig")
    reference = pd.read_csv(ROOT / CONFIG["reference_results"], encoding="utf-8-sig")
    key = ["模型", "核心变量", "变量"]
    merged = actual.merge(reference, on=key, how="outer", suffixes=("_复现", "_参考"), indicator=True)
    numeric = ["系数", "标准误", "t值", "p值", "样本数", "调整R方"]
    differences = []
    common = merged[merged["_merge"] == "both"]
    for field in numeric:
        left = pd.to_numeric(common[f"{field}_复现"], errors="coerce")
        right = pd.to_numeric(common[f"{field}_参考"], errors="coerce")
        error = (left - right).abs()
        bad = common.loc[error > 1e-8, key].copy()
        bad["字段"] = field
        bad["复现值"] = left[error > 1e-8].values
        bad["参考值"] = right[error > 1e-8].values
        bad["绝对误差"] = error[error > 1e-8].values
        differences.append(bad)
    diff = pd.concat(differences, ignore_index=True) if differences else pd.DataFrame()
    summary = {"actual_rows": int(len(actual)), "reference_rows": int(len(reference)), "missing_keys": int((merged["_merge"] == "right_only").sum()), "extra_keys": int((merged["_merge"] == "left_only").sum()), "differing_cells": int(len(diff)), "passed": bool(len(diff) == 0 and bool((merged["_merge"] == "both").all()))}
    output = ROOT / "outputs/validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    diff.to_csv(output / "coefficient_differences.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise AssertionError("回归结果与参考文件不一致")
    return summary


if __name__ == "__main__":
    validate()
