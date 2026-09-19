from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/regression_config.json").read_text(encoding="utf-8"))


def validate() -> dict[str, int | bool]:
    panel = pd.read_csv(ROOT / CONFIG["input_panel"], dtype={"证券代码": str}, encoding="utf-8-sig", low_memory=False)
    actual = pd.read_csv(ROOT / CONFIG["output_dir"] / "基准回归完整系数.csv", encoding="utf-8-sig")
    samples = pd.read_csv(ROOT / CONFIG["output_dir"] / "基准回归样本信息.csv", encoding="utf-8-sig")
    expected = int(CONFIG["expected_common_sample"])
    sample_sizes = pd.to_numeric(samples["样本数"], errors="coerce")
    required_models = 8
    summary = {
        "input_rows": int(len(panel)),
        "unique_company_year_keys": int(panel[["证券代码", "时间"]].drop_duplicates().shape[0]),
        "coefficient_rows": int(len(actual)),
        "model_rows": int(len(samples)),
        "common_sample": int(sample_sizes.iloc[0]) if not samples.empty else 0,
        "expected_common_sample": expected,
        "all_models_use_common_sample": bool(sample_sizes.eq(expected).all()),
        "unique_company_year_passed": bool(panel[["证券代码", "时间"]].duplicated().sum() == 0),
        "expected_model_rows": required_models,
        "passed": bool(len(actual) > 0 and len(samples) == required_models and sample_sizes.eq(expected).all() and panel[["证券代码", "时间"]].duplicated().sum() == 0),
    }
    out = ROOT / "outputs/validation"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([summary]).to_csv(out / "validation_checks.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise AssertionError("regression2 validation failed")
    return summary


if __name__ == "__main__":
    validate()
