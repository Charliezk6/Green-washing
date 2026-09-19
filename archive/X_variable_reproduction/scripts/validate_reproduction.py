from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / "config/reproduction_config.json").read_text(encoding="utf-8"))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str) -> float | None:
    return None if value in (None, "") else float(value)


def compare(actual_path: Path, reference_path: Path, keys: list[str], fields: list[str], tolerance: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actual_rows, reference_rows = read(actual_path), read(reference_path)
    actual = {tuple(row[field] for field in keys): row for row in actual_rows}
    reference = {tuple(row[field] for field in keys): row for row in reference_rows}
    differences = []
    max_errors = {field: 0.0 for field in fields}
    for key in sorted(actual.keys() & reference.keys()):
        for field in fields:
            left, right = number(actual[key].get(field, "")), number(reference[key].get(field, ""))
            error = 0.0 if left is None and right is None else math.inf if left is None or right is None else abs(left - right)
            max_errors[field] = max(max_errors[field], error)
            if error > tolerance:
                differences.append({**dict(zip(keys, key)), "字段": field, "复现值": actual[key].get(field, ""),
                                    "参考值": reference[key].get(field, ""), "绝对误差": error})
    summary = {"actual_rows": len(actual), "reference_rows": len(reference),
               "missing_keys": len(reference.keys() - actual.keys()), "extra_keys": len(actual.keys() - reference.keys()),
               "differing_cells": len(differences), "max_absolute_errors": max_errors}
    summary["passed"] = summary["missing_keys"] == summary["extra_keys"] == summary["differing_cells"] == 0
    return summary, differences


def validate_all(expertise_path: Path | None = None, company_path: Path | None = None) -> dict[str, Any]:
    expertise_path = expertise_path or ROOT / CONFIG["expertise_output"]
    company_path = company_path or ROOT / CONFIG["company_output"]
    expertise_summary, expertise_differences = compare(
        expertise_path, ROOT / CONFIG["expertise_reference"],
        ["时间", "InstitutionID", "ExpertiseIndustryCode"],
        ["ClientCount", "FirmIndustryRevenue", "IndustryYearRevenue", "FirmYearRevenue", "IMS", "IMS_D", "IPS", "IPS_D"], 0.02)
    company_summary, company_differences = compare(
        company_path, ROOT / CONFIG["company_reference"], ["证券代码", "时间"],
        ["IMS", "IMS_D", "IPS", "IPS_D"], 1e-10)
    output_dir = ROOT / "outputs/validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("expertise_differences.csv", expertise_differences), ("company_differences.csv", company_differences)):
        fields = list(rows[0]) if rows else ["证券代码", "时间", "字段", "复现值", "参考值", "绝对误差"]
        with (output_dir / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    summary = {"expertise": expertise_summary, "company_year": company_summary,
               "passed": expertise_summary["passed"] and company_summary["passed"]}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise AssertionError("X 变量复现结果与参考文件不一致，详见 outputs/validation")
    return summary


if __name__ == "__main__":
    validate_all()
