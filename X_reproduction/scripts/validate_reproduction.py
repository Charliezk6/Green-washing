from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/reproduction_config.json").read_text(encoding="utf-8"))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate() -> dict[str, int | bool]:
    expertise = read(ROOT / CONFIG["expertise_output"])
    company = read(ROOT / CONFIG["company_output"])
    expertise_keys = {(row["时间"], row["InstitutionID"], row["ExpertiseIndustryCode"]) for row in expertise}
    company_keys = {(row["证券代码"].zfill(6), row["时间"]) for row in company}
    complete = [row for row in company if row["X变量匹配状态"] == "complete"]
    invalid_continuous = [row for row in complete if not (0 <= float(row["IMS"]) <= 1 and 0 <= float(row["IPS"]) <= 1)]
    invalid_binary = [row for row in complete if row["IMS_D"] not in {"0", "1"} or row["IPS_D"] not in {"0", "1"}]
    summary = {"expertise_rows": len(expertise), "expected_expertise_rows": CONFIG["expected_expertise_rows"], "expertise_unique_keys": len(expertise_keys), "company_rows": len(company), "company_unique_keys": len(company_keys), "complete_x_matches": len(complete), "expected_complete_x_matches": CONFIG["expected_company_rows"], "invalid_continuous": len(invalid_continuous), "invalid_binary": len(invalid_binary), "passed": len(expertise) == len(expertise_keys) == CONFIG["expected_expertise_rows"] and len(company) == len(company_keys) and len(complete) == CONFIG["expected_company_rows"] and not invalid_continuous and not invalid_binary}
    out = ROOT / "outputs/validation"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in (("invalid_continuous.csv", invalid_continuous), ("invalid_binary.csv", invalid_binary)):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]) if data else ["证券代码", "时间"])
            writer.writeheader(); writer.writerows(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise AssertionError("X_reproduction2 validation failed")
    return summary


if __name__ == "__main__":
    validate()
