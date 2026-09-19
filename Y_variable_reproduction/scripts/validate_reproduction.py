from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/reproduction_config.json").read_text(encoding="utf-8"))


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate() -> dict[str, int | bool]:
    rows = read(ROOT / CONFIG["output"])
    expected = CONFIG["expected"]
    complete = [row for row in rows if row["完成状态"] == "complete"]
    gw_complete = [row for row in rows if row["Greenwashing构造状态"] == "complete"]
    bad_gi = [row for row in complete if not 0 <= float(row["GI"]) <= 1]
    bad_gw = [row for row in gw_complete if row["Greenwashing"] not in {"0", "1"}]
    bad_level = [row for row in rows if row["样本类型"] == "制造业" and row["行业基准层级"] != "制造业二级"] + [row for row in rows if row["样本类型"] == "非制造业" and row["行业基准层级"] != "非制造业一级"]
    summary = {"rows": len(rows), "unique_company_year_keys": len({(row["证券代码"], row["时间"]) for row in rows}), "gi_complete": len(complete), "greenwashing_complete": len(gw_complete), "greenwashing_1": sum(row["Greenwashing"] == "1" for row in rows), "greenwashing_0": sum(row["Greenwashing"] == "0" for row in rows), "invalid_gi": len(bad_gi), "invalid_greenwashing": len(bad_gw), "invalid_industry_level": len(bad_level), "expected": expected, "passed": len(rows) == expected["samples"] and len(complete) == expected["gi_complete"] and len(gw_complete) == expected["greenwashing_complete"] and sum(row["Greenwashing"] == "1" for row in rows) == expected["greenwashing_1"] and sum(row["Greenwashing"] == "0" for row in rows) == expected["greenwashing_0"] and len({(row["证券代码"], row["时间"]) for row in rows}) == len(rows) and not bad_gi and not bad_gw and not bad_level}
    out = ROOT / "outputs/validation"; out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, data in (("invalid_gi.csv", bad_gi), ("invalid_greenwashing.csv", bad_gw), ("invalid_industry_level.csv", bad_level)):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise AssertionError("Updated Y reproduction validation failed")
    return summary


if __name__ == "__main__":
    validate()
