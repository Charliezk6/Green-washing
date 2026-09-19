from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/reproduction_config.json").read_text(encoding="utf-8"))


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def expertise_code(industry_code: str) -> str:
    code = industry_code.strip()
    return code if code.startswith("C") else code[:1]


def reproduce_expertise() -> list[dict[str, Any]]:
    market, _ = read_csv(ROOT / CONFIG["market_input"])
    valid = []
    for row in market:
        try:
            revenue = float(row[CONFIG["scale_variable"]])
            institution = int(float(row["InstitutionID"]))
        except (TypeError, ValueError):
            continue
        code = expertise_code(row.get("IndustryCode", ""))
        if int(row["时间"]) not in CONFIG["years"] or revenue <= 0 or not code or code == "O":
            continue
        valid.append({**row, "时间": int(row["时间"]), "InstitutionID": institution,
                      "ExpertiseIndustryCode": code, "Revenue": revenue})

    firm_industry: dict[tuple[int, int, str], dict[str, Any]] = defaultdict(
        lambda: {"ClientCount": 0, "FirmIndustryRevenue": 0.0}
    )
    industry_year: dict[tuple[int, str], float] = defaultdict(float)
    firm_year: dict[tuple[int, int], float] = defaultdict(float)
    for row in valid:
        key = (row["时间"], row["InstitutionID"], row["ExpertiseIndustryCode"])
        firm_industry[key]["ClientCount"] += 1
        firm_industry[key]["FirmIndustryRevenue"] += row["Revenue"]
        industry_year[(row["时间"], row["ExpertiseIndustryCode"])] += row["Revenue"]
        firm_year[(row["时间"], row["InstitutionID"])] += row["Revenue"]

    threshold = float(CONFIG["binary_threshold"])
    name_lookup: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in valid:
        key = (row["时间"], row["InstitutionID"], row["ExpertiseIndustryCode"])
        name_lookup.setdefault(key, row)

    rows = []
    for (year, institution, code), aggregate in sorted(firm_industry.items()):
        sample = name_lookup[(year, institution, code)]
        firm_industry_revenue = aggregate["FirmIndustryRevenue"]
        industry_revenue = industry_year[(year, code)]
        firm_revenue = firm_year[(year, institution)]
        ims = firm_industry_revenue / industry_revenue
        ips = firm_industry_revenue / firm_revenue
        rows.append({
            "时间": year, "InstitutionID": institution, "ExpertiseIndustryCode": code,
            "AccountingFirmName": sample["AccountingFirmName"],
            "IndustryName": sample["IndustryName"] if code.startswith("C") else code,
            "ClientCount": aggregate["ClientCount"], "FirmIndustryRevenue": firm_industry_revenue,
            "IndustryYearRevenue": industry_revenue, "FirmYearRevenue": firm_revenue,
            "IMS": ims, "IMS_D": int(ims >= threshold), "IPS": ips, "IPS_D": int(ips >= threshold),
        })
    return rows


def main() -> None:
    expertise_rows = reproduce_expertise()
    expertise_fields = ["时间", "InstitutionID", "ExpertiseIndustryCode", "AccountingFirmName", "IndustryName", "ClientCount", "FirmIndustryRevenue", "IndustryYearRevenue", "FirmYearRevenue", "IMS", "IMS_D", "IPS", "IPS_D"]
    expertise_path = ROOT / CONFIG["expertise_output"]
    write_csv(expertise_path, expertise_rows, expertise_fields)

    panel_rows, panel_fields = read_csv(ROOT / CONFIG["company_panel_input"])
    market_rows, _ = read_csv(ROOT / CONFIG["market_input"])
    market = {(row["证券代码"].zfill(6), row["时间"]): row for row in market_rows}
    expertise = {(str(row["时间"]), str(row["InstitutionID"]), row["ExpertiseIndustryCode"]): row for row in expertise_rows}
    output_fields = panel_fields + ["AccountingFirmName", "InstitutionID", "IndustryName", "IndustryCode", "ExpertiseIndustryCode", "IMS", "IMS_D", "IPS", "IPS_D", "X变量匹配状态"]
    output: list[dict[str, Any]] = []
    missing_market: list[dict[str, Any]] = []
    missing_expertise: list[dict[str, Any]] = []
    for row in panel_rows:
        key = (row["证券代码"].zfill(6), row["时间"])
        audit = market.get(key)
        if not audit:
            output.append({**row, "X变量匹配状态": "missing_audit_market"})
            missing_market.append(row)
            continue
        code = expertise_code(audit.get("IndustryCode", ""))
        try:
            institution = str(int(float(audit["InstitutionID"])))
        except (KeyError, TypeError, ValueError):
            institution = ""
        match = expertise.get((row["时间"], institution, code))
        record = {**row, "AccountingFirmName": audit.get("AccountingFirmName", ""), "InstitutionID": institution, "IndustryName": audit.get("IndustryName", ""), "IndustryCode": audit.get("IndustryCode", ""), "ExpertiseIndustryCode": code}
        if match:
            output.append({**record, "IMS": match["IMS"], "IMS_D": match["IMS_D"], "IPS": match["IPS"], "IPS_D": match["IPS_D"], "X变量匹配状态": "complete"})
        else:
            output.append({**record, "X变量匹配状态": "missing_expertise"})
            missing_expertise.append(record)
    write_csv(ROOT / CONFIG["company_output"], output, output_fields)
    write_csv(ROOT / "outputs/reports/missing_audit_market.csv", missing_market, panel_fields)
    write_csv(ROOT / "outputs/reports/missing_expertise.csv", missing_expertise, output_fields)
    summary = {"expansion_panel_rows": len(panel_rows), "expertise_rows": len(expertise_rows), "complete_x_matches": sum(row["X变量匹配状态"] == "complete" for row in output), "missing_audit_market": len(missing_market), "missing_expertise": len(missing_expertise), "x_rule": "全A股市场营业收入份额；制造业C保留三位代码，其他行业取首字母；IMS/IPS阈值为20%"}
    reports = ROOT / "outputs/reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
