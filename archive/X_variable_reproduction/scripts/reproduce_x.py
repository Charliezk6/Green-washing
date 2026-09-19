from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
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


def check_inputs() -> None:
    required = [ROOT / CONFIG[name] for name in ("market_input", "company_panel_input", "expertise_reference", "company_reference")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少复现输入：\n" + "\n".join(missing))


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
    rows = []
    for (year, institution, code), aggregate in sorted(firm_industry.items()):
        sample = next(row for row in valid if row["时间"] == year and row["InstitutionID"] == institution
                      and row["ExpertiseIndustryCode"] == code)
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


def merge_company_panel(expertise_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    company_rows, fields = read_csv(ROOT / CONFIG["company_panel_input"])
    expertise = {(str(row["时间"]), str(row["InstitutionID"]), row["ExpertiseIndustryCode"]): row
                 for row in expertise_rows}
    output_fields = fields + ["ExpertiseIndustryCode", "IMS", "IMS_D", "IPS", "IPS_D"]
    output = []
    for row in company_rows:
        industry_code = row.get("IndustryCodeD") or row.get("IndustryCode") or ""
        code = expertise_code(industry_code)
        institution = row.get("InstitutionID", "")
        institution = str(int(float(institution))) if institution else ""
        match = expertise.get((row["时间"], institution, code), {})
        if match:
            output.append({**row, "ExpertiseIndustryCode": code,
                           "IMS": match["IMS"], "IMS_D": match["IMS_D"],
                           "IPS": match["IPS"], "IPS_D": match["IPS_D"]})
    return output, output_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="从全 A 股审计市场复现会计师事务所行业专长 X 变量")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    check_inputs()
    expertise_rows = reproduce_expertise()
    expertise_fields = ["时间", "InstitutionID", "ExpertiseIndustryCode", "AccountingFirmName", "IndustryName",
                        "ClientCount", "FirmIndustryRevenue", "IndustryYearRevenue", "FirmYearRevenue",
                        "IMS", "IMS_D", "IPS", "IPS_D"]
    expertise_path = ROOT / CONFIG["expertise_output"]
    write_csv(expertise_path, expertise_rows, expertise_fields)
    company_rows, company_fields = merge_company_panel(expertise_rows)
    company_path = ROOT / CONFIG["company_output"]
    write_csv(company_path, company_rows, company_fields)
    print(f"已生成 {len(expertise_rows)} 条事务所行业年度记录和 {len(company_rows)} 条公司年度记录")
    if not args.skip_validation:
        from validate_reproduction import validate_all
        validate_all(expertise_path, company_path)


if __name__ == "__main__":
    main()
