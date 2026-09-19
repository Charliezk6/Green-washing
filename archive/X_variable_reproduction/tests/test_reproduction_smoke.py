from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_x.py")], cwd=ROOT, check=True)
    expertise_path = ROOT / "outputs/expertise/auditor_industry_expertise_2021_2024.csv"
    company_path = ROOT / "outputs/company_year/greenwashing_panel_with_auditor_expertise_2021_2024.csv"
    with expertise_path.open(encoding="utf-8-sig", newline="") as handle:
        expertise = list(csv.DictReader(handle))
    with company_path.open(encoding="utf-8-sig", newline="") as handle:
        companies = list(csv.DictReader(handle))
    assert expertise and companies
    assert len({(row["时间"], row["InstitutionID"], row["ExpertiseIndustryCode"]) for row in expertise}) == len(expertise)
    assert len({(row["证券代码"], row["时间"]) for row in companies}) == len(companies)
    assert all(0 <= float(row["IMS"]) <= 1 and 0 <= float(row["IPS"]) <= 1 for row in expertise)
    assert {row["IMS_D"] for row in expertise} <= {"0", "1"}
    assert {row["IPS_D"] for row in expertise} <= {"0", "1"}
    print(f"Smoke test passed: expertise={len(expertise)} company_year={len(companies)}")


if __name__ == "__main__":
    main()
