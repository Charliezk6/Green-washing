from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/run_ips_regression.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_paper_tables.py")], cwd=ROOT, check=True)
    result = pd.read_csv(ROOT / "outputs/baseline/ips_regression_coefficients.csv", encoding="utf-8-sig")
    samples = pd.read_csv(ROOT / "outputs/baseline/ips_regression_sample_info.csv", encoding="utf-8-sig")
    expected = 14778
    assert not result.empty
    assert set(result["核心变量"]) == {"IPS", "IPS_D"}
    assert samples["样本数"].nunique() == 1
    assert int(samples["样本数"].iloc[0]) == expected
    assert (ROOT / "outputs/paper_tables/ips_regression_table.tex").exists()
    assert (ROOT / "outputs/paper_tables/ips_d_regression_table.tex").exists()
    print(f"IPS regression smoke test passed: N={expected}")


if __name__ == "__main__":
    main()
