from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_regression.py")], cwd=ROOT, check=True)
    result = pd.read_csv(ROOT / "outputs/baseline/基准回归完整系数.csv", encoding="utf-8-sig")
    samples = pd.read_csv(ROOT / "outputs/baseline/基准回归样本信息.csv", encoding="utf-8-sig")
    assert len(result) > 0
    assert set(result["核心变量"]) == {"IMS", "IMS_D"}
    assert samples["样本数"].nunique() == 1
    assert int(samples["样本数"].iloc[0]) == 8129
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_reproduction.py")], cwd=ROOT, check=True)
    summary = json.loads((ROOT / "outputs/validation/summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    print("Regression smoke test passed: N=8129")


if __name__ == "__main__":
    main()
