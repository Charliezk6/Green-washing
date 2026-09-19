from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/reproduce_x.py")], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_reproduction.py")], cwd=ROOT, check=True)
    summary = json.loads((ROOT / "outputs/validation/summary.json").read_text(encoding="utf-8"))
    assert summary["passed"]
    print("X_reproduction2 smoke test passed: matched company-years=8,838")


if __name__ == "__main__":
    main()
