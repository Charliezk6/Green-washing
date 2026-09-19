from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "deepgreen_reproduction/02_environment_extraction/scripts/audit_extraction_quality.py"

spec = importlib.util.spec_from_file_location("manufacturing_extraction_quality_audit", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载原环境文本质检脚本：{SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.STAGE = ROOT / "deepgreen_reproduction2/02_environment_extraction"
module.MANIFEST = module.STAGE / "reports/extraction_manifest.csv"
module.SENTENCES = module.STAGE / "sentences"
module.S1 = module.STAGE / "s1/all_s1_words.csv"
module.REPORTS = module.STAGE / "reports/quality_audit"

if __name__ == "__main__":
    module.main()
