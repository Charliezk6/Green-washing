from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "deepgreen_reproduction/02_environment_extraction/scripts/extract_environment_text.py"

spec = importlib.util.spec_from_file_location("manufacturing_environment_extraction", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载原环境文本抽取脚本：{SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.DEFAULT_PANEL = ROOT / "deepgreen_reproduction2/01_sample_panel/data/non_manufacturing_panel_2021_2024.csv"
module.OUTPUT_ROOT = ROOT / "deepgreen_reproduction2/02_environment_extraction"

if __name__ == "__main__":
    module.main()
