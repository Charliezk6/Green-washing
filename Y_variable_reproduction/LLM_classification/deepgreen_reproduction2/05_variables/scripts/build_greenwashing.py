from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "deepgreen_reproduction/05_variables/scripts/build_greenwashing.py"

spec = importlib.util.spec_from_file_location("manufacturing_greenwashing_variables", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载原变量构造脚本：{SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.LAYER_B = ROOT / "deepgreen_reproduction2/04_layer_b_rag/results/layer_b_results.jsonl"
module.WHITELIST = ROOT / "deepgreen_reproduction2/02_environment_extraction/reports/exact_good_samples.csv"
module.ESG = ROOT / "deepgreen_reproduction2/01_sample_panel/data/non_manufacturing_esg_universe_2021_2024.csv"
module.OUT = ROOT / "deepgreen_reproduction2/05_variables"

if __name__ == "__main__":
    module.main()
