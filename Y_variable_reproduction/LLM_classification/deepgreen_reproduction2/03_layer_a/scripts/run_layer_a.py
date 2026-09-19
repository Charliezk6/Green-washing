from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "deepgreen_reproduction/03_layer_a/scripts/run_layer_a.py"

spec = importlib.util.spec_from_file_location("manufacturing_layer_a", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载原 Layer A 脚本：{SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.INPUT_WORDS = ROOT / "deepgreen_reproduction2/02_environment_extraction/s1/all_s1_words.csv"
module.WHITELIST = ROOT / "deepgreen_reproduction2/02_environment_extraction/reports/exact_good_samples.csv"
module.SENTENCES_DIR = ROOT / "deepgreen_reproduction2/02_environment_extraction/sentences"
module.OUTPUT_ROOT = ROOT / "deepgreen_reproduction2/03_layer_a"

if __name__ == "__main__":
    module.main()
