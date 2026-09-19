from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "deepgreen_reproduction/04_layer_b_rag/scripts/run_layer_b.py"

spec = importlib.util.spec_from_file_location("expansion_layer_b", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载原 Layer B 脚本：{SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module.INPUT_S2 = ROOT / "deepgreen_reproduction3/03_layer_a/s2/all_s2.jsonl"
module.OUTPUT_ROOT = ROOT / "deepgreen_reproduction3/04_layer_b_rag"

if __name__ == "__main__":
    module.main()
