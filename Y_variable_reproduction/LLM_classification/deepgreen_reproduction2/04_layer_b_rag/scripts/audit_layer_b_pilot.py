from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCE_SCRIPT = ROOT / "deepgreen_reproduction/04_layer_b_rag/scripts/audit_layer_b_pilot.py"

spec = importlib.util.spec_from_file_location("manufacturing_layer_b_audit", SOURCE_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载原 Layer B 复核脚本：{SOURCE_SCRIPT}")
module = importlib.util.module_from_spec(spec)

module.__dict__["ROOT"] = ROOT
source = SOURCE_SCRIPT.read_text(encoding="utf-8")
source = source.replace(
    "SOURCE=ROOT/'deepgreen_reproduction/04_layer_b_rag/results/unique_task_results.jsonl'",
    "SOURCE=ROOT/'deepgreen_reproduction2/04_layer_b_rag/results/unique_task_results.jsonl'",
).replace(
    "OUT=ROOT/'deepgreen_reproduction/04_layer_b_rag/validation'",
    "OUT=ROOT/'deepgreen_reproduction2/04_layer_b_rag/validation'",
)
exec(compile(source, str(SOURCE_SCRIPT), "exec"), module.__dict__)
