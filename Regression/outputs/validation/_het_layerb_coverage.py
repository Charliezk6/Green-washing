# -*- coding: utf-8 -*-
import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"d:\Charlie 应用\Documents\zkk\Research Project\greenwashing")
files = [
    ROOT / "Y_variable_reproduction/data/input/manufacturing/layer_b_results.jsonl",
    ROOT / "Y_variable_reproduction/data/input/non_manufacturing/layer_b_results.jsonl",
    ROOT / "Y_variable_reproduction/data/input/expansion/layer_b_results.jsonl",
]
keys = {name: set() for name in ["mfg", "non", "exp"]}
names = ["mfg", "non", "exp"]
for name, path in zip(names, files):
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            keys[name].add((str(rec["证券代码"]).zfill(6), int(rec["时间"])))
    print(name, "fy", len(keys[name]))

union = keys["mfg"] | keys["non"] | keys["exp"]
print("union", len(union), "mfg&non", len(keys["mfg"] & keys["non"]), "mfg&exp", len(keys["mfg"] & keys["exp"]), "non&exp", len(keys["non"] & keys["exp"]))
common = pd.read_csv(ROOT / "Regression/outputs/paper_tables/common_sample_15381.csv", dtype={"证券代码": str}, encoding="utf-8-sig", usecols=["证券代码", "时间"])
common_keys = set(zip(common["证券代码"].str.zfill(6), common["时间"].astype(int)))
print("common", len(common_keys))
print("common in union", len(common_keys & union), "missing", len(common_keys - union))
print("common in mfg", len(common_keys & keys["mfg"]))
print("common in non", len(common_keys & keys["non"]))
print("common in exp", len(common_keys & keys["exp"]))
panel = pd.read_csv(ROOT / "Regression/data/input/combined_greenwashing_auditor_expertise_panel.csv", dtype={"证券代码": str}, encoding="utf-8-sig", usecols=["证券代码", "时间", "绿色披露总量", "GI"])
print("panel", len(panel), "GI notna", panel.GI.notna().sum(), "green总量>0", (pd.to_numeric(panel["绿色披露总量"], errors="coerce")>0).sum())
