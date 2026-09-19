from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
LAYER_B = ROOT / "deepgreen_reproduction/04_layer_b_rag/results/layer_b_results.jsonl"
WHITELIST = ROOT / "deepgreen_reproduction/02_environment_extraction/reports/frozen_low_risk_samples.csv"
ESG = ROOT / "deepgreen_reproduction/01_sample_panel/data/manufacturing_esg_universe_2021_2024.csv"
OUT = ROOT / "deepgreen_reproduction/05_variables"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with WHITELIST.open(encoding="utf-8-sig", newline="") as handle:
        whitelist = list(csv.DictReader(handle))
    metadata = {
        f"{row['证券代码'].zfill(6)}_{row['时间']}": row
        for row in whitelist
    }
    with ESG.open(encoding="utf-8-sig", newline="") as handle:
        esg_rows = list(csv.DictReader(handle))
    esg = {
        f"{row['证券代码'].zfill(6)}_{row['时间']}": row
        for row in esg_rows
    }

    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"X": 0, "Y": 0, "confidences": [], "words": set(), "sentences": set(), "failed": 0}
    )
    with LAYER_B.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["key"])
            aggregate = aggregates[key]
            if row.get("status") != "success" or row.get("judgment") not in (0, 1):
                aggregate["failed"] += 1
                continue
            aggregate["X" if row["judgment"] == 1 else "Y"] += 1
            aggregate["confidences"].append(float(row["confidence"]))
            aggregate["words"].add(str(row["word"]))
            aggregate["sentences"].add(int(row["sentence_id"]))

    gi_rows: list[dict[str, Any]] = []
    for key in sorted(metadata):
        source = metadata[key]
        aggregate = aggregates.get(key)
        x = int(aggregate["X"]) if aggregate else 0
        y = int(aggregate["Y"]) if aggregate else 0
        failed = int(aggregate["failed"]) if aggregate else 0
        total = x + y
        complete = aggregate is not None and failed == 0 and total > 0
        gi = x / total if complete else None
        esg_row = esg.get(key, {})
        gi_rows.append({
            "证券代码": source["证券代码"].zfill(6), "时间": source["时间"],
            "证券简称": source["证券简称"], "证监会行业新": source["证监会行业新"],
            "X_实质性披露": x, "Y_象征性披露": y, "绿色披露总量": total,
            "GI": f"{gi:.10f}" if gi is not None else "",
            "唯一绿色关键词数": len(aggregate["words"]) if aggregate else 0,
            "唯一绿色句子数": len(aggregate["sentences"]) if aggregate else 0,
            "平均置信度": f"{mean(aggregate['confidences']):.6f}" if aggregate and aggregate["confidences"] else "",
            "失败记录数": failed, "完成状态": "complete" if complete else "incomplete",
            "E评级": esg_row.get("E评级", ""), "E得分": esg_row.get("E得分", ""),
        })

    valid = [row for row in gi_rows if row["完成状态"] == "complete" and row["E得分"] != ""]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[(str(row["时间"]), str(row["证监会行业新"]))].append(row)
    benchmark_rows: list[dict[str, Any]] = []
    benchmarks: dict[tuple[str, str], tuple[float, float, int]] = {}
    for group_key, rows in sorted(groups.items()):
        gi_mean = mean(float(row["GI"]) for row in rows)
        esge_mean = mean(float(row["E得分"]) for row in rows)
        benchmarks[group_key] = (gi_mean, esge_mean, len(rows))
        benchmark_rows.append({
            "时间": group_key[0], "证监会行业新": group_key[1], "行业样本数": len(rows),
            "GI_mean": f"{gi_mean:.10f}", "ESGE_mean": f"{esge_mean:.10f}",
            "小样本行业": int(len(rows) < 30),
        })

    final_rows: list[dict[str, Any]] = []
    for row in gi_rows:
        benchmark = benchmarks.get((str(row["时间"]), str(row["证监会行业新"])))
        eligible = row["完成状态"] == "complete" and row["E得分"] != "" and benchmark is not None
        gi_mean, esge_mean, industry_n = benchmark if benchmark else (None, None, 0)
        greenwashing = (
            int(float(row["GI"]) > gi_mean and float(row["E得分"]) < esge_mean)
            if eligible and gi_mean is not None and esge_mean is not None else ""
        )
        final_rows.append({
            **row, "行业样本数": industry_n,
            "GI_mean": f"{gi_mean:.10f}" if gi_mean is not None else "",
            "ESGE_mean": f"{esge_mean:.10f}" if esge_mean is not None else "",
            "Greenwashing": greenwashing,
            "Greenwashing构造状态": "complete" if eligible else "missing_input",
        })

    gi_fields = list(gi_rows[0])
    final_fields = list(final_rows[0])
    write_csv(OUT / "gi/company_year_gi.csv", gi_rows, gi_fields)
    write_csv(OUT / "industry_benchmarks/industry_year_benchmarks.csv", benchmark_rows, list(benchmark_rows[0]))
    write_csv(OUT / "final_panel/greenwashing_panel.csv", final_rows, final_fields)
    missing = [row for row in final_rows if row["Greenwashing构造状态"] != "complete"]
    write_csv(OUT / "reports/missing_or_incomplete.csv", missing, final_fields)
    summary = {
        "whitelist_samples": len(metadata), "gi_complete": sum(row["完成状态"] == "complete" for row in gi_rows),
        "gi_incomplete": sum(row["完成状态"] != "complete" for row in gi_rows),
        "greenwashing_complete": sum(row["Greenwashing构造状态"] == "complete" for row in final_rows),
        "greenwashing_missing": len(missing),
        "greenwashing_1": sum(row["Greenwashing"] == 1 for row in final_rows),
        "greenwashing_0": sum(row["Greenwashing"] == 0 for row in final_rows),
        "formula": "Greenwashing=1 iff GI > industry-year GI_mean and ESGE < industry-year ESGE_mean",
        "strict_inequalities": True,
    }
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
