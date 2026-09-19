from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/reproduction_config.json").read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sample_key(row: dict[str, str]) -> str:
    return f"{row['证券代码'].zfill(6)}_{row['时间']}"


def sample_variance(values: list[float], center: float) -> float:
    return sum((value - center) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0


def main() -> None:
    metadata = {sample_key(row): row for row in read_csv(ROOT / CONFIG["sample_metadata"])}
    reference = {sample_key(row): row for row in read_csv(ROOT / CONFIG["reference_panel"])}
    aggregates: dict[str, dict[str, Any]] = defaultdict(lambda: {"X": 0, "Y": 0, "failed": 0, "confidence": [], "words": set(), "sentences": set()})
    with (ROOT / CONFIG["layer_b"]).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            aggregate = aggregates[str(item["key"])]
            if item.get("status") != "success" or item.get("judgment") not in (0, 1):
                aggregate["failed"] += 1
                continue
            aggregate["X" if item["judgment"] == 1 else "Y"] += 1
            aggregate["confidence"].append(float(item["confidence"]))
            aggregate["words"].add(str(item["word"]))
            aggregate["sentences"].add(int(item["sentence_id"]))

    rows: list[dict[str, Any]] = []
    for key in sorted(metadata):
        base, ref, aggregate = metadata[key], reference.get(key, {}), aggregates.get(key)
        x, y = (aggregate["X"], aggregate["Y"]) if aggregate else (0, 0)
        total = x + y
        complete = aggregate is not None and aggregate["failed"] == 0 and total > 0
        rows.append({
            "证券代码": base["证券代码"].zfill(6), "时间": base["时间"], "证券简称": base["证券简称"], "样本类型": base["样本类型"],
            "证监会行业新": ref.get("证监会行业新", ""), "证监会行业代码": ref.get("证监会行业代码", ""),
            "行业基准分组": ref.get("行业基准分组", ""), "行业基准层级": ref.get("行业基准层级", ""),
            "E评级": ref.get("E评级", ""), "E得分": ref.get("E得分", ""), "章节定位来源": base.get("章节定位来源", ""),
            "X_实质性披露": x, "Y_象征性披露": y, "绿色披露总量": total,
            "GI": x / total if complete else None, "唯一绿色关键词数": len(aggregate["words"]) if aggregate else 0,
            "唯一绿色句子数": len(aggregate["sentences"]) if aggregate else 0,
            "平均置信度": mean(aggregate["confidence"]) if aggregate and aggregate["confidence"] else None,
            "失败记录数": aggregate["failed"] if aggregate else 0, "完成状态": "complete" if complete else "incomplete",
        })

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["完成状态"] == "complete" and row["E得分"] and row["行业基准分组"]:
            groups[(row["时间"], row["行业基准分组"])].append(row)
    benchmarks: dict[tuple[str, str], tuple[float, float, float, float, int]] = {}
    benchmark_rows = []
    for group_key, group in sorted(groups.items()):
        gi_values, e_values = [float(row["GI"]) for row in group], [float(row["E得分"]) for row in group]
        gi_mean, e_mean = mean(gi_values), mean(e_values)
        gi_sd, e_sd = sample_variance(gi_values, gi_mean) ** .5, sample_variance(e_values, e_mean) ** .5
        benchmarks[group_key] = (gi_mean, e_mean, gi_sd, e_sd, len(group))
        benchmark_rows.append({"时间": group_key[0], "行业基准分组": group_key[1], "行业基准层级": group[0]["行业基准层级"], "行业样本数": len(group), "GI_mean": f"{gi_mean:.10f}", "ESGE_mean": f"{e_mean:.10f}", "GI_sd": f"{gi_sd:.10f}", "ESGE_sd": f"{e_sd:.10f}", "小样本行业": int(len(group) < 30)})

    for row in rows:
        benchmark = benchmarks.get((row["时间"], row["行业基准分组"]))
        eligible = row["完成状态"] == "complete" and benchmark is not None
        if not eligible:
            row.update({"行业样本数": 0, "GI_mean": None, "ESGE_mean": None, "GI_sd": None, "ESGE_sd": None, "Greenwashing": None, "Greenwashing构造状态": "missing_input", "Greenwashing连续": None, "连续变量构造状态": "insufficient_industry_year_variation"})
            continue
        gi_mean, e_mean, gi_sd, e_sd, n = benchmark
        gi, e_score = float(row["GI"]), float(row["E得分"])
        continuous = (gi - gi_mean) / gi_sd - (e_score - e_mean) / e_sd if gi_sd and e_sd else None
        row.update({"行业样本数": n, "GI_mean": gi_mean, "ESGE_mean": e_mean, "GI_sd": gi_sd, "ESGE_sd": e_sd, "Greenwashing": int(gi > gi_mean and e_score < e_mean), "Greenwashing构造状态": "complete", "Greenwashing连续": continuous, "连续变量构造状态": "complete" if continuous is not None else "insufficient_industry_year_variation"})

    fields = ["证券代码", "时间", "证券简称", "样本类型", "证监会行业新", "证监会行业代码", "行业基准分组", "行业基准层级", "E评级", "E得分", "章节定位来源", "X_实质性披露", "Y_象征性披露", "绿色披露总量", "GI", "唯一绿色关键词数", "唯一绿色句子数", "平均置信度", "失败记录数", "完成状态", "行业样本数", "GI_mean", "ESGE_mean", "GI_sd", "ESGE_sd", "Greenwashing", "Greenwashing构造状态", "Greenwashing连续", "连续变量构造状态"]
    write_csv(ROOT / CONFIG["output"], rows, fields)
    write_csv(ROOT / CONFIG["benchmark_output"], benchmark_rows, list(benchmark_rows[0]))
    summary = {"samples": len(rows), "layer_b_company_years": len(aggregates), "gi_complete": sum(row["完成状态"] == "complete" for row in rows), "greenwashing_complete": sum(row["Greenwashing构造状态"] == "complete" for row in rows), "greenwashing_1": sum(row["Greenwashing"] == 1 for row in rows), "greenwashing_0": sum(row["Greenwashing"] == 0 for row in rows)}
    reports = ROOT / "outputs/reports"; reports.mkdir(parents=True, exist_ok=True)
    (reports / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
