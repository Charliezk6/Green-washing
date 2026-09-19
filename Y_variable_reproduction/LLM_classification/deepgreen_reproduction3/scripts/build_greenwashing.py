from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LAYER_B = ROOT / "deepgreen_reproduction3/04_layer_b_rag/results/layer_b_results.jsonl"
EXPANSION_PANEL = ROOT / "deepgreen_reproduction3/01_sample_panel/data/environment_social_responsibility_expansion_panel_2021_2024.csv"
MANUFACTURING = ROOT / "deepgreen_reproduction/01_sample_panel/data/manufacturing_panel_2021_2024.csv"
NON_MANUFACTURING = ROOT / "deepgreen_reproduction2/01_sample_panel/data/non_manufacturing_panel_2021_2024.csv"
HISTORICAL_PANEL = ROOT.parents[1] / "Regression_reproduction/data/input/greenwashing_auditor_expertise_panel.csv"
OUT = ROOT / "deepgreen_reproduction3/05_variables"

# The historical regression panel supplies the official CSRC industry-code crosswalk.
# For non-manufacturing observations, benchmarks aggregate to the first code letter.
FALLBACK_PRIMARY_CODES = {
    "采矿业": "B", "农、林、牧、渔业": "A", "教育": "P",
    "广播、电视、电影和影视录音制作业": "R", "建筑装饰和其他建筑业": "E",
    "综合": "S", "石油加工、炼焦和核燃料加工业": "C", "社会工作": "Q",
    "装卸搬运和仓储业": "G", "非金属矿采选业": "B", "租赁业": "L",
    "货币金融服务": "J",
}


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


def main() -> None:
    expansion = {sample_key(row): row for row in read_csv(EXPANSION_PANEL)}
    source_rows = read_csv(MANUFACTURING) + read_csv(NON_MANUFACTURING)
    source = {sample_key(row): row for row in source_rows}
    historical_rows = read_csv(HISTORICAL_PANEL)
    secondary_code = {row["IndustryName"]: row["IndustryCode"] for row in historical_rows if row.get("IndustryName") and row.get("IndustryCode")}
    missing_metadata = sorted(set(expansion) - set(source))

    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"X": 0, "Y": 0, "confidences": [], "words": set(), "sentences": set(), "failed": 0}
    )
    with LAYER_B.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            aggregate = aggregates[str(item["key"])]
            if item.get("status") != "success" or item.get("judgment") not in (0, 1):
                aggregate["failed"] += 1
                continue
            aggregate["X" if item["judgment"] == 1 else "Y"] += 1
            aggregate["confidences"].append(float(item["confidence"]))
            aggregate["words"].add(str(item["word"]))
            aggregate["sentences"].add(int(item["sentence_id"]))

    gi_rows: list[dict[str, Any]] = []
    for key in sorted(expansion):
        expanded = expansion[key]
        metadata = source.get(key, {})
        aggregate = aggregates.get(key)
        x = int(aggregate["X"]) if aggregate else 0
        y = int(aggregate["Y"]) if aggregate else 0
        failed = int(aggregate["failed"]) if aggregate else 0
        total = x + y
        complete = aggregate is not None and failed == 0 and total > 0
        gi = x / total if complete else None
        industry = metadata.get("证监会行业新", "")
        industry_code = secondary_code.get(industry, "")
        if expanded["样本类型"] == "制造业":
            benchmark_group = industry_code or industry
            benchmark_level = "制造业二级"
        else:
            benchmark_group = industry_code[:1] or FALLBACK_PRIMARY_CODES.get(industry, "")
            benchmark_level = "非制造业一级"
        gi_rows.append({
            "证券代码": expanded["证券代码"].zfill(6), "时间": expanded["时间"],
            "证券简称": expanded["证券简称"], "样本类型": expanded["样本类型"],
            "证监会行业新": industry, "证监会行业代码": industry_code,
            "行业基准分组": benchmark_group, "行业基准层级": benchmark_level,
            "E评级": metadata.get("E评级", ""), "E得分": metadata.get("E得分", ""),
            "章节定位来源": expanded["章节定位来源"],
            "X_实质性披露": x, "Y_象征性披露": y, "绿色披露总量": total,
            "GI": f"{gi:.10f}" if gi is not None else "",
            "唯一绿色关键词数": len(aggregate["words"]) if aggregate else 0,
            "唯一绿色句子数": len(aggregate["sentences"]) if aggregate else 0,
            "平均置信度": f"{mean(aggregate['confidences']):.6f}" if aggregate and aggregate["confidences"] else "",
            "失败记录数": failed, "完成状态": "complete" if complete else "incomplete",
        })

    valid = [row for row in gi_rows if row["完成状态"] == "complete" and row["E得分"] and row["行业基准分组"]]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        groups[(row["时间"], row["行业基准分组"])].append(row)
    benchmarks: dict[tuple[str, str], tuple[float, float, int]] = {}
    benchmark_rows = []
    for group_key, rows in sorted(groups.items()):
        gi_mean = mean(float(row["GI"]) for row in rows)
        esge_mean = mean(float(row["E得分"]) for row in rows)
        gi_values = [float(row["GI"]) for row in rows]
        esge_values = [float(row["E得分"]) for row in rows]
        gi_sd = (sum((value - gi_mean) ** 2 for value in gi_values) / (len(rows) - 1)) ** 0.5 if len(rows) > 1 else 0.0
        esge_sd = (sum((value - esge_mean) ** 2 for value in esge_values) / (len(rows) - 1)) ** 0.5 if len(rows) > 1 else 0.0
        benchmarks[group_key] = (gi_mean, esge_mean, gi_sd, esge_sd, len(rows))
        benchmark_rows.append({"时间": group_key[0], "行业基准分组": group_key[1], "行业基准层级": rows[0]["行业基准层级"], "行业样本数": len(rows), "GI_mean": f"{gi_mean:.10f}", "ESGE_mean": f"{esge_mean:.10f}", "GI_sd": f"{gi_sd:.10f}", "ESGE_sd": f"{esge_sd:.10f}", "小样本行业": int(len(rows) < 30)})

    final_rows = []
    for row in gi_rows:
        benchmark = benchmarks.get((row["时间"], row["行业基准分组"]))
        eligible = row["完成状态"] == "complete" and row["E得分"] and benchmark is not None
        gi_mean, esge_mean, gi_sd, esge_sd, industry_n = benchmark if benchmark else (None, None, None, None, 0)
        greenwashing = int(float(row["GI"]) > gi_mean and float(row["E得分"]) < esge_mean) if eligible else ""
        continuous = ((float(row["GI"]) - gi_mean) / gi_sd - (float(row["E得分"]) - esge_mean) / esge_sd) if eligible and gi_sd and esge_sd else ""
        continuous_status = "complete" if eligible and gi_sd and esge_sd else "insufficient_industry_year_variation"
        final_rows.append({**row, "行业样本数": industry_n, "GI_mean": f"{gi_mean:.10f}" if gi_mean is not None else "", "ESGE_mean": f"{esge_mean:.10f}" if esge_mean is not None else "", "GI_sd": f"{gi_sd:.10f}" if gi_sd is not None else "", "ESGE_sd": f"{esge_sd:.10f}" if esge_sd is not None else "", "Greenwashing": greenwashing, "Greenwashing构造状态": "complete" if eligible else "missing_input", "Greenwashing连续": f"{continuous:.10f}" if continuous != "" else "", "连续变量构造状态": continuous_status})

    gi_fields = list(gi_rows[0])
    final_fields = list(final_rows[0])
    write_csv(OUT / "gi/company_year_gi.csv", gi_rows, gi_fields)
    write_csv(OUT / "industry_benchmarks/industry_year_benchmarks.csv", benchmark_rows, list(benchmark_rows[0]))
    write_csv(OUT / "final_panel/greenwashing_panel_expansion.csv", final_rows, final_fields)
    missing = [row for row in final_rows if row["Greenwashing构造状态"] != "complete"]
    write_csv(OUT / "reports/missing_or_incomplete.csv", missing, final_fields)
    write_csv(OUT / "reports/missing_metadata.csv", [{"key": key} for key in missing_metadata], ["key"])
    summary = {
        "expansion_samples": len(expansion), "metadata_matched": len(expansion) - len(missing_metadata),
        "metadata_missing": len(missing_metadata), "layer_b_keys": len(aggregates),
        "gi_complete": sum(row["完成状态"] == "complete" for row in gi_rows),
        "gi_incomplete": sum(row["完成状态"] != "complete" for row in gi_rows),
        "greenwashing_complete": sum(row["Greenwashing构造状态"] == "complete" for row in final_rows),
        "greenwashing_missing": len(missing), "greenwashing_1": sum(row["Greenwashing"] == 1 for row in final_rows),
        "greenwashing_0": sum(row["Greenwashing"] == 0 for row in final_rows),
        "benchmark_standard": "制造业按证监会二级行业代码；非制造业按证监会一级行业代码首字母；均在年度内计算均值",
        "formula": "Greenwashing=1 iff GI > industry-year GI_mean and ESGE < industry-year ESGE_mean",
        "strict_inequalities": True,
    }
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
