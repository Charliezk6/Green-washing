from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "人工复核样本"
OUT = ROOT / "检验结果"
OUT.mkdir(exist_ok=True)


def read_rows(layer: str) -> list[dict[str, str]]:
    rows_by_id: dict[str, dict[str, str]] = {}
    for annotator in (1, 2, 3):
        path = SAMPLE_DIR / f"{layer}_人工{annotator}标注样本.csv"
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"{path} is empty")
        for row in rows:
            sample_id = row["sample_id"]
            label = row["人工标签"].strip()
            if label not in {"0", "1"}:
                raise ValueError(f"{path}: invalid label for {sample_id}: {label!r}")
            if sample_id not in rows_by_id:
                rows_by_id[sample_id] = {
                    "sample_id": sample_id,
                    "模型标签": row["模型标签"].strip(),
                    "模型confidence": row.get("模型confidence", ""),
                    "round": row.get("round", ""),
                }
            rows_by_id[sample_id][f"人工标签_{annotator}"] = label
    rows = list(rows_by_id.values())
    for row in rows:
        labels = [int(row[f"人工标签_{i}"]) for i in (1, 2, 3)]
        row["人工最终标签"] = str(int(sum(labels) >= 2))
        row["一致票数"] = str(max(labels.count(0), labels.count(1)))
        row["三人一致"] = str(int(len(set(labels)) == 1))
    return rows


def metrics(predicted: list[int], actual: list[int]) -> dict[str, float | int]:
    tp = sum(p == a == 1 for p, a in zip(predicted, actual))
    tn = sum(p == a == 0 for p, a in zip(predicted, actual))
    fp = sum(p == 1 and a == 0 for p, a in zip(predicted, actual))
    fn = sum(p == 0 and a == 1 for p, a in zip(predicted, actual))
    n = len(actual)
    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0
    return {"sample": n, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Accuracy": accuracy, "Precision": precision, "Recall": recall,
            "F1": f1, "MCC": mcc}


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n if n else 0
    p_a1 = sum(a) / n if n else 0
    p_b1 = sum(b) / n if n else 0
    pe = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def summarize(layer: str, rows: list[dict[str, str]]) -> dict[str, object]:
    model = [int(row["模型标签"]) for row in rows]
    final = [int(row["人工最终标签"]) for row in rows]
    annotator = {i: [int(row[f"人工标签_{i}"]) for row in rows] for i in (1, 2, 3)}
    result: dict[str, object] = {
        "layer": layer,
        "sample": len(rows),
        "model_vs_majority": metrics(model, final),
        "human_pairwise_kappa": {
            "annotator_1_vs_2": kappa(annotator[1], annotator[2]),
            "annotator_1_vs_3": kappa(annotator[1], annotator[3]),
            "annotator_2_vs_3": kappa(annotator[2], annotator[3]),
        },
        "human_pairwise_agreement": {
            "annotator_1_vs_2": sum(a == b for a, b in zip(annotator[1], annotator[2])) / len(rows),
            "annotator_1_vs_3": sum(a == b for a, b in zip(annotator[1], annotator[3])) / len(rows),
            "annotator_2_vs_3": sum(a == b for a, b in zip(annotator[2], annotator[3])) / len(rows),
        },
        "unanimous_rate": sum(row["三人一致"] == "1" for row in rows) / len(rows),
        "majority_vote_distribution": {"0": final.count(0), "1": final.count(1)},
        "model_distribution": {"0": model.count(0), "1": model.count(1)},
    }
    unanimous = [row for row in rows if row["三人一致"] == "1"]
    result["model_vs_unanimous"] = metrics(
        [int(row["模型标签"]) for row in unanimous],
        [int(row["人工最终标签"]) for row in unanimous],
    )
    return result


def write_outputs(layer: str) -> dict[str, object]:
    rows = read_rows(layer)
    fields = list(rows[0])
    with (OUT / f"{layer}_三人合并标签.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = summarize(layer, rows)
    (OUT / f"{layer}_检验结果.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def fmt_metrics(value: dict[str, object]) -> str:
    return " | ".join(f"{key}={value[key]}" for key in ("sample", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1", "MCC"))


def main() -> None:
    results = {layer: write_outputs(layer) for layer in ("LayerA", "LayerB")}
    lines = ["# DeepGreen 人工复核检验结果", "", "## 检验说明", "", "三名人工标注者对同一批样本分别进行独立标注。人工最终标签采用多数票：至少两人标记为 1，则最终标签为 1；否则为 0。模型性能同时以人工多数票和三人完全一致样本为参照。", "", "## Layer A", "", f"- 样本数：{results['LayerA']['sample']}", f"- 模型对人工多数票：`{fmt_metrics(results['LayerA']['model_vs_majority'])}`", f"- 三人完全一致样本中的模型指标：`{fmt_metrics(results['LayerA']['model_vs_unanimous'])}`", f"- 三人完全一致比例：{results['LayerA']['unanimous_rate']:.4f}", "- 人工两两 Cohen's κ："]
    for pair, value in results["LayerA"]["human_pairwise_kappa"].items():
        lines.append(f"  - {pair}: `{value:.4f}`")
    lines.extend(["", "## Layer B", "", f"- 样本数：{results['LayerB']['sample']}", f"- 模型对人工多数票：`{fmt_metrics(results['LayerB']['model_vs_majority'])}`", f"- 三人完全一致样本中的模型指标：`{fmt_metrics(results['LayerB']['model_vs_unanimous'])}`", f"- 三人完全一致比例：{results['LayerB']['unanimous_rate']:.4f}", "- 人工两两 Cohen's κ："])
    for pair, value in results["LayerB"]["human_pairwise_kappa"].items():
        lines.append(f"  - {pair}: `{value:.4f}`")
    lines.extend(["", "## 文件", "", "- `LayerA_三人合并标签.csv`：Layer A 三人标签、人工多数票和一致性字段。", "- `LayerB_三人合并标签.csv`：Layer B 三人标签、人工多数票和一致性字段。", "- `LayerA_检验结果.json`、`LayerB_检验结果.json`：完整机器可读结果。"])
    (OUT / "DeepGreen_人工复核检验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
