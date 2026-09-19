from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
DG = ROOT / "Y_variable_reproduction/LLM_classification/deepgreen_reproduction"
LAYER_A_RESULTS = DG / "03_layer_a/results/layer_a_results.csv"
WORDS = DG / "02_environment_extraction/s1/all_s1_words.csv"
SENTENCES = DG / "02_environment_extraction/sentences/all_sentences.csv"
LAYER_B_RESULTS = DG / "04_layer_b_rag/results/unique_task_results.jsonl"
SEED = 20260717


def load_layer_a_results() -> dict[str, dict[str, str]]:
    with LAYER_A_RESULTS.open(encoding="utf-8-sig", newline="") as f:
        return {
            row["word"]: row
            for row in csv.DictReader(f)
            if row["status"] == "success" and row["judgment"] in {"0", "1"}
        }


def load_contexts(valid_words: set[str]) -> dict[str, list[tuple[str, str, str]]]:
    sentences: dict[tuple[str, str], list[str]] = defaultdict(list)
    with SENTENCES.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sentences[(row["证券代码"].zfill(6), row["时间"])].append(row["sentence"])

    contexts: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    with WORDS.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            word = row["word"].strip()
            if word not in valid_words:
                continue
            code, year = row["证券代码"].zfill(6), row["时间"]
            for sentence in sentences.get((code, year), []):
                if word in sentence:
                    key = (word, code, year)
                    if key not in seen:
                        contexts[word].append((code, year, sentence))
                        seen.add(key)
                    break
    return contexts


def prepare_layer_a() -> None:
    results = load_layer_a_results()
    contexts = load_contexts(set(results))
    eligible = {label: [] for label in (0, 1)}
    for word, result in results.items():
        if contexts.get(word):
            eligible[int(result["judgment"])].append(word)

    rows: list[dict[str, object]] = []
    for round_no in range(1, 11):
        rng = random.Random(SEED + round_no - 1)
        selected = rng.sample(eligible[0], 50) + rng.sample(eligible[1], 50)
        rng.shuffle(selected)
        for position, word in enumerate(selected, 1):
            result = results[word]
            code, year, sentence = rng.choice(contexts[word])
            rows.append({
                "sample_id": f"A-R{round_no:02d}-{position:03d}",
                "round": round_no,
                "证券代码": code,
                "年份": year,
                "目标词": word,
                "完整原句": sentence.replace(word, f"##{word}##"),
                "人工标签": "",
                "简短判断依据": "",
                "模型标签": result["judgment"],
                "模型confidence": result["confidence"],
            })

    fields = list(rows[0])
    with (OUT / "LayerA_人工标注样本.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_layer_b() -> None:
    with LAYER_B_RESULTS.open(encoding="utf-8") as f:
        source = [json.loads(line) for line in f if line.strip()]
    groups = {
        label: [row for row in source if int(row["judgment"]) == label]
        for label in (0, 1)
    }
    rng = random.Random(SEED)
    selected = rng.sample(groups[0], 250) + rng.sample(groups[1], 250)
    rng.shuffle(selected)
    rows = []
    for position, row in enumerate(selected, 1):
        rows.append({
            "sample_id": f"B-{position:03d}",
            "关键词": row["word"],
            "完整原句": row.get("marked_sentence") or row["sentence"],
            "人工标签": "",
            "简短判断依据": "",
            "模型标签": row["judgment"],
            "模型confidence": row["confidence"],
        })
    fields = list(rows[0])
    with (OUT / "LayerB_人工标注样本.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    pairs = [(int(row["模型标签"]), int(row["人工标签"])) for row in rows]
    tp = sum(model == human == 1 for model, human in pairs)
    tn = sum(model == human == 0 for model, human in pairs)
    fp = sum(model == 1 and human == 0 for model, human in pairs)
    fn = sum(model == 0 and human == 1 for model, human in pairs)
    n = len(pairs)
    accuracy = (tp + tn) / n if n else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0
    return {"sample": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": f1, "mcc": mcc}


def calculate_if_labeled(filename: str, layer: str) -> None:
    path = OUT / filename
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    invalid = [row["sample_id"] for row in rows if row["人工标签"] not in {"0", "1"}]
    if invalid:
        print(f"{layer}: awaiting human labels ({len(invalid)} unlabeled rows)")
        return
    result: dict[str, object] = {"layer": layer, "pooled": metrics(rows)}
    if layer == "Layer A":
        result["rounds"] = [
            {"round": round_no, **metrics([row for row in rows if int(row["round"]) == round_no])}
            for round_no in range(1, 11)
        ]
    (OUT / f"{layer.replace(' ', '')}_检验结果.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    prepare_layer_a()
    prepare_layer_b()
    calculate_if_labeled("LayerA_人工标注样本.csv", "Layer A")
    calculate_if_labeled("LayerB_人工标注样本.csv", "Layer B")
