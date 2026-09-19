from __future__ import annotations

import asyncio
import csv
import json
import math
import os
import random
import re
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "deepgreen_reproduction/03_layer_a/results/layer_a_results.csv"
OUT = ROOT / "deepgreen_reproduction/03_layer_a/validation"
MODEL = "qwen3.6-35b-a3b-fp8"
BASE = "https://token.nau.edu.cn/v1"
ROUNDS = 10
PER_CLASS = 50
SEED = 20260717
PROMPT = """你是独立的ESG披露关键词人工复核员。不要参考、猜测或迎合其他模型的标签。现在只根据给定的孤立词语，判断它本身是否应作为公司年报环境信息中的绿色披露候选关键词。

标为1：词语本身明确指向环境保护、生态治理、污染及污染物、排放与减排、节能降碳、清洁或可再生能源、资源循环利用、废水/废气/固废/危废处理、环境监测、环保设施或明确的污染治理技术。明确属于污染物、工业废弃物或环保处理技术的专业词，也标为1，以便后续Layer B结合句子继续判断。
标为0：普通经营、财务、组织、时间、地点或动作词；仅在特定上下文中才可能与环保有关、但孤立词语本身没有明确环境含义的化学原料、产品、设备、法规通用词或模糊词。

边界原则：Layer A是候选词筛选，应兼顾召回率，但不能因为一个词“可能出现在环保句子中”就标1。若仅凭该词无法稳定判断其绿色披露属性，标0。
只返回纯JSON：{"label":0或1,"reason":"一句简短理由"}。"""


def parse(content: str) -> tuple[int, str]:
    match = re.search(r"\{[\s\S]*?\}", content)
    if not match:
        raise ValueError("response contains no JSON object")
    payload = json.loads(match.group())
    value = payload["label"]
    if isinstance(value, bool) or str(value) not in {"0", "1"}:
        raise ValueError("label must be 0 or 1")
    return int(value), str(payload.get("reason", ""))


async def audit_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    round_no: int,
    row: dict[str, str],
) -> dict[str, object]:
    async with semaphore:
        response = await client.post(
            BASE + "/chat/completions",
            headers={"Authorization": "Bearer " + api_key},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": PROMPT},
                    {"role": "user", "content": f"待复核词语：{row['word']}"},
                ],
                "temperature": 0,
                "max_tokens": 128,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    response.raise_for_status()
    message = response.json()["choices"][0]["message"]
    content = message.get("content") or message.get("reasoning_content") or ""
    label, reason = parse(content)
    model_label = int(row["judgment"])
    outcome = (
        "TP" if model_label == label == 1
        else "TN" if model_label == label == 0
        else "FP" if model_label == 1
        else "FN"
    )
    return {
        "round": round_no,
        "word": row["word"],
        "frequency": row["frequency"],
        "judgment": model_label,
        "confidence": row["confidence"],
        "audit_label": label,
        "audit_reason": reason,
        "outcome": outcome,
    }


def summarize(rows: list[dict[str, object]], round_no: int | str) -> dict[str, object]:
    tp = sum(row["outcome"] == "TP" for row in rows)
    tn = sum(row["outcome"] == "TN" for row in rows)
    fp = sum(row["outcome"] == "FP" for row in rows)
    fn = sum(row["outcome"] == "FN" for row in rows)
    total = len(rows)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0
    return {
        "round": round_no,
        "sample": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "disagreements": fp + fn,
    }


async def main() -> None:
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "success"]
    groups = {
        0: [row for row in rows if int(row["judgment"]) == 0],
        1: [row for row in rows if int(row["judgment"]) == 1],
    }
    samples: list[tuple[int, dict[str, str]]] = []
    for round_no in range(1, ROUNDS + 1):
        rng = random.Random(SEED + round_no - 1)
        sample = rng.sample(groups[0], PER_CLASS) + rng.sample(groups[1], PER_CLASS)
        rng.shuffle(sample)
        samples.extend((round_no, row) for row in sample)

    api_key = os.environ["NAU_API_KEY"]
    semaphore = asyncio.Semaphore(20)
    timeout = httpx.Timeout(120, connect=30)
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=20)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        audited = await asyncio.gather(
            *(audit_one(client, semaphore, api_key, round_no, row) for round_no, row in samples)
        )

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["round", "word", "frequency", "judgment", "confidence", "audit_label", "audit_reason", "outcome"]
    with (OUT / "simulated_manual_audit_10rounds.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audited)

    round_summaries = [
        summarize([row for row in audited if row["round"] == round_no], round_no)
        for round_no in range(1, ROUNDS + 1)
    ]
    metric_names = ["accuracy", "precision", "recall", "f1", "mcc"]
    aggregate: dict[str, object] = {
        "rounds": ROUNDS,
        "sample_per_round": PER_CLASS * 2,
        "sampling": "50 model-negative + 50 model-positive per round",
        "seed_start": SEED,
        "audit_model": MODEL,
        "round_results": round_summaries,
        "pooled": summarize(audited, "pooled"),
        "note": "AI-simulated independent audit; not human annotation",
    }
    for metric in metric_names:
        values = [float(row[metric]) for row in round_summaries]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
        aggregate[f"{metric}_mean"] = mean
        aggregate[f"{metric}_sd"] = sd
        aggregate[f"{metric}_min"] = min(values)
        aggregate[f"{metric}_max"] = max(values)

    (OUT / "simulated_manual_audit_10rounds_summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False))


asyncio.run(main())
