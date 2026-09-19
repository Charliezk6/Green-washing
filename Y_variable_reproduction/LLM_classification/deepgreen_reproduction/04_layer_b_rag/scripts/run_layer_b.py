from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[3]
INPUT_S2 = ROOT / "deepgreen_reproduction/03_layer_a/s2/all_s2.jsonl"
OUTPUT_ROOT = ROOT / "deepgreen_reproduction/04_layer_b_rag"
BASE_URL = "https://token.nau.edu.cn/v1"
MODEL = "qwen3.6-35b-a3b-fp8"
PROMPT_VERSION = "paper_appendix_b_zh_verbatim_no_rag_strict_pilot_v1"

HEADING = re.compile(
    r"^(?:上市公司及其子公司是否|环境保护相关政策|环境保护行政许可|行业排放标准|"
    r"对污染物的处理|防治污染设施|建设项目环境影响评价|突发环境事件|环境自行监测|"
    r"报告期内因环境问题|其他应当公开|其他环保)"
)
RETRIEVAL_CUE = re.compile(
    r"[《》]|(?:法|条例|标准|规范|办法|指南|导则)$|许可证|认证|体系|技术|工艺|"
    r"化学|反应|催化|吸附|脱硫|脱硝|氧化|还原|过滤|焚烧|生物处理"
)

SYSTEM_PROMPT = """你是一名外部 ESG 分析师，最关注公司财务报告 ESG 部分中的环境方面。

你非常熟悉公司 ESG 的各个维度，包括但不限于：绿色、环境保护、环境、清洁能源、节能、降碳、低碳、减排、污水处理、空气污染、公司环境标准、国家环境法规等。

你是一名诚实的分析师，不会编造材料。你将依次获得：（1）一个词语；（2）一个包含该词语的完整句子（该词语使用 `##` 标记）。你收到的词语主要分为以下四类：

（a）带有书名号的词语，通常是某一文件。你需要根据给定语境判断，该文件是否由公司自身发布、响应或制定，用于解决公司自身及其子公司的环境问题。例如，《XX公司环境项目评估》《XX公司污染防治方案》等文件，表明这是公司自愿披露其环境信息。但是，《环境保护法》《XX市污染治理要求》等文件由国家、政府及其他非公司主体发布，即使公司响应了该文件，也不属于公司自愿披露自身相关环境信息。

（b）化学物质或化学术语，例如“锰”“氧化物”“缩合反应”等。你需要根据给定语境判断，公司是否通过技术手段处理了该化学物质，或者是否在公司的生产经营过程中通过这种化学手段缓解了某一类环境问题。

（c）在语义上与环境、环境保护和绿色直接相关的词语，例如“环境保护部”“绿色生产”“碳排放”等。你需要根据给定语境判断，公司是否通过发布具体文件或采用具体技术方法对该词语所指事项作出了贡献，或者公司是否通过该词语所指事项缓解了其生产经营过程中的某一类环境问题。例如，“公司响应碳排放政策”只是一句没有实际行动的口号，而“公司响应碳排放政策并净化污染气体”则包含实际行动。

（d）其他通常与公司和工厂相关联的词语，例如“大烟囱”。你需要根据给定语境判断，公司是否实际且具体地处理了该污染性词语所指对象，或者是否实际且具体地促进了该有益于环境的词语所指事项。

补充判断要求：仅有理念、目标、规划、重视、推动、提升、持续改进、预案备案、监测数据公开或一般管理性描述，而未披露已经实施的具体项目、设施、技术、处理措施或明确完成的环境行动时，应判断为 0。

请严格遵循以下 JSON Schema 格式，仅返回纯 JSON 字符串，不要返回任何不必要的解释：
{\"judgment\": <int>, \"confidence\": <float>}
其中，judgment = 1 表示肯定，judgment = 0 表示否定；confidence 的取值范围为 0 到 1。"""


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def read_s2(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def retrieval_needed(row: dict[str, Any]) -> bool:
    return bool(RETRIEVAL_CUE.search(str(row["word"])))


def retrieval_query(row: dict[str, Any]) -> str:
    return f"{row['word']} 环境保护 技术 用途 法规 标准 性质"


def parse_response(content: str) -> tuple[int, float]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    match = re.search(r"\{[\s\S]*?\}", cleaned)
    if not match:
        raise ValueError("response contains no JSON object")
    payload = json.loads(match.group())
    value = payload.get("judgment")
    if isinstance(value, bool) or str(value) not in {"0", "1"}:
        raise ValueError("invalid judgment")
    confidence = payload.get("confidence")
    if isinstance(confidence, bool):
        raise ValueError("invalid confidence")
    confidence = float(confidence)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence out of range")
    return int(value), confidence


def retrieval_cache_path(query: str) -> Path:
    return OUTPUT_ROOT / "cache/retrieval" / f"{stable_hash(query)}.json"


def judgment_cache_path(row: dict[str, Any], evidence_hash: str) -> Path:
    digest = stable_hash(PROMPT_VERSION, MODEL, str(row["word"]), str(row["sentence"]), evidence_hash)
    return OUTPUT_ROOT / "cache/judgments" / f"{digest}.json"


async def retrieve(client: httpx.AsyncClient, row: dict[str, Any], tavily_key: str | None) -> dict[str, Any]:
    if not retrieval_needed(row):
        return {"status": "not_needed", "query": "", "evidence": []}
    query = retrieval_query(row)
    path = retrieval_cache_path(query)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not tavily_key:
        return {"status": "failed", "query": query, "evidence": [], "error": "TAVILY_API_KEY is missing"}
    try:
        response = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": tavily_key, "query": query, "search_depth": "basic", "max_results": 3},
        )
        response.raise_for_status()
        payload = response.json()
        evidence = [
            {"title": item.get("title", ""), "url": item.get("url", ""), "snippet": item.get("content", "")[:1200]}
            for item in payload.get("results", [])[:3]
        ]
        result = {"status": "success" if evidence else "no_results", "query": query, "evidence": evidence}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result
    except Exception as exc:
        return {"status": "failed", "query": query, "evidence": [], "error": f"{type(exc).__name__}: {exc}"}


async def judge(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, row: dict[str, Any], api_key: str,
    rag: dict[str, Any], retries: int,
) -> dict[str, Any]:
    evidence_text = "\n".join(
        f"- {item['title']} | {item['url']} | {item['snippet']}" for item in rag.get("evidence", [])
    ) or "无检索证据；仅依据年报句子判断。"
    evidence_hash = stable_hash(evidence_text)
    cache = judgment_cache_path(row, evidence_hash)
    if cache.exists():
        result = json.loads(cache.read_text(encoding="utf-8"))
        if result.get("status") == "success":
            result["from_cache"] = True
            return result
    user_prompt = f"目标词：{row['word']}\n完整句子：{row['marked_sentence']}\n检索证据：\n{evidence_text}"
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    error = ""
    for attempt in range(1, retries + 1):
        try:
            async with semaphore:
                response = await client.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"}, json=payload,
                )
            response.raise_for_status()
            response_json = response.json()
            content = response_json["choices"][0]["message"].get("content", "")
            judgment, confidence = parse_response(content)
            result = {
                **row, "judgment": judgment, "confidence": confidence, "status": "success",
                "rag_status": rag["status"], "retrieval_query": rag.get("query", ""),
                "evidence": rag.get("evidence", []), "attempts": attempt, "model": MODEL,
                "prompt_version": PROMPT_VERSION, "raw_response": content,
                "usage": response_json.get("usage", {}), "from_cache": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(min(30, 2 ** (attempt - 1) + random.random()))
    return {**row, "judgment": None, "confidence": None, "status": "failed", "rag_status": rag["status"], "error": error}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_tasks(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    mappings: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        task_key = (str(row["word"]), str(row["sentence"]))
        mappings.setdefault(task_key, []).append(row)
    tasks = []
    for task_id, ((word, sentence), occurrences) in enumerate(mappings.items(), start=1):
        representative = dict(occurrences[0])
        representative["task_id"] = task_id
        representative["occurrence_count"] = len(occurrences)
        representative["marked_sentence"] = sentence.replace(word, f"##{word}##")
        tasks.append(representative)
    return tasks, mappings


def expand_results(
    task_results: list[dict[str, Any]], mappings: dict[tuple[str, str], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for result in task_results:
        task_key = (str(result["word"]), str(result["sentence"]))
        judgment_fields = {
            key: value for key, value in result.items()
            if key not in {"key", "证券代码", "时间", "sentence_id", "marked_sentence"}
        }
        for occurrence in mappings[task_key]:
            expanded.append({**occurrence, **judgment_fields})
    return expanded


def prepare_pilot(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"heading": [], "rag": [], "plain": []}
    for row in rows:
        group = "heading" if HEADING.match(str(row["sentence"])) else "rag" if retrieval_needed(row) else "plain"
        groups[group].append(row)
    random.seed(20260717)
    selected: list[dict[str, Any]] = []
    allocation = {"heading": size // 5, "rag": size * 2 // 5, "plain": size - size // 5 - size * 2 // 5}
    for group, count in allocation.items():
        selected.extend(random.sample(groups[group], min(count, len(groups[group]))))
    return selected


async def run(rows: list[dict[str, Any]], api_key: str, concurrency: int, retries: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(120, connect=30)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for start in range(0, len(rows), max(50, concurrency * 5)):
            batch = rows[start:start + max(50, concurrency * 5)]
            rags = [{"status": "disabled", "query": "", "evidence": []} for _ in batch]
            judged = await asyncio.gather(*(judge(client, semaphore, row, api_key, rag, retries) for row, rag in zip(batch, rags)))
            results.extend(judged)
            print(f"processed={len(results)}/{len(rows)} success={sum(x['status'] == 'success' for x in results)}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepGreen Layer B substantive disclosure judgment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-pilot", type=int, default=0)
    parser.add_argument("--pilot-file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    rows = read_s2(args.pilot_file or INPUT_S2)
    tasks, task_mappings = unique_tasks(rows)
    if args.limit:
        tasks = tasks[:args.limit]
        selected_keys = {(str(row["word"]), str(row["sentence"])) for row in tasks}
        task_mappings = {key: value for key, value in task_mappings.items() if key in selected_keys}
    if args.prepare_pilot:
        pilot = prepare_pilot(tasks, args.prepare_pilot)
        path = OUTPUT_ROOT / "input/pilot_s2.jsonl"
        write_jsonl(path, pilot)
        print(f"pilot={len(pilot)} path={path}")
        return
    selected_occurrences = [item for occurrences in task_mappings.values() for item in occurrences]
    profile = {
        "source_pairs": len(rows), "selected_pairs": len(selected_occurrences),
        "unique_tasks": len(tasks), "api_calls_saved": len(selected_occurrences) - len(tasks),
        "unique_reports": len({row["key"] for row in selected_occurrences}),
        "unique_sentences": len({(row["key"], row["sentence_id"]) for row in selected_occurrences}),
        "heading_tasks": sum(bool(HEADING.match(str(row["sentence"]))) for row in tasks),
        "retrieval_needed_tasks": sum(retrieval_needed(row) for row in tasks),
    }
    (OUTPUT_ROOT / "input").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "input/input_summary.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(profile, ensure_ascii=False))
        return
    api_key = os.getenv("NAU_API_KEY")
    if not api_key:
        raise SystemExit("NAU_API_KEY is required")
    task_results = asyncio.run(run(tasks, api_key, args.concurrency, args.retries))
    write_jsonl(OUTPUT_ROOT / "results/unique_task_results.jsonl", task_results)
    results = expand_results(task_results, task_mappings)
    write_jsonl(OUTPUT_ROOT / "results/layer_b_results.jsonl", results)
    failures = [row for row in results if row["status"] != "success"]
    task_failures = [row for row in task_results if row["status"] != "success"]
    write_jsonl(OUTPUT_ROOT / "logs/failed_items.jsonl", failures)
    summary = {
        **profile, "successful_tasks": len(task_results) - len(task_failures), "failed_tasks": len(task_failures),
        "success": len(results) - len(failures), "failed": len(failures),
        "substantive_x": sum(row.get("judgment") == 1 for row in results),
        "symbolic_y": sum(row.get("judgment") == 0 for row in results),
        "rag_status": dict(Counter(row.get("rag_status", "") for row in task_results)),
        "model": MODEL, "prompt_version": PROMPT_VERSION,
    }
    (OUTPUT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "logs/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
