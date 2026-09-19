from __future__ import annotations

import argparse
import asyncio
import csv
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
import jieba

ROOT = Path(__file__).resolve().parents[3]
INPUT_WORDS = ROOT / "deepgreen_reproduction/02_environment_extraction/s1/all_s1_words.csv"
WHITELIST = ROOT / "deepgreen_reproduction/02_environment_extraction/reports/frozen_low_risk_samples.csv"
SENTENCES_DIR = ROOT / "deepgreen_reproduction/02_environment_extraction/sentences"
OUTPUT_ROOT = ROOT / "deepgreen_reproduction/03_layer_a"
BASE_URL = "https://token.nau.edu.cn/v1"
MODEL = "qwen3.6-35b-a3b-fp8"
PROMPT_VERSION = "paper_appendix_b_zh_v1"

SYSTEM_PROMPT = """你是一名外部 ESG 分析师，最关注公司财务报告 ESG 部分中的环境方面。
你非常熟悉公司 ESG 的各个维度，包括但不限于：绿色、环境保护、环境、清洁能源、节能、降碳、低碳、减排、污水处理、空气污染、公司环境标准、国家环境法规等。
你是一名诚实的分析师，不会编造材料。你需要判断给定词语是否为与公司绿色披露相关的关键词。
请严格遵循以下 JSON Schema 格式，仅返回纯 JSON 字符串，不要返回任何不必要的解释：
{"judgment": <int>, "confidence": <float>}
其中，judgment = 1 表示肯定，judgment = 0 表示否定；confidence 的取值范围为 0 到 1。"""


def read_whitelist_keys(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            f"{row['证券代码'].zfill(6)}_{row['时间']}"
            for row in csv.DictReader(handle)
        }


def read_words(path: Path, allowed_keys: set[str]) -> tuple[list[dict[str, Any]], list[str]]:
    occurrences: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['证券代码'].zfill(6)}_{row['时间']}"
            if key not in allowed_keys:
                continue
            word = row["word"].strip()
            if word:
                occurrences.append(row)
                counts[word] += 1
    words = sorted(counts, key=lambda item: (-counts[item], item))
    input_dir = OUTPUT_ROOT / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    with (input_dir / "unique_words.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "frequency"])
        writer.writeheader()
        writer.writerows({"word": word, "frequency": counts[word]} for word in words)
    return occurrences, words


def parse_response(content: str) -> tuple[int, float]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    match = re.search(r"\{[\s\S]*?\}", cleaned)
    if not match:
        raise ValueError("response contains no JSON object")
    payload = json.loads(match.group(0))
    judgment_raw = payload["judgment"]
    confidence_raw = payload["confidence"]
    if isinstance(judgment_raw, bool) or judgment_raw not in (0, 1, "0", "1"):
        raise ValueError("judgment must be exactly 0 or 1")
    if isinstance(confidence_raw, bool):
        raise ValueError("confidence must be numeric")
    judgment = int(judgment_raw)
    confidence = float(confidence_raw)
    if not 0 <= confidence <= 1:
        raise ValueError("confidence is out of range")
    return judgment, confidence


def cache_path(word: str) -> Path:
    digest = hashlib.sha256(f"{PROMPT_VERSION}\0{MODEL}\0{word}".encode("utf-8")).hexdigest()
    return OUTPUT_ROOT / "cache" / f"{digest}.json"


def load_cached(word: str) -> dict[str, Any] | None:
    path = cache_path(word)
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("word") == word and result.get("status") == "success":
            return result
    except (OSError, json.JSONDecodeError):
        return None
    return None


async def judge_word(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    word: str,
    retries: int,
) -> dict[str, Any]:
    cached = load_cached(word)
    if cached:
        cached["from_cache"] = True
        return cached
    request_payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"待判断词语：{word}"},
        ],
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            async with semaphore:
                response = await client.post(
                    f"{BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=request_payload,
                )
            response.raise_for_status()
            response_json = response.json()
            message = response_json["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            try:
                judgment, confidence = parse_response(content)
            except Exception as exc:
                raise ValueError(f"invalid model content: {content[:500]!r}") from exc
            result = {
                "word": word,
                "judgment": judgment,
                "confidence": confidence,
                "status": "success",
                "attempts": attempt,
                "model": MODEL,
                "prompt_version": PROMPT_VERSION,
                "raw_response": content,
                "usage": response_json.get("usage", {}),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "from_cache": False,
            }
            path = cache_path(word)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(min(30, 2 ** (attempt - 1) + random.random()))
    return {
        "word": word,
        "judgment": "",
        "confidence": "",
        "status": "failed",
        "attempts": retries,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "raw_response": "",
        "usage": {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "from_cache": False,
        "error": last_error,
    }


async def run_judgments(words: list[str], api_key: str, concurrency: int, retries: int) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(120, connect=30)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for start in range(0, len(words), max(concurrency * 5, 50)):
            batch = words[start : start + max(concurrency * 5, 50)]
            batch_results = await asyncio.gather(
                *(judge_word(client, semaphore, api_key, word, retries) for word in batch)
            )
            results.extend(batch_results)
            success = sum(result["status"] == "success" for result in results)
            print(f"processed={len(results)}/{len(words)} success={success}", flush=True)
    return results


def write_results(results: list[dict[str, Any]], frequencies: Counter[str]) -> None:
    results_dir = OUTPUT_ROOT / "results"
    logs_dir = OUTPUT_ROOT / "logs"
    dictionary_dir = OUTPUT_ROOT / "dictionary"
    for directory in (results_dir, logs_dir, dictionary_dir):
        directory.mkdir(parents=True, exist_ok=True)
    fields = [
        "word", "frequency", "judgment", "confidence", "status", "attempts", "model",
        "prompt_version", "from_cache", "created_at", "raw_response", "error",
    ]
    with (results_dir / "layer_a_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "frequency": frequencies[result["word"]]})
    with (results_dir / "layer_a_results.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps({**result, "frequency": frequencies[result["word"]]}, ensure_ascii=False) + "\n")
    failures = [result for result in results if result["status"] != "success"]
    with (logs_dir / "failed_words.jsonl").open("w", encoding="utf-8") as handle:
        for result in failures:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    green = [result for result in results if result["status"] == "success" and result["judgment"] == 1]
    with (dictionary_dir / "green_dictionary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "frequency", "confidence"])
        writer.writeheader()
        writer.writerows(
            {"word": result["word"], "frequency": frequencies[result["word"]], "confidence": result["confidence"]}
            for result in green
        )
    (dictionary_dir / "green_dictionary.json").write_text(
        json.dumps([result["word"] for result in green], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "unique_words": len(results),
        "success": len(results) - len(failures),
        "failed": len(failures),
        "green_words": len(green),
        "negative_words": sum(result["status"] == "success" and result["judgment"] == 0 for result in results),
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
    }
    (logs_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip()


def mark_target_word(sentence: str, word: str) -> str:
    return sentence.replace(word, f"##{word}##")


def exact_green_tokens(sentence: str, green_words: set[str]) -> list[str]:
    tokens = {token.strip() for token in jieba.cut(sentence, cut_all=False) if token.strip()}
    return sorted(tokens & green_words)


def build_s2(green_words: set[str], allowed_keys: set[str]) -> None:
    s2_dir = OUTPUT_ROOT / "s2"
    s2_dir.mkdir(parents=True, exist_ok=True)
    for old_path in s2_dir.glob("*.jsonl"):
        old_path.unlink()
    all_path = s2_dir / "all_s2.jsonl"
    total = 0
    reports = 0
    duplicate_pairs_removed = 0
    with all_path.open("w", encoding="utf-8") as all_handle:
        for sentence_path in sorted(SENTENCES_DIR.glob("*.jsonl")):
            key = sentence_path.stem
            if key not in allowed_keys:
                continue
            rows = []
            seen_pairs: set[tuple[str, str]] = set()
            with sentence_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    sentence = normalize_sentence(str(item["sentence"]))
                    for word in exact_green_tokens(sentence, green_words):
                        pair = (word, sentence)
                        if pair in seen_pairs:
                            duplicate_pairs_removed += 1
                            continue
                        seen_pairs.add(pair)
                        record = {
                            "key": key,
                            "证券代码": key[:6],
                            "时间": int(key[-4:]),
                            "sentence_id": item["sentence_id"],
                            "word": word,
                            "sentence": sentence,
                            "marked_sentence": mark_target_word(sentence, word),
                        }
                        rows.append(record)
                        all_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if rows:
                reports += 1
                total += len(rows)
                with (s2_dir / f"{key}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path = OUTPUT_ROOT / "logs" / "s2_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "reports_with_s2": reports,
                "keyword_sentence_pairs": total,
                "duplicate_pairs_removed": duplicate_pairs_removed,
                "matching_method": "jieba_exact_token_intersection",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepGreen Layer A green keyword classification")
    parser.add_argument("--limit", type=int, default=0, help="Only classify the first N unique words")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--build-s2", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate whitelist input without API calls")
    args = parser.parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.retries < 1:
        raise SystemExit("--retries must be at least 1")
    allowed_keys = read_whitelist_keys(WHITELIST)
    if not allowed_keys:
        raise SystemExit(f"Whitelist is empty: {WHITELIST}")
    occurrences, words = read_words(INPUT_WORDS, allowed_keys)
    if not words:
        raise SystemExit("No Layer A words were found for the whitelist samples")
    frequencies = Counter(row["word"].strip() for row in occurrences)
    if args.dry_run:
        print(
            f"whitelist_samples={len(allowed_keys)} word_occurrences={len(occurrences)} "
            f"unique_words={len(words)}"
        )
        return
    api_key = os.getenv("NAU_API_KEY")
    if not api_key:
        raise SystemExit("NAU_API_KEY environment variable is required")
    selected = words[: args.limit] if args.limit else words
    results = asyncio.run(run_judgments(selected, api_key, args.concurrency, args.retries))
    write_results(results, frequencies)
    if args.build_s2:
        if len(selected) != len(words):
            raise SystemExit("S2 cannot be built from a limited vocabulary run")
        failures = [result for result in results if result["status"] != "success"]
        if failures:
            raise SystemExit(f"S2 cannot be built while {len(failures)} Layer A words failed")
        green_words = {result["word"] for result in results if result["judgment"] == 1}
        build_s2(green_words, allowed_keys)


if __name__ == "__main__":
    main()
