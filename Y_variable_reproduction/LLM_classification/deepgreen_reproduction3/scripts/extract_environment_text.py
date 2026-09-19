from __future__ import annotations

import csv
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parents[1]
PANEL = ROOT / "deepgreen_reproduction3/01_sample_panel/data/environment_social_responsibility_expansion_panel_2021_2024.csv"
OUTPUT = ROOT / "deepgreen_reproduction3/02_environment_extraction"
SOURCE = ROOT / "deepgreen_reproduction/02_environment_extraction/scripts/extract_environment_text.py"

spec = importlib.util.spec_from_file_location("original_environment_extraction", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(f"无法加载原始抽取工具：{SOURCE}")
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

NEXT_TOP_LEVEL = re.compile(r"(?m)^\s*第\s*[一二三四五六七八九十百0-9]{1,3}\s*节\s*[^\n]{0,120}$")
SOCIAL_END = [
    re.compile(r"(?m)^\s*[二2][、.．]?[ \t　]*社会责任(?:工作|履行|相关)?(?:具体)?(?:情况)?[ \t　]*$"),
    re.compile(r"(?m)^\s*[二2][、.．]?[ \t　]*(?:巩固拓展)?脱贫攻坚成果(?:同|与|、)?乡村振兴(?:的|等工作)?(?:具体)?情况[ \t　]*$"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def first_match(patterns: list[re.Pattern[str]], text: str, start: int, end: int) -> re.Match[str] | None:
    matches = [match for pattern in patterns if (match := pattern.search(text, start, end))]
    return min(matches, key=lambda match: match.start()) if matches else None


def extract_from_located_chapter(text: str, saved_start: int) -> dict[str, object]:
    # The saved position was computed from the original, unnormalized TXT. Select
    # the matching major heading nearest to that position; this avoids choosing the
    # earlier contents entry when a report repeats the title.
    candidates = [
        match for match in NEXT_TOP_LEVEL.finditer(text)
        if "环境" in match.group(0) and "社会责任" in match.group(0)
    ]
    if not candidates:
        return {"status": "failed", "rule": "saved_heading_not_found", "text": "", "start": -1, "end": -1, "removed": 0}
    chapter = min(candidates, key=lambda match: abs(match.start() - saved_start))
    if abs(chapter.start() - saved_start) > 5000:
        return {"status": "failed", "rule": "saved_heading_position_mismatch", "text": "", "start": -1, "end": -1, "removed": 0}

    next_chapter = NEXT_TOP_LEVEL.search(text, chapter.end())
    section_end = next_chapter.start() if next_chapter else len(text)
    start_candidates = []
    for name, pattern in original.ENV_START_PATTERNS:
        match = pattern.search(text, chapter.end(), section_end)
        if match:
            start_candidates.append((match.start(), name, match))
    start_match = min(start_candidates, key=lambda item: item[0])[2] if start_candidates else None
    start_rule = min(start_candidates, key=lambda item: item[0])[1] if start_candidates else "section_whole"
    content_start = start_match.end() if start_match else chapter.end()
    social_end = first_match(SOCIAL_END, text, content_start, section_end)
    content_end = social_end.start() if social_end else section_end
    cleaned, removed = original.clean_extracted_text(text[content_start:content_end])
    return {
        "status": "exact" if start_match else "chapter_whole",
        "rule": start_rule,
        "text": cleaned,
        "start": chapter.start(),
        "end": content_end,
        "removed": removed,
    }


def main() -> None:
    dirs = {name: OUTPUT / name for name in ("environment_text", "sentences", "s1", "reports", "resources")}
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    stopwords = original.load_stopwords(dirs["resources"] / "hit_stopwords.txt")

    manifest: list[dict[str, object]] = []
    sentence_rows: list[dict[str, object]] = []
    word_rows: list[dict[str, object]] = []
    for index, source in enumerate(read_csv(PANEL), start=1):
        code, year = source["证券代码"].zfill(6), source["时间"]
        key = f"{code}_{year}"
        record: dict[str, object] = {
            "证券代码": code, "时间": year, "证券简称": source["证券简称"], "样本类型": source["样本类型"],
            "年报路径": source["年报路径"], "章节定位来源": source["章节定位来源"],
            "已验证章节标题": source["环境和社会责任章节标题"], "已验证标题位置": source["章节标题位置"],
        }
        try:
            raw, encoding = original.read_text(PROJECT_ROOT / Path(source["年报路径"]))
            result = extract_from_located_chapter(raw, int(source["章节标题位置"]))
            environment_text = str(result["text"])
            fragmented, metrics = original.severely_fragmented(environment_text)
            no_disclosure = bool(original.NO_DISCLOSURE.fullmatch(environment_text.strip()))
            if fragmented:
                environment_text = ""
                result["status"] = "parse_failed"
                result["rule"] = "fragmented_or_mojibake"
            elif no_disclosure:
                environment_text = ""
                result["status"] = "no_disclosure"
                result["rule"] = "explicit_no_disclosure"
            sentences, excluded_short = original.split_sentences(environment_text) if environment_text else ([], 0)
            s1, token_count = original.make_s1(environment_text, stopwords) if environment_text else ([], 0)
            env_path = dirs["environment_text"] / f"{key}.txt"
            sentence_path = dirs["sentences"] / f"{key}.jsonl"
            s1_path = dirs["s1"] / f"{key}.json"
            env_path.write_text(environment_text, encoding="utf-8")
            with sentence_path.open("w", encoding="utf-8") as handle:
                for sentence_id, sentence in enumerate(sentences, start=1):
                    handle.write(json.dumps({"sentence_id": sentence_id, "sentence": sentence}, ensure_ascii=False) + "\n")
                    sentence_rows.append({"证券代码": code, "时间": year, "sentence_id": sentence_id, "sentence": sentence})
            s1_path.write_text(json.dumps({"证券代码": code, "时间": int(year), "s1": s1}, ensure_ascii=False, indent=2), encoding="utf-8")
            for word_id, word in enumerate(s1, start=1):
                word_rows.append({"证券代码": code, "时间": year, "word_id": word_id, "word": word})
            record.update({
                "读取编码": encoding, "抽取状态": result["status"], "抽取规则": result["rule"],
                "实际章节起点": result["start"], "内容终点": result["end"], "环境文本字符数": len(environment_text),
                "句子数": len(sentences), "排除无语境短句数": excluded_short,
                "删除表格行数": result["removed"], "抽取文本微小行比例": f"{metrics['tiny_line_share']:.4%}",
                "乱码标记数": metrics["mojibake_count"], "分词总数": token_count, "S1唯一词数": len(s1),
                "环境文本路径": env_path.relative_to(ROOT), "句子路径": sentence_path.relative_to(ROOT), "S1路径": s1_path.relative_to(ROOT),
                "异常信息": "" if environment_text else "未提取到有效环境文本",
            })
        except Exception as exc:
            record.update({
                "读取编码": "", "抽取状态": "error", "抽取规则": "exception", "实际章节起点": -1, "内容终点": -1,
                "环境文本字符数": 0, "句子数": 0, "排除无语境短句数": 0, "删除表格行数": 0,
                "抽取文本微小行比例": "0%", "乱码标记数": 0, "分词总数": 0, "S1唯一词数": 0,
                "环境文本路径": "", "句子路径": "", "S1路径": "", "异常信息": f"{type(exc).__name__}: {exc}",
            })
        manifest.append(record)
        if index % 500 == 0:
            print(f"processed={index}/{len(manifest)}", flush=True)

    fields = list(manifest[0])
    write_csv(dirs["reports"] / "extraction_manifest.csv", manifest, fields)
    write_csv(dirs["sentences"] / "all_sentences.csv", sentence_rows, ["证券代码", "时间", "sentence_id", "sentence"])
    write_csv(dirs["s1"] / "all_s1_words.csv", word_rows, ["证券代码", "时间", "word_id", "word"])
    status = Counter(str(row["抽取状态"]) for row in manifest)
    summary = [{"指标": "扩展样本数", "数值": len(manifest)}]
    summary.extend({"指标": f"抽取状态_{name}", "数值": count} for name, count in sorted(status.items()))
    for sample_type in ("制造业", "非制造业"):
        group = [row for row in manifest if row["样本类型"] == sample_type]
        valid = [row for row in group if int(row["环境文本字符数"]) > 0]
        summary.extend([{"指标": f"{sample_type}_样本数", "数值": len(group)}, {"指标": f"{sample_type}_有效文本数", "数值": len(valid)}])
    write_csv(dirs["reports"] / "summary.csv", summary, ["指标", "数值"])
    failed = [row for row in manifest if int(row["环境文本字符数"]) == 0]
    write_csv(dirs["reports"] / "failed_or_empty.csv", failed, fields)
    print(f"samples={len(manifest)} valid={len(manifest) - len(failed)} sentences={len(sentence_rows)} s1_words={len(word_rows)}")


if __name__ == "__main__":
    main()
