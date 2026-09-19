from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import jieba


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PANEL = ROOT / "deepgreen_reproduction/01_sample_panel/data/manufacturing_panel_2021_2024.csv"
OUTPUT_ROOT = ROOT / "deepgreen_reproduction/02_environment_extraction"

SECTION_HEADING = re.compile(
    r"(?m)^\s*第\s*[五5]\s*节[ \t　]*(?:环境(?:和|与)?社会责任|环境、社会责任|环境保护与社会责任)[ \t　]*$"
)
PAGE_PREFIX = r"(?:\d{1,4}[ \t　]*)?"
NEXT_SECTION_HEADING = re.compile(
    rf"(?m)^\s*{PAGE_PREFIX}第\s*[六6]\s*节[ \t　]*[^\n]{{0,60}}$"
)
NEXT_ANY_SECTION_HEADING = re.compile(
    rf"(?m)^\s*{PAGE_PREFIX}第\s*(?:[二三四五六七八九十]|1[0-9]|[2-9])\s*节[ \t　]*[^\n]{{0,60}}$"
)
ENV_START_PATTERNS = [
    ("major_environment", re.compile(r"(?m)^\s*[一1]、?[ \t　]*重大环保问题[ \t　]*$")),
    ("environment_information", re.compile(r"(?m)^\s*[一1]、?[ \t　]*环境信息(?:情况)?[ \t　]*$")),
    ("environment_protection", re.compile(r"(?m)^\s*[一1]、?[ \t　]*环境保护(?:相关)?(?:情况|信息)?[ \t　]*$")),
]
SOCIAL_END_PATTERNS = [
    re.compile(
        r"(?m)^\s*[二2][、.．]?[ \t　]*社会责任(?:工作|履行|相关)?(?:具体)?(?:情况)?[ \t　]*$"
    ),
    re.compile(
        r"(?m)^\s*[二2][、.．]?[ \t　]*(?:巩固拓展)?脱贫攻坚成果(?:同|与|、)?乡村振兴(?:的|等工作)?(?:具体)?情况[ \t　]*$"
    ),
]
FALLBACK_ENV_HEADING = re.compile(
    r"(?m)^\s*(?:[一1]、?[ \t　]*)?(?:重大环保问题|环境信息(?:情况)?|环境保护(?:相关)?(?:情况|信息)?)[ \t　]*$"
)
PAGE_NO = re.compile(r"(?m)^\s*(?:第?\s*\d+\s*页(?:\s*共\s*\d+\s*页)?|\d{1,4})\s*$")
COMPANY_HEADER = re.compile(r"(?m)^\s*.{2,40}(?:股份有限公司|集团有限公司)\s*20(?:21|22|23|24)\s*年年度报告(?:全文)?\s*$")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])")
TOKEN_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9\-·（）()《》/\.]*$")
HEADING_LINE = re.compile(
    r"^(?:第\s*[一二三四五六七八九十0-9]+\s*节|[一二三四五六七八九十0-9]+[、.]|"
    r"上市公司及其子公司是否|环境保护|环境信息|重大环保|排污信息|防治污染设施|"
    r"建设项目环境影响评价|突发环境事件|环境自行监测|报告期内因环境问题|"
    r"参照重点排污单位|在报告期内为减少|未披露其他环境信息|其他环保)"
)
LIST_OR_TABLE_LINE = re.compile(
    r"^(?:[（(]?[一二三四五六七八九十0-9]+[）)、.]|√|□|单位[:：]|序号|公司名称|"
    r"主要污染物|排放口|执行的污染物排放标准)"
)
TEXTUAL_END = re.compile(r"[。！？!?；;：:，,、）)》]$")
TEXTUAL_START = re.compile(r"^[\u4e00-\u9fffA-Za-z《（(]")
CONTEXT_FREE_SHORT = re.compile(
    r"^(?:是|否|无|有|适用|不适用|正常|达标|未超标|废水|废气|噪声|排放|污染|污染物|"
    r"排污|环保|环境|监测|处理|治理|COD|氨氮|总磷|总氮|颗粒物|二氧化硫|氮氧化物)$",
    re.IGNORECASE,
)

# HIT 停用词表的常用核心项；项目资源目录中的 hit_stopwords.txt 可追加完整列表。
BUILTIN_STOPWORDS = {
    "的", "了", "和", "与", "及", "或", "等", "在", "是", "为", "对", "将", "由", "于", "中", "内",
    "本", "该", "其", "并", "且", "而", "但", "被", "以", "向", "从", "到", "有", "无", "不", "未",
    "公司", "企业", "相关", "进行", "通过", "以及", "根据", "按照", "包括", "主要", "进一步", "持续",
    "年度", "报告", "情况", "方面", "工作", "管理", "要求", "规定", "实现", "开展", "加强", "提高",
    "一个", "一种", "各项", "各类", "其中", "同时", "目前", "已经", "可以", "能够", "有效", "积极",
}


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def likely_toc(text: str, match: re.Match[str]) -> bool:
    line = match.group(0)
    neighborhood = text[max(0, match.start() - 200) : min(len(text), match.end() + 200)]
    return "...." in line or "……" in line or neighborhood.count(".....") >= 2 or neighborhood.count("……") >= 2


def heading_context_score(text: str, match: re.Match[str]) -> int:
    following = text[match.end() : min(len(text), match.end() + 2500)]
    score = 0
    score += 4 * len(re.findall(r"环境保护相关政策|重点排污单位|排污许可证|污染物排放", following))
    score += 3 * len(re.findall(r"防治污染设施|环境影响评价|突发环境事件|自行监测", following))
    score -= 5 * len(re.findall(r"单位[:：]?元|期末余额|期初余额|财务报表|人民币", following))
    return score


def choose_body_heading(text: str, pattern: re.Pattern[str]) -> re.Match[str] | None:
    matches = list(pattern.finditer(text))
    non_toc = [match for match in matches if not likely_toc(text, match)]
    candidates = non_toc or matches
    if not candidates:
        return None
    return max(candidates, key=lambda match: (heading_context_score(text, match), -match.start()))


def first_match(patterns: Iterable[re.Pattern[str]], text: str, start: int, end: int) -> re.Match[str] | None:
    candidates = []
    for pattern in patterns:
        match = pattern.search(text, start, end)
        if match:
            candidates.append(match)
    return min(candidates, key=lambda item: item.start()) if candidates else None


TABLE_HEADER_TERMS = re.compile(
    r"^(?:公司|子公司|单位|项目|指标|类别|名称|序号|污染物|主要污染物|排放口|排放方式|"
    r"排放浓度|排放总量|核定总量|执行标准|废气主要污染物排放量|废水排放量情况|"
    r"颗粒物(?:（吨）)?|二氧化硫(?:（吨）)?|氮氧化物(?:（吨）)?|VOCs(?:（吨）)?|"
    r"废水量(?:（m³）)?|化学需氧量(?:（吨）)?|氨氮(?:（吨）)?)$",
    re.IGNORECASE,
)
NUMERIC_TABLE_CELL = re.compile(r"^[\d,.%/—－-]+(?:吨|万元|万立方米|立方米|m[³3]|mg/L)?$")
PAGE_FRACTION = re.compile(r"^\d{1,4}\s*/\s*\d{1,4}$")
CHECKBOX_LINE = re.compile(r"^[√□☑☐]?\s*(?:适用|不适用)(?:\s+[√□☑☐]?\s*(?:适用|不适用))*$")
PUNCTUATION_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
MOJIBAKE = re.compile(r"(?:�|锟斤拷|Ⱦ����|ˮ��|���)")
NO_DISCLOSURE = re.compile(
    r"^(?:不适用|无|无相关情况|无相关披露|报告期内不存在|不属于重点排污单位)[。；;]?$"
)
TABLE_BLOCK_START = re.compile(
    r"^(?:行业排放标准及生产经营活动中涉及的污染物排放的具体情况|"
    r"公司或子公司名称.*(?:证书编号|主要污染物|特征污染物|处罚原因|整改措施)|"
    r"排污许可证编号(?:发证日期|有效期限)?|环保法规及标准名称(?:实施时间)?|"
    r"主要污染物及特征污染物的种类及名称|是否备案(?:备案机构|备案编号)*|"
    r"备案机构|备案编号|监测位点.*(?:监测项目|监测频次|执行标准|排放限值|分析方法|方法来源)|"
    r"监测频次.*(?:执行标准|排放限值|分析方法|方法来源)|处罚原因.*(?:违规情形|处罚结果|整改措施)|"
    r"建设项目环境影响评价报告书的批复|公司名称.*(?:是否备案|防治污染设施|监测方式|监测位点|"
    r"主要污染物|处罚原因|建设项目环境影响评价))$"
)
TABLE_BLOCK_END = re.compile(
    r"^(?:对污染物的处理|防治污染设施的建设和运行情况|建设项目环境影响评价|"
    r"突发环境事件应急预案|环境自行监测方案|报告期内因环境问题受到行政处罚|"
    r"其他应当公开的环境信息|重点排污单位之外的公司环保情况说明|其他环保相关信息)"
)
TABLE_SIGNAL = re.compile(
    r"(?:排放方式|排放口数量|排放浓度|排放总量|核定.*总量|超标排放情况|发证日期|"
    r"到期时间|证书编号|执行的污染物排放标准|主要污染物|特征污染物)"
)
ALWAYS_TABULAR_SECTION = re.compile(
    r"^(?:环境自行监测方案|报告期内因环境问题受到行政处罚的情况|其他应当公开的环境信息|"
    r"其他环保相关信息|建设项目环境影响评价报告书的批复)$"
)


def line_fragmentation(text: str) -> dict[str, float | int]:
    lines = [re.sub(r"\s+", "", line) for line in text.splitlines() if line.strip()]
    linguistic = [line for line in lines if re.search(r"[\u4e00-\u9fffA-Za-z]", line)]
    tiny = sum(len(re.findall(r"[\u4e00-\u9fffA-Za-z]", line)) <= 3 for line in linguistic)
    punctuation = sum(bool(PUNCTUATION_ONLY.fullmatch(line)) for line in lines)
    return {
        "line_count": len(lines),
        "tiny_line_share": tiny / len(linguistic) if linguistic else 0.0,
        "punctuation_line_share": punctuation / len(lines) if lines else 0.0,
        "mojibake_count": len(MOJIBAKE.findall(text)),
    }


def severely_fragmented(text: str) -> tuple[bool, dict[str, float | int]]:
    metrics = line_fragmentation(text)
    bad = bool(
        metrics["mojibake_count"] >= 3
        or (metrics["line_count"] >= 30 and metrics["tiny_line_share"] >= 0.55)
        or (metrics["line_count"] >= 20 and metrics["punctuation_line_share"] >= 0.20)
    )
    return bad, metrics


def keep_text_line(line: str) -> bool:
    """保留可形成自然语言语境的行，剔除 PDF 表格单元格和版式残留。"""
    line = line.strip()
    if not line or PUNCTUATION_ONLY.fullmatch(line):
        return False
    if PAGE_FRACTION.fullmatch(line) or CHECKBOX_LINE.fullmatch(line):
        return False
    if NUMERIC_TABLE_CELL.fullmatch(line) or TABLE_HEADER_TERMS.fullmatch(line):
        return False
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", line))
    alnum_count = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", line))
    if alnum_count and chinese_count / alnum_count < 0.2 and not HEADING_LINE.match(line):
        return False
    if chinese_count <= 7 and not HEADING_LINE.match(line) and not TEXTUAL_END.search(line):
        return False
    return True


def remove_table_blocks(text: str) -> tuple[str, int]:
    """删除结构已丢失的排放/许可表格块，保留表格前后的自然语言正文。"""
    kept: list[str] = []
    in_table = False
    removed = 0
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t　]+", " ", raw_line).strip()
        if not in_table and (TABLE_BLOCK_START.match(line) or (TABLE_SIGNAL.search(line) and len(line) > 25)):
            in_table = True
            removed += 1
            continue
        if in_table and TABLE_BLOCK_END.match(line):
            kept.append(line)
            # 某些标题之后固定跟随明细表；标题保留作语境，表体继续删除。
            in_table = bool(ALWAYS_TABULAR_SECTION.match(line))
            continue
        if in_table:
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


def clean_extracted_text(text: str) -> tuple[str, int]:
    text = PAGE_NO.sub("", text)
    text = COMPANY_HEADER.sub("", text)
    text, removed_table_lines = remove_table_blocks(text)
    # 表格块先剔除，再恢复由页面宽度造成的正常段落换行，避免跨列单元格被拼接成伪句。
    restored = restore_layout_lines(text)
    lines = []
    for raw_line in restored.splitlines():
        line = re.sub(r"[ \t　]+", " ", raw_line).strip()
        if keep_text_line(line):
            lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), removed_table_lines


def is_layout_wrap(previous: str, current: str) -> bool:
    if not previous or not current:
        return False
    if HEADING_LINE.match(previous) or HEADING_LINE.match(current):
        return False
    if LIST_OR_TABLE_LINE.match(previous) or LIST_OR_TABLE_LINE.match(current):
        return False
    if TEXTUAL_END.search(previous):
        return False
    if not TEXTUAL_START.match(current):
        return False
    chinese_previous = len(re.findall(r"[\u4e00-\u9fff]", previous))
    chinese_current = len(re.findall(r"[\u4e00-\u9fff]", current))
    compact_previous = chinese_previous == len(previous) and 2 <= len(previous) <= 6
    compact_current = chinese_current == len(current) and 2 <= len(current) <= 6
    if compact_previous and compact_current:
        return True
    return chinese_previous >= 4 and chinese_current >= 4


def restore_layout_lines(text: str) -> str:
    restored: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"[ \t　]+", " ", raw_line).strip()
        if not line:
            if restored and restored[-1] != "":
                restored.append("")
            continue
        if restored and restored[-1] and is_layout_wrap(restored[-1], line):
            restored[-1] += line
        else:
            restored.append(line)
    return "\n".join(restored).strip()


def extract_environment(text: str) -> dict[str, object]:
    normalized = normalize_text(text)
    section = choose_body_heading(normalized, SECTION_HEADING)
    if section:
        section_end_match = NEXT_SECTION_HEADING.search(normalized, section.end())
        section_end = section_end_match.start() if section_end_match else len(normalized)
        start_match = None
        start_rule = "section_whole"
        for name, pattern in ENV_START_PATTERNS:
            candidate = pattern.search(normalized, section.end(), section_end)
            if candidate and (start_match is None or candidate.start() < start_match.start()):
                start_match = candidate
                start_rule = name
        content_start = start_match.end() if start_match else section.end()
        end_match = first_match(SOCIAL_END_PATTERNS, normalized, content_start, section_end)
        content_end = end_match.start() if end_match else section_end
        extracted, removed_table_lines = clean_extracted_text(normalized[content_start:content_end])
        return {
            "status": "exact" if start_match else "fallback",
            "rule": start_rule,
            "section_start": section.start(),
            "content_start": content_start,
            "content_end": content_end,
            "removed_table_lines": removed_table_lines,
            "text": extracted,
        }

    fallback = choose_body_heading(normalized, FALLBACK_ENV_HEADING)
    if fallback:
        next_section = NEXT_ANY_SECTION_HEADING.search(normalized, fallback.end())
        fallback_end = next_section.start() if next_section else len(normalized)
        social = first_match(SOCIAL_END_PATTERNS, normalized, fallback.end(), fallback_end)
        content_end = social.start() if social else fallback_end
        extracted, removed_table_lines = clean_extracted_text(normalized[fallback.end():content_end])
        return {
            "status": "fallback",
            "rule": "environment_heading_without_section",
            "section_start": fallback.start(),
            "content_start": fallback.end(),
            "content_end": content_end,
            "removed_table_lines": removed_table_lines,
            "text": extracted,
        }

    return {
        "status": "failed", "rule": "no_heading", "section_start": -1, "content_start": -1,
        "content_end": -1, "removed_table_lines": 0, "text": "",
    }


def split_sentences(text: str) -> tuple[list[str], int]:
    sentences: list[str] = []
    excluded_context_free_short = 0
    restored = restore_layout_lines(text)
    for line in restored.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = SENTENCE_BOUNDARY.split(line)
        for raw in parts:
            sentence = re.sub(r"\s+", " ", raw).strip(" \t　")
            if not sentence or sentence.isdigit():
                continue
            if len(sentence) == 1 and not re.search(r"[\u4e00-\u9fffA-Za-z]", sentence):
                continue
            if len(sentence) <= 4 and CONTEXT_FREE_SHORT.fullmatch(sentence):
                excluded_context_free_short += 1
                continue
            sentences.append(sentence)
    return sentences, excluded_context_free_short


def load_stopwords(path: Path) -> set[str]:
    stopwords = set(BUILTIN_STOPWORDS)
    if path.exists():
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                stopwords.update(line.strip() for line in path.read_text(encoding=encoding).splitlines() if line.strip())
                break
            except UnicodeDecodeError:
                continue
    return stopwords


def make_s1(text: str, stopwords: set[str]) -> tuple[list[str], int]:
    tokens = []
    restored_text = restore_layout_lines(text).replace("\n", "。")
    for token in jieba.cut(restored_text, cut_all=False):
        token = token.strip()
        if not token or token in stopwords or len(token) < 2:
            continue
        if not TOKEN_PATTERN.match(token) or token.isdigit():
            continue
        tokens.append(token)
    return list(dict.fromkeys(tokens)), len(tokens)


def read_text(path: Path) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding), encoding
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace"), "utf-8-replace"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def is_low_risk(row: dict[str, object]) -> bool:
    return bool(
        row["抽取状态"] == "exact" and row["质量状态"] == "pass"
        and int(row["乱码标记数"]) == 0 and 100 <= int(row["环境文本字符数"]) <= 20000
        and int(row["句子数"]) <= 500
        and float(str(row["短句比例_不超过4字"]).rstrip("%")) <= 15
        and float(str(row["抽取文本微小行比例"]).rstrip("%")) <= 15
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="抽取制造业年报环境信息文本并构造 S1")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条；0 表示全量")
    parser.add_argument(
        "--review-only",
        action="store_true",
        help="冻结已通过保守质量标准的样本，仅重跑其余有效样本，并剔除失败或未定位样本",
    )
    args = parser.parse_args()

    output_dirs = {
        name: OUTPUT_ROOT / name for name in ("environment_text", "sentences", "s1", "reports", "resources")
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    stopwords = load_stopwords(output_dirs["resources"] / "hit_stopwords.txt")
    with args.panel.open("r", encoding="utf-8-sig", newline="") as handle:
        panel = list(csv.DictReader(handle))
    if args.limit:
        panel = panel[: args.limit]

    frozen_keys: set[str] = set()
    previous_manifest: dict[str, dict[str, str]] = {}
    if args.review_only:
        manifest_path = output_dirs["reports"] / "extraction_manifest.csv"
        whitelist_path = output_dirs["reports"] / "frozen_low_risk_samples.csv"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            previous_rows = list(csv.DictReader(handle))
        previous_manifest = {f"{row['证券代码'].zfill(6)}_{row['时间']}": row for row in previous_rows}
        if whitelist_path.exists():
            with whitelist_path.open("r", encoding="utf-8-sig", newline="") as handle:
                frozen_keys.update(
                    f"{row['证券代码'].zfill(6)}_{row['时间']}" for row in csv.DictReader(handle)
                )
        for key, row in previous_manifest.items():
            if is_low_risk(row):
                frozen_keys.add(key)
        panel = [row for row in panel if f"{row['证券代码'].zfill(6)}_{row['时间']}" not in frozen_keys]
        panel = [
            row for row in panel
            if (
                previous_manifest.get(f"{row['证券代码'].zfill(6)}_{row['时间']}", {}).get("抽取状态")
                in {"exact", "fallback"}
                or (
                    (output_dirs["environment_text"] / f"{row['证券代码'].zfill(6)}_{row['时间']}.txt").exists()
                    and (output_dirs["environment_text"] / f"{row['证券代码'].zfill(6)}_{row['时间']}.txt").stat().st_size > 0
                )
            )
        ]
        write_csv(
            output_dirs["reports"] / "frozen_low_risk_samples.csv",
            [previous_manifest[key] for key in sorted(frozen_keys)],
            list(previous_rows[0].keys()) if previous_rows else [],
        )

    manifest: list[dict[str, object]] = []
    sentence_rows: list[dict[str, object]] = []
    s1_rows: list[dict[str, object]] = []

    for index, row in enumerate(panel, start=1):
        code, year = row["证券代码"].zfill(6), row["时间"]
        report_path = ROOT / Path(row["年报路径"])
        key = f"{code}_{year}"
        record: dict[str, object] = {
            "证券代码": code,
            "时间": year,
            "证券简称": row["证券简称"],
            "证监会行业新": row["证监会行业新"],
            "年报路径": row["年报路径"],
            "环境文本路径": "",
            "句子路径": "",
            "S1路径": "",
        }
        try:
            raw_text, encoding = read_text(report_path)
            source_fragmented, source_metrics = severely_fragmented(raw_text)
            result = extract_environment(raw_text)
            environment_text = str(result["text"])
            extracted_fragmented, extracted_metrics = severely_fragmented(environment_text)
            parse_failure = source_fragmented or extracted_fragmented
            no_disclosure = bool(NO_DISCLOSURE.fullmatch(environment_text.strip()))
            if parse_failure:
                environment_text = ""
                result["status"] = "parse_failed"
                result["rule"] = "fragmented_or_mojibake"
            elif no_disclosure:
                environment_text = ""
                result["status"] = "no_disclosure"
                result["rule"] = "explicit_no_disclosure"
            sentences, excluded_short = split_sentences(environment_text) if environment_text else ([], 0)
            s1, token_count = make_s1(environment_text, stopwords) if environment_text else ([], 0)

            env_path = output_dirs["environment_text"] / f"{key}.txt"
            sentence_path = output_dirs["sentences"] / f"{key}.jsonl"
            s1_path = output_dirs["s1"] / f"{key}.json"
            env_path.write_text(environment_text, encoding="utf-8")
            with sentence_path.open("w", encoding="utf-8") as handle:
                for sentence_id, sentence in enumerate(sentences, start=1):
                    handle.write(json.dumps({"sentence_id": sentence_id, "sentence": sentence}, ensure_ascii=False) + "\n")
                    sentence_rows.append({"证券代码": code, "时间": year, "sentence_id": sentence_id, "sentence": sentence})
            s1_path.write_text(json.dumps({"证券代码": code, "时间": int(year), "s1": s1}, ensure_ascii=False, indent=2), encoding="utf-8")
            for word_id, word in enumerate(s1, start=1):
                s1_rows.append({"证券代码": code, "时间": year, "word_id": word_id, "word": word})

            record.update(
                {
                    "读取编码": encoding,
                    "抽取状态": result["status"],
                    "抽取规则": result["rule"],
                    "章节起点": result["section_start"],
                    "内容起点": result["content_start"],
                    "内容终点": result["content_end"],
                    "环境文本字符数": len(environment_text),
                    "句子数": len(sentences),
                    "排除无语境短句数": excluded_short,
                    "平均句长": f"{sum(map(len, sentences)) / len(sentences):.4f}" if sentences else "0",
                    "短句比例_不超过4字": f"{sum(len(item) <= 4 for item in sentences) / len(sentences):.4%}" if sentences else "0%",
                    "源文件微小行比例": f"{source_metrics['tiny_line_share']:.4%}",
                    "抽取文本微小行比例": f"{extracted_metrics['tiny_line_share']:.4%}",
                    "乱码标记数": int(source_metrics["mojibake_count"]) + int(extracted_metrics["mojibake_count"]),
                    "删除表格行数": int(result.get("removed_table_lines", 0)),
                    "质量状态": (
                        "failed" if parse_failure else "no_disclosure" if no_disclosure
                        else "review" if len(environment_text) > 50000 or len(sentences) > 2000
                        or (sentences and sum(map(len, sentences)) / len(sentences) < 8) else "pass"
                    ),
                    "分词总数": token_count,
                    "S1唯一词数": len(s1),
                    "环境文本路径": env_path.relative_to(ROOT),
                    "句子路径": sentence_path.relative_to(ROOT),
                    "S1路径": s1_path.relative_to(ROOT),
                    "异常信息": (
                        "TXT/PDF解析碎片化或乱码，已排除并保持缺失" if parse_failure
                        else "明确表示无环境信息披露" if no_disclosure
                        else "" if environment_text else "未提取到有效环境文本"
                    ),
                }
            )
        except Exception as exc:  # 单份年报失败不应中断全样本
            record.update(
                {
                    "读取编码": "",
                    "抽取状态": "error",
                    "抽取规则": "exception",
                    "章节起点": -1,
                    "内容起点": -1,
                    "内容终点": -1,
                    "环境文本字符数": 0,
                    "句子数": 0,
                    "排除无语境短句数": 0,
                    "平均句长": "0",
                    "短句比例_不超过4字": "0%",
                    "源文件微小行比例": "0%",
                    "抽取文本微小行比例": "0%",
                    "乱码标记数": 0,
                    "删除表格行数": 0,
                    "质量状态": "failed",
                    "分词总数": 0,
                    "S1唯一词数": 0,
                    "异常信息": f"{type(exc).__name__}: {exc}",
                }
            )
        manifest.append(record)
        if index % 500 == 0:
            print(f"processed={index}/{len(panel)}", flush=True)

    if args.review_only:
        retained_manifest = [previous_manifest[key] for key in sorted(frozen_keys)]
        manifest = retained_manifest + [
            row for row in manifest if row["抽取状态"] in {"exact", "fallback"} and int(row["环境文本字符数"]) > 0
        ]
        retained_keys = {f"{row['证券代码']}_{row['时间']}" for row in manifest}
        sentence_rows = []
        s1_rows = []
        for key in sorted(retained_keys):
            code, year = key.split("_")
            sentence_path = output_dirs["sentences"] / f"{key}.jsonl"
            if sentence_path.exists():
                for line in sentence_path.read_text(encoding="utf-8").splitlines():
                    item = json.loads(line)
                    sentence_rows.append({"证券代码": code, "时间": year, **item})
            s1_path = output_dirs["s1"] / f"{key}.json"
            if s1_path.exists():
                words = json.loads(s1_path.read_text(encoding="utf-8")).get("s1", [])
                for word_id, word in enumerate(words, start=1):
                    s1_rows.append({"证券代码": code, "时间": year, "word_id": word_id, "word": word})

    manifest_fields = [
        "证券代码", "时间", "证券简称", "证监会行业新", "年报路径", "读取编码", "抽取状态", "抽取规则",
        "章节起点", "内容起点", "内容终点", "环境文本字符数", "句子数", "排除无语境短句数",
        "平均句长", "短句比例_不超过4字", "源文件微小行比例", "抽取文本微小行比例", "乱码标记数",
        "删除表格行数", "质量状态", "分词总数", "S1唯一词数", "环境文本路径", "句子路径", "S1路径", "异常信息",
    ]
    write_csv(output_dirs["reports"] / "extraction_manifest.csv", manifest, manifest_fields)
    write_csv(output_dirs["sentences"] / "all_sentences.csv", sentence_rows, ["证券代码", "时间", "sentence_id", "sentence"])
    write_csv(output_dirs["s1"] / "all_s1_words.csv", s1_rows, ["证券代码", "时间", "word_id", "word"])

    status_counts = Counter(str(row["抽取状态"]) for row in manifest)
    rule_counts = Counter(str(row["抽取规则"]) for row in manifest)
    summary_rows = [{"指标": "样本总数", "数值": len(manifest)}]
    summary_rows.extend({"指标": f"抽取状态_{key}", "数值": value} for key, value in sorted(status_counts.items()))
    summary_rows.extend({"指标": f"抽取规则_{key}", "数值": value} for key, value in sorted(rule_counts.items()))
    for year in ("2021", "2022", "2023", "2024"):
        annual = [row for row in manifest if row["时间"] == year]
        success = [row for row in annual if row["抽取状态"] in {"exact", "fallback"} and int(row["环境文本字符数"]) > 0]
        summary_rows.append({"指标": f"{year}_样本数", "数值": len(annual)})
        summary_rows.append({"指标": f"{year}_有效文本数", "数值": len(success)})
    valid = [row for row in manifest if row["抽取状态"] in {"exact", "fallback"} and int(row["环境文本字符数"]) > 0]
    summary_rows.extend(
        [
            {"指标": "有效环境文本数", "数值": len(valid)},
            {"指标": "有效环境文本比例", "数值": f"{len(valid) / len(manifest):.4%}" if manifest else "0%"},
            {"指标": "总句子数", "数值": len(sentence_rows)},
            {"指标": "S1公司年度词条总数", "数值": len(s1_rows)},
        ]
    )
    write_csv(output_dirs["reports"] / "summary.csv", summary_rows, ["指标", "数值"])

    failures = [row for row in manifest if row["抽取状态"] in {"failed", "parse_failed", "error"} or int(row["环境文本字符数"]) == 0]
    write_csv(output_dirs["reports"] / "failed_or_empty.csv", failures, manifest_fields)
    short_texts = [row for row in valid if int(row["环境文本字符数"]) < 100]
    write_csv(output_dirs["reports"] / "short_text_review.csv", short_texts, manifest_fields)
    quality_review = [row for row in valid if row["质量状态"] == "review"]
    write_csv(output_dirs["reports"] / "quality_review.csv", quality_review, manifest_fields)

    if args.review_only:
        whitelist = [row for row in manifest if is_low_risk(row)]
        whitelist_keys = {f"{row['证券代码']}_{row['时间']}" for row in whitelist}
        pending_review = [
            row for row in valid if f"{row['证券代码']}_{row['时间']}" not in whitelist_keys
        ]
        write_csv(output_dirs["reports"] / "frozen_low_risk_samples.csv", whitelist, manifest_fields)
        write_csv(output_dirs["reports"] / "pending_quality_review.csv", pending_review, manifest_fields)

    print(
        f"samples={len(manifest)} valid={len(valid)} sentences={len(sentence_rows)} "
        f"s1_rows={len(s1_rows)} quality_review={len(quality_review)}"
    )


if __name__ == "__main__":
    main()
