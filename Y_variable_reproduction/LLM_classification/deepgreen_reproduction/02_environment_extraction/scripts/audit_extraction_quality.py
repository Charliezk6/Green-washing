from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "deepgreen_reproduction/02_environment_extraction"
MANIFEST = STAGE / "reports/extraction_manifest.csv"
SENTENCES = STAGE / "sentences"
S1 = STAGE / "s1/all_s1_words.csv"
REPORTS = STAGE / "reports/quality_audit"
KNOWN_FRAGMENTS = {"染物排", "水处", "气污", "能环", "排放控", "制标准", "人民共和国"}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    valid = [row for row in manifest if int(row["环境文本字符数"]) > 0]
    metrics: dict[str, dict[str, object]] = {}
    short_lengths = Counter()
    total_sentence_lengths: list[int] = []
    for index, row in enumerate(valid, start=1):
        key = f'{row["证券代码"].zfill(6)}_{row["时间"]}'
        lengths = []
        path = SENTENCES / f"{key}.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                sentence = str(json.loads(line)["sentence"])
                length = len(sentence.strip())
                lengths.append(length)
                total_sentence_lengths.append(length)
                if length <= 4:
                    short_lengths[length] += 1
        count = len(lengths)
        metrics[key] = {
            **row,
            "key": key,
            "实际句子数": count,
            "平均句长": sum(lengths) / count if count else 0,
            "中位句长": percentile([float(value) for value in lengths], 0.5),
            "短句数_不超过4字": sum(value <= 4 for value in lengths),
            "短句比例_不超过4字": sum(value <= 4 for value in lengths) / count if count else 0,
            "最长句": max(lengths) if lengths else 0,
        }
        if index % 1000 == 0:
            print(f"sentences_audited={index}/{len(valid)}", flush=True)

    fragment_counts = Counter()
    fragment_reports: dict[str, set[str]] = {word: set() for word in KNOWN_FRAGMENTS}
    with S1.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            word = row["word"].strip()
            if word in KNOWN_FRAGMENTS:
                fragment_counts[word] += 1
                fragment_reports[word].add(f'{row["证券代码"].zfill(6)}_{row["时间"]}')

    metric_rows = list(metrics.values())
    summary = {
        "samples": len(manifest),
        "valid": len(valid),
        "valid_share": len(valid) / len(manifest),
        "failed_or_empty": len(manifest) - len(valid),
        "total_sentences": len(total_sentence_lengths),
        "sentence_length_median": percentile([float(value) for value in total_sentence_lengths], 0.5),
        "sentence_length_p90": percentile([float(value) for value in total_sentence_lengths], 0.9),
        "sentence_length_p99": percentile([float(value) for value in total_sentence_lengths], 0.99),
        "sentences_at_most_4_chars": sum(short_lengths.values()),
        "sentences_at_most_4_chars_share": sum(short_lengths.values()) / len(total_sentence_lengths),
        "reports_over_20000_chars": sum(int(row["环境文本字符数"]) > 20000 for row in valid),
        "reports_over_50000_chars": sum(int(row["环境文本字符数"]) > 50000 for row in valid),
        "reports_over_2000_sentences": sum(int(row["句子数"]) > 2000 for row in valid),
        "reports_average_sentence_under_8": sum(float(row["平均句长"]) < 8 for row in metric_rows),
        "known_fragment_occurrences": dict(fragment_counts),
        "known_fragment_total": sum(fragment_counts.values()),
    }
    (REPORTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = list(metric_rows[0]) if metric_rows else []
    write_csv(REPORTS / "top_long_texts.csv", sorted(metric_rows, key=lambda row: int(row["环境文本字符数"]), reverse=True)[:100], fields)
    write_csv(REPORTS / "lowest_average_sentence_length.csv", sorted(metric_rows, key=lambda row: float(row["平均句长"]))[:100], fields)
    write_csv(REPORTS / "highest_short_sentence_share.csv", sorted(metric_rows, key=lambda row: float(row["短句比例_不超过4字"]), reverse=True)[:100], fields)
    write_csv(REPORTS / "short_valid_texts.csv", [row for row in metric_rows if int(row["环境文本字符数"]) < 100], fields)
    write_csv(
        REPORTS / "known_fragments.csv",
        [{"word": word, "occurrences": fragment_counts[word], "reports": len(fragment_reports[word])} for word in sorted(KNOWN_FRAGMENTS)],
        ["word", "occurrences", "reports"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
