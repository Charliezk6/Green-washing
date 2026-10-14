"""Share of substantive Layer B sentences that describe adverse environmental events."""

from __future__ import annotations

import json
import random
from collections import defaultdict

import pandas as pd

from common import (
    ADVERSE_TERMS,
    DERIVED,
    FIRM_YEAR_COUNTS,
    LAYER_B_PATHS,
    TABLES,
    adverse_hit,
    ensure_dirs,
    pad_code,
)


def main() -> None:
    ensure_dirs()
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0, 0])
    seen: set[tuple[str, int, str]] = set()
    conflicts = 0
    substantive = 0
    raw_adverse = 0
    adverse = 0
    symbolic = 0
    adverse_examples: list[dict[str, object]] = []
    clean_examples: list[dict[str, object]] = []
    adverse_seen = 0
    clean_seen = 0
    rng = random.Random(7)

    for path in LAYER_B_PATHS:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("status") != "success" or item.get("judgment") not in (0, 1):
                    continue
                code = pad_code(item.get("证券代码"))
                year = int(item["时间"])
                sentence_id = int(item.get("sentence_id", -1))
                word = str(item.get("word", ""))
                identity = (f"{code}_{year}", sentence_id, word)
                if identity in seen:
                    conflicts += 1
                    continue
                seen.add(identity)
                bucket = counts[(code, year)]
                if item["judgment"] == 0:
                    bucket[2] += 1
                    symbolic += 1
                    continue
                sentence = item.get("sentence", "")
                raw_flag = adverse_hit(sentence, drop_negation=False)
                flagged = adverse_hit(sentence, drop_negation=True)
                bucket[0] += 1
                substantive += 1
                if raw_flag:
                    raw_adverse += 1
                record = {"证券代码": code, "时间": year, "word": word, "sentence": sentence}
                if flagged:
                    bucket[1] += 1
                    adverse += 1
                    adverse_seen += 1
                    record["类型"] = "不利"
                    if len(adverse_examples) < 20:
                        adverse_examples.append(record)
                    else:
                        slot = rng.randrange(adverse_seen)
                        if slot < 20:
                            adverse_examples[slot] = record
                else:
                    clean_seen += 1
                    record["类型"] = "未命中不利词"
                    if len(clean_examples) < 20:
                        clean_examples.append(record)
                    else:
                        slot = rng.randrange(clean_seen)
                        if slot < 20:
                            clean_examples[slot] = record

    rows = [
        {
            "证券代码": code,
            "时间": year,
            "S": values[0],
            "S_adv": values[1],
            "P": values[2],
        }
        for (code, year), values in counts.items()
    ]
    firm_year = pd.DataFrame(rows)
    firm_year.to_csv(FIRM_YEAR_COUNTS, index=False, encoding="utf-8-sig")
    substantive_firms = int((firm_year["S"] > 0).sum())
    adverse_firms = int((firm_year["S_adv"] > 0).sum())
    share = pd.DataFrame([
        {
            "口径": "关键词—句子对，词表原命中",
            "分子": raw_adverse,
            "分母": substantive,
            "占比": raw_adverse / substantive if substantive else None,
            "象征性对子数": symbolic,
            "跨文件重复剔除数": conflicts,
            "不利词": "、".join(ADVERSE_TERMS),
        },
        {
            "口径": "关键词—句子对，剔除否定表述",
            "分子": adverse,
            "分母": substantive,
            "占比": adverse / substantive if substantive else None,
            "象征性对子数": symbolic,
            "跨文件重复剔除数": conflicts,
            "不利词": "、".join(ADVERSE_TERMS),
        },
        {
            "口径": "公司—年，剔除否定表述",
            "分子": adverse_firms,
            "分母": substantive_firms,
            "占比": adverse_firms / substantive_firms if substantive_firms else None,
            "象征性对子数": symbolic,
            "跨文件重复剔除数": conflicts,
            "不利词": "、".join(ADVERSE_TERMS),
        },
    ])
    share.to_csv(TABLES / "adverse_share.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(adverse_examples + clean_examples).to_csv(
        TABLES / "adverse_examples.csv", index=False, encoding="utf-8-sig"
    )
    DERIVED.joinpath("layer_b_read_summary.json").write_text(
        share.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(share.to_string(index=False))
    print(f"firm_years={len(firm_year)} conflicts={conflicts}")


if __name__ == "__main__":
    main()
