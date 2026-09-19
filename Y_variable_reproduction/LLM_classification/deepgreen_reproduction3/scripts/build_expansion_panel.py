from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "deepgreen_reproduction3" / "reports" / "environment_social_responsibility_8982_clean_candidates.csv"
OUTPUT = ROOT / "deepgreen_reproduction3" / "01_sample_panel"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    source_rows = read_csv(SOURCE)
    by_key: dict[tuple[str, int], dict[str, str]] = {}
    duplicates: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    for row in source_rows:
        try:
            key = (row["证券代码"].zfill(6), int(row["时间"]))
        except (KeyError, ValueError):
            invalid.append(row)
            continue
        if key in by_key:
            duplicates.append(row)
        by_key[key] = row

    panel = []
    for key, row in sorted(by_key.items()):
        panel.append({
            "证券代码": key[0], "时间": key[1], "证券简称": row["证券简称"],
            "样本类型": row["样本类型"], "年报路径": row["年报路径"],
            "章节定位来源": row["定位来源"], "环境和社会责任章节标题": row["章节标题"],
            "章节标题位置": row["标题位置"], "章节字符数": row["章节字符数"],
            "实质绿色信号数": row["实质绿色信号数"], "章节证据摘录": row["章节证据摘录"],
        })

    fields = list(panel[0])
    write_csv(OUTPUT / "data" / "environment_social_responsibility_expansion_panel_2021_2024.csv", panel, fields)
    write_csv(OUTPUT / "reports" / "duplicate_candidate_keys.csv", duplicates, list(source_rows[0]))
    write_csv(OUTPUT / "reports" / "invalid_candidate_rows.csv", invalid, list(source_rows[0]))

    years = Counter(row["时间"] for row in panel)
    types = Counter(row["样本类型"] for row in panel)
    sources = Counter(row["章节定位来源"] for row in panel)
    summary = [
        {"指标": "输入候选记录数", "数值": len(source_rows)},
        {"指标": "第一步扩展面板数", "数值": len(panel)},
        {"指标": "主键唯一数", "数值": len(by_key)},
        {"指标": "重复主键记录数", "数值": len(duplicates)},
        {"指标": "无效记录数", "数值": len(invalid)},
        *({"指标": f"样本类型_{name}", "数值": count} for name, count in sorted(types.items())),
        *({"指标": f"章节定位来源_{name}", "数值": count} for name, count in sorted(sources.items())),
        *({"指标": f"样本_{year}", "数值": count} for year, count in sorted(years.items())),
    ]
    write_csv(OUTPUT / "reports" / "summary.csv", summary, ["指标", "数值"])
    print("\n".join(f"{row['指标']}: {row['数值']}" for row in summary))


if __name__ == "__main__":
    main()
