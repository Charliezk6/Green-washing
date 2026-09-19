from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
APPENDIX = ROOT / "appendix"
OUTPUT = ROOT / "deepgreen_reproduction2" / "01_sample_panel"
ESG_FILE = APPENDIX / "2.华证指数esg 评级评分（年度） 2009-2024(排序).csv"
YEARS = {2021, 2022, 2023, 2024}
A_SHARE_PATTERN = re.compile(r"^(?:00|001|002|003|30|60|68)\d{4}$")
FILE_PATTERN = re.compile(
    r"^(?P<code>\d{6})_(?P<year>20\d{2})_(?P<short_name>[^_]+)_(?P<title>.+)_(?P<publish_date>20\d{2}-\d{2}-\d{2})\.txt$"
)
REVISION_TERMS = ("更正", "修订", "更新", "补充")
ST_PATTERN = re.compile(r"(?:\*|＊)?ST|PT", re.IGNORECASE)
MANUFACTURING_INDUSTRIES = {
    "农副食品加工业", "食品制造业", "酒、饮料和精制茶制造业", "烟草制品业",
    "纺织业", "纺织服装、服饰业", "皮革、毛皮、羽毛及其制品和制鞋业",
    "木材加工和木、竹、藤、棕、草制品业", "家具制造业", "造纸和纸制品业",
    "印刷和记录媒介复制业", "文教、工美、体育和娱乐用品制造业",
    "石油、煤炭及其他燃料加工业", "化学原料和化学制品制造业", "医药制造业",
    "化学纤维制造业", "橡胶和塑料制品业", "非金属矿物制品业",
    "黑色金属冶炼和压延加工业", "有色金属冶炼和压延加工业", "金属制品业",
    "通用设备制造业", "专用设备制造业", "汽车制造业",
    "铁路、船舶、航空航天和其他运输设备制造业", "电气机械和器材制造业",
    "计算机、通信和其他电子设备制造业", "仪器仪表制造业", "其他制造业",
    "废弃资源综合利用业", "金属制品、机械和设备修理业",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error or RuntimeError(f"Unable to decode {path}")


def normalize_code(value: str) -> str:
    return re.sub(r"\D", "", value or "").zfill(6)


def is_non_manufacturing(industry: str) -> bool:
    normalized = (industry or "").strip()
    return bool(normalized) and normalized not in MANUFACTURING_INDUSTRIES


def parse_report(path: Path) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    match = FILE_PATTERN.match(path.name)
    if not match:
        return None, {"file_path": str(path.relative_to(ROOT)), "reason": "filename_parse_failed"}
    values = match.groupdict()
    year = int(values["year"])
    if year not in YEARS:
        return None, None
    publish_date = datetime.strptime(values["publish_date"], "%Y-%m-%d").date()
    title = values["title"]
    short_name = values["short_name"].strip()
    code = values["code"]
    return {
        "证券代码": code, "时间": year, "年报简称": short_name, "报告标题": title,
        "发布日期": publish_date.isoformat(), "修订标记": int(any(term in title for term in REVISION_TERMS)),
        "ST_PT标记": int(bool(ST_PATTERN.search(short_name))),
        "A股代码标记": int(bool(A_SHARE_PATTERN.fullmatch(code))), "年报路径": str(path.relative_to(ROOT)),
    }, None


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    esg_rows = read_csv_rows(ESG_FILE)
    esg_by_key: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in esg_rows:
        try:
            year = int(float(row["时间"]))
        except (KeyError, TypeError, ValueError):
            continue
        code = normalize_code(row.get("证券代码", ""))
        if year not in YEARS or not A_SHARE_PATTERN.fullmatch(code) or not is_non_manufacturing(row.get("证监会行业新", "")):
            continue
        normalized = dict(row)
        normalized["证券代码"], normalized["时间"] = code, str(year)
        esg_by_key[(code, year)].append(normalized)

    report_rows: list[dict[str, object]] = []
    parse_failures: list[dict[str, str]] = []
    for path in APPENDIX.glob("20??_*份/*.txt"):
        parsed, failure = parse_report(path)
        if parsed:
            report_rows.append(parsed)
        if failure:
            parse_failures.append(failure)

    reports_by_key: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in report_rows:
        reports_by_key[(str(row["证券代码"]), int(row["时间"]))].append(row)

    selected_reports: dict[tuple[str, int], dict[str, object]] = {}
    duplicate_log: list[dict[str, object]] = []
    for key, candidates in reports_by_key.items():
        ranked = sorted(candidates, key=lambda row: (str(row["发布日期"]), int(row["修订标记"]), str(row["年报路径"])), reverse=True)
        selected_reports[key] = ranked[0]
        if len(ranked) > 1:
            for rank, row in enumerate(ranked, 1):
                duplicate_log.append({"证券代码": key[0], "时间": key[1], "选择状态": "selected" if rank == 1 else "dropped", "排序序号": rank, **row})

    panel: list[dict[str, object]] = []
    unmatched_esg: list[dict[str, object]] = []
    duplicate_esg: list[dict[str, object]] = []
    for key, matches in sorted(esg_by_key.items()):
        if len(matches) > 1:
            duplicate_esg.extend(matches)
        esg, report = matches[-1], selected_reports.get(key)
        short_name = (esg.get("证券简称") or "").strip()
        excluded_st = bool(ST_PATTERN.search(short_name)) or bool(report and report["ST_PT标记"])
        record: dict[str, object] = {
            "证券代码": key[0], "时间": key[1], "证券简称": short_name,
            "证监会行业新": esg.get("证监会行业新", ""), "同花顺行业新": esg.get("同花顺行业新", ""),
            "申万行业": esg.get("申万行业", ""), "综合评级": esg.get("综合评级", ""),
            "综合得分": esg.get("综合得分", ""), "E评级": esg.get("E评级", ""), "E得分": esg.get("E得分", ""),
            "年报匹配": int(report is not None), "ST_PT排除": int(excluded_st),
        }
        if report:
            record.update({name: report[name] for name in ("年报简称", "报告标题", "发布日期", "修订标记", "年报路径")})
        else:
            record.update({"年报简称": "", "报告标题": "", "发布日期": "", "修订标记": "", "年报路径": ""})
            unmatched_esg.append(record)
        if report and not excluded_st:
            panel.append(record)

    reports_without_esg = [report for key, report in sorted(selected_reports.items()) if key not in esg_by_key and int(report["A股代码标记"]) == 1]
    panel_fields = ["证券代码", "时间", "证券简称", "证监会行业新", "同花顺行业新", "申万行业", "综合评级", "综合得分", "E评级", "E得分", "年报匹配", "ST_PT排除", "年报简称", "报告标题", "发布日期", "修订标记", "年报路径"]
    report_fields = ["证券代码", "时间", "年报简称", "报告标题", "发布日期", "修订标记", "ST_PT标记", "A股代码标记", "年报路径"]
    write_csv(OUTPUT / "data" / "non_manufacturing_panel_2021_2024.csv", panel, panel_fields)
    universe = [{"证券代码": key[0], "时间": key[1], "证券简称": matches[-1].get("证券简称", ""), "证监会行业新": matches[-1].get("证监会行业新", ""), "E评级": matches[-1].get("E评级", ""), "E得分": matches[-1].get("E得分", "")} for key, matches in sorted(esg_by_key.items())]
    write_csv(OUTPUT / "data" / "non_manufacturing_esg_universe_2021_2024.csv", universe, ["证券代码", "时间", "证券简称", "证监会行业新", "E评级", "E得分"])
    write_csv(OUTPUT / "reports" / "report_deduplication.csv", duplicate_log, ["证券代码", "时间", "选择状态", "排序序号", *report_fields[2:]])
    write_csv(OUTPUT / "reports" / "esg_without_report.csv", unmatched_esg, panel_fields)
    write_csv(OUTPUT / "reports" / "reports_without_non_manufacturing_esg.csv", reports_without_esg, report_fields)
    write_csv(OUTPUT / "reports" / "filename_parse_failures.csv", parse_failures, ["file_path", "reason"])
    write_csv(OUTPUT / "reports" / "duplicate_esg_keys.csv", duplicate_esg, list(esg_rows[0].keys()) if esg_rows else [])

    year_counts = Counter(int(row["时间"]) for row in panel)
    matched = sum(key in selected_reports for key in esg_by_key)
    st_excluded = sum(1 for key, matches in esg_by_key.items() if key in selected_reports and (ST_PATTERN.search(matches[-1].get("证券简称", "")) or selected_reports[key]["ST_PT标记"]))
    summary = [
        {"指标": "华证非制造业公司年度数（A股代码）", "数值": len(esg_by_key)},
        {"指标": "成功匹配年报数（含ST/PT）", "数值": matched},
        {"指标": "年报匹配率", "数值": f"{matched / len(esg_by_key):.4%}" if esg_by_key else ""},
        {"指标": "ST/PT排除数", "数值": st_excluded},
        {"指标": "最终非制造业面板数", "数值": len(panel)},
        {"指标": "最终主键唯一数", "数值": len({(row['证券代码'], row['时间']) for row in panel})},
        {"指标": "TXT文件名解析失败数", "数值": len(parse_failures)},
        {"指标": "重复华证主键行数", "数值": len(duplicate_esg)},
        *({"指标": f"最终样本_{year}", "数值": year_counts[year]} for year in sorted(YEARS)),
    ]
    write_csv(OUTPUT / "reports" / "summary.csv", summary, ["指标", "数值"])
    print("\n".join(f"{row['指标']}: {row['数值']}" for row in summary))


if __name__ == "__main__":
    main()
