"""构造同年度、同行业无序公司对的会计师事务所联结自变量，并输出描述性统计。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PANEL = PROJECT / "Regression" / "data" / "input" / "combined_greenwashing_auditor_expertise_panel.csv"
CLEANING = ROOT / "data" / "cleaning"
PAIRS = ROOT / "data" / "pairs"
DESCRIPTIVE = ROOT / "descriptive"

EXPECTED_FIRM_YEARS = 18077
EXPECTED_PAIRS = 2051610
EXPECTED_CORRECTIONS = 194


def as_id(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "<NA>": np.nan})


def parse_auditors(value: object) -> frozenset[str] | None:
    if pd.isna(value):
        return None
    names = {part.strip() for part in str(value).split(",") if part.strip() and part.strip().lower() != "nan"}
    return frozenset(names) or None


def load_panel() -> pd.DataFrame:
    data = pd.read_csv(
        PANEL,
        dtype={"证券代码": str, "InstitutionID": str, "ExpertiseIndustryCode": str},
        encoding="utf-8-sig",
        low_memory=False,
    )
    data["证券代码"] = data["证券代码"].str.zfill(6)
    data["时间"] = pd.to_numeric(data["时间"], errors="raise").astype(int)
    data["InstitutionID"] = as_id(data["InstitutionID"])
    data["Big4"] = pd.to_numeric(data["Big4"], errors="coerce")
    data["DomesticTop10"] = pd.to_numeric(data["DomesticTop10"], errors="coerce")
    data["TerritoryAccountingFirm"] = data["TerritoryAccountingFirm"].fillna("").astype(str).str.strip()
    data["AccountingFirmName"] = data["AccountingFirmName"].fillna("").astype(str).str.strip()
    data["ExpertiseIndustryCode"] = data["ExpertiseIndustryCode"].fillna("").astype(str).str.strip()
    return data


def clean_firm_year(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    comparable = data["TerritoryAccountingFirm"].ne("") & data["AccountingFirmName"].ne("")
    matched = data.loc[comparable & data["TerritoryAccountingFirm"].eq(data["AccountingFirmName"])]
    mapping = matched.groupby("TerritoryAccountingFirm")["InstitutionID"].agg(lambda s: sorted(set(s.dropna())))
    if mapping.map(len).gt(1).any():
        conflict = mapping[mapping.map(len).gt(1)]
        raise ValueError(f"境内所名称对应多个机构代码: {conflict.to_dict()}")
    name_to_id = mapping.map(lambda values: values[0])

    territory_known = data["TerritoryAccountingFirm"].ne("")
    mapped = data["TerritoryAccountingFirm"].map(name_to_id)
    if territory_known.any() and mapped.loc[territory_known].isna().any():
        missing_names = sorted(data.loc[territory_known & mapped.isna(), "TerritoryAccountingFirm"].unique())
        raise ValueError(f"境内所名称无法映射: {missing_names}")

    cleaned = data.copy()
    cleaned["原InstitutionID"] = cleaned["InstitutionID"]
    cleaned["原AccountingFirmName"] = cleaned["AccountingFirmName"]
    cleaned["CleanInstitutionID"] = mapped.where(territory_known, cleaned["InstitutionID"])
    if cleaned["CleanInstitutionID"].isna().any():
        raise ValueError("清洗后仍有缺失的境内主审所代码")
    cleaned["CleanFirmName"] = cleaned["TerritoryAccountingFirm"].where(
        territory_known, cleaned["AccountingFirmName"]
    )

    firm_big4 = cleaned.dropna(subset=["Big4"]).groupby("CleanInstitutionID")["Big4"].agg(["nunique", "max"])
    if firm_big4["nunique"].gt(1).any():
        conflict = firm_big4.loc[firm_big4["nunique"].gt(1)]
        raise ValueError(f"同一境内所的四大标识不一致: {conflict.index.tolist()}")
    cleaned["Big4"] = cleaned["CleanInstitutionID"].map(firm_big4["max"]).astype(int)

    top10_by_year = {}
    for year in (2021, 2022, 2023):
        members = set(cleaned.loc[(cleaned["时间"] == year) & (cleaned["DomesticTop10"] == 1), "CleanInstitutionID"])
        if len(members) != 10:
            raise ValueError(f"{year} 年国内十大家数不是 10: {sorted(members)}")
        top10_by_year[year] = members
    top10_2023 = top10_by_year[2023]
    cleaned["DomesticTop10_raw"] = cleaned["DomesticTop10"]
    cleaned["DomesticTop10_filled"] = np.where(
        cleaned["时间"].eq(2024),
        cleaned["CleanInstitutionID"].isin(top10_2023),
        cleaned["DomesticTop10"].eq(1),
    ).astype(int)
    cleaned["DomesticTop10_source"] = np.where(cleaned["时间"].eq(2024), "carried_forward_2023", "observed")

    parsed = cleaned["Auditor"].map(parse_auditors)
    cleaned["签字师人数"] = parsed.map(lambda names: np.nan if names is None else len(names))
    cleaned["清洗标记"] = (
        cleaned["TerritoryAccountingFirm"].ne(cleaned["原AccountingFirmName"])
        | cleaned["CleanInstitutionID"].ne(cleaned["原InstitutionID"])
    ).astype(int)

    correction = cleaned.loc[cleaned["清洗标记"].eq(1), [
        "证券代码", "时间", "证券简称", "TerritoryAccountingFirm", "原AccountingFirmName",
        "原InstitutionID", "CleanInstitutionID", "CleanFirmName",
    ]].copy()
    correction["清洗方式"] = np.where(
        correction["TerritoryAccountingFirm"].eq(""),
        "境内所字段缺失，回退原机构代码",
        "按境内主审所名称映射",
    )
    correction = correction.sort_values(["时间", "证券代码"]).reset_index(drop=True)

    firm_year = cleaned[[
        "证券代码", "时间", "证券简称", "ExpertiseIndustryCode", "CleanInstitutionID", "CleanFirmName",
        "原InstitutionID", "原AccountingFirmName", "TerritoryAccountingFirm", "清洗标记", "Big4",
        "DomesticTop10_raw", "DomesticTop10_filled", "DomesticTop10_source", "Auditor", "签字师人数",
    ]].sort_values(["证券代码", "时间"]).reset_index(drop=True)
    firm_year.attrs["auditor_sets"] = {
        (code, year): names for code, year, names in zip(cleaned["证券代码"], cleaned["时间"], parsed)
    }
    firm_year.attrs["top10_2023"] = top10_2023
    return firm_year, correction, top10_2023


def build_static_pairs(firm_year: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (year, industry), group in firm_year.groupby(["时间", "ExpertiseIndustryCode"], sort=False):
        group = group.sort_values("证券代码", kind="mergesort")
        size = len(group)
        if size < 2:
            continue
        left, right = np.triu_indices(size, k=1)
        codes = group["证券代码"].to_numpy()
        firms = group["CleanInstitutionID"].to_numpy()
        big4 = group["Big4"].to_numpy()
        top10 = group["DomesticTop10_filled"].to_numpy()
        same = firms[left] == firms[right]
        if same.any() and np.any(big4[left][same] != big4[right][same]):
            raise ValueError(f"{year}-{industry} 同所公司对的四大标识不一致")
        if same.any() and np.any(top10[left][same] != top10[right][same]):
            raise ValueError(f"{year}-{industry} 同所公司对的十大标识不一致")
        frames.append(pd.DataFrame({
            "时间": np.full(len(left), year, dtype=np.int16),
            "ExpertiseIndustryCode": industry,
            "证券代码_i": codes[left],
            "证券代码_j": codes[right],
            "CleanInstitutionID_i": firms[left],
            "CleanInstitutionID_j": firms[right],
            "SameAF": same.astype(np.int8),
            "Big4_shared": np.where(same, big4[left], np.nan),
            "Top10_shared": np.where(same, top10[left], np.nan),
        }))
    pairs = pd.concat(frames, ignore_index=True)
    return pairs.sort_values(["时间", "ExpertiseIndustryCode", "证券代码_i", "证券代码_j"], kind="mergesort").reset_index(drop=True)


def attach_previous_firm(pairs: pd.DataFrame, firm_year: pd.DataFrame) -> pd.DataFrame:
    previous = firm_year[["证券代码", "时间", "CleanInstitutionID", "Big4", "DomesticTop10_filled"]].copy()
    previous["时间"] = previous["时间"] + 1
    previous = previous.rename(columns={
        "CleanInstitutionID": "PrevID",
        "Big4": "PrevBig4",
        "DomesticTop10_filled": "PrevTop10",
    })
    result = pairs.merge(
        previous.add_suffix("_i").rename(columns={"时间_i": "时间"}),
        on=["证券代码_i", "时间"], how="left", validate="m:1",
    )
    result = result.merge(
        previous.add_suffix("_j").rename(columns={"时间_j": "时间"}),
        on=["证券代码_j", "时间"], how="left", validate="m:1",
    )
    return result


def overlap_statistics(left: frozenset[str] | None, right: frozenset[str] | None) -> tuple[float, float, float, float, float]:
    if left is None or right is None:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)
    shared = len(left & right)
    total = len(left) + len(right)
    union = len(left | right)
    return (
        float(shared),
        float(shared > 0),
        float(left == right),
        shared / total,
        shared / union,
    )


def add_dynamic_and_auditor_variables(pairs: pd.DataFrame, firm_year: pd.DataFrame) -> pd.DataFrame:
    result = attach_previous_firm(pairs, firm_year)
    both_previous = result["PrevID_i"].notna() & result["PrevID_j"].notna()
    previous_same = both_previous & result["PrevID_i"].eq(result["PrevID_j"])
    previous_different = both_previous & ~previous_same
    current_same = result["SameAF"].eq(1)

    result["ChgCo"] = np.where(both_previous, (previous_different & current_same).astype(float), np.nan)
    result["ChgCo_FromDiff"] = np.where(previous_different, current_same.astype(float), np.nan)
    result["BreakCo"] = np.where(both_previous, (previous_same & ~current_same).astype(float), np.nan)

    linked_and_tracked = current_same & both_previous
    result["ChgBig4"] = np.where(
        linked_and_tracked,
        (previous_same & result["PrevBig4_i"].eq(0) & result["Big4_shared"].eq(1)).astype(float),
        np.nan,
    )
    result["ChgBig10"] = np.where(
        linked_and_tracked,
        (previous_same & result["PrevTop10_i"].eq(0) & result["Top10_shared"].eq(1)).astype(float),
        np.nan,
    )
    result["SameBig4"] = result["Big4_shared"]
    result["SameBig10"] = result["Top10_shared"]

    auditors = firm_year.attrs["auditor_sets"]
    current_left = [auditors[(code, year)] for code, year in zip(result["证券代码_i"], result["时间"])]
    current_right = [auditors[(code, year)] for code, year in zip(result["证券代码_j"], result["时间"])]
    previous_left = [auditors.get((code, year - 1)) for code, year in zip(result["证券代码_i"], result["时间"])]
    previous_right = [auditors.get((code, year - 1)) for code, year in zip(result["证券代码_j"], result["时间"])]

    shared = np.full(len(result), np.nan)
    any_shared = np.full(len(result), np.nan)
    same_team = np.full(len(result), np.nan)
    sim_rt = np.full(len(result), np.nan)
    sim_jaccard = np.full(len(result), np.nan)
    change_team = np.full(len(result), np.nan)
    change_rate = np.full(len(result), np.nan)

    previous_same_flags = previous_same.to_numpy()
    same_index = np.flatnonzero(current_same.to_numpy())
    for index in same_index:
        shared[index], any_shared[index], same_team[index], sim_rt[index], sim_jaccard[index] = overlap_statistics(
            current_left[index], current_right[index]
        )
        prior_names_left = previous_left[index]
        prior_names_right = previous_right[index]
        current_ready = current_left[index] is not None and current_right[index] is not None
        if prior_names_left is None or prior_names_right is None or not current_ready:
            continue
        if previous_same_flags[index]:
            _, _, prior_same_team, prior_sim, _ = overlap_statistics(prior_names_left, prior_names_right)
        else:
            prior_same_team, prior_sim = 0.0, 0.0
        change_team[index] = float(same_team[index] == 1.0 and prior_same_team == 0.0)
        change_rate[index] = float(sim_rt[index] > prior_sim)

    result["SharedCPACount"] = shared
    result["AnySharedAud"] = any_shared
    result["SameAud"] = same_team
    result["SimRt"] = sim_rt
    result["SimJaccard"] = sim_jaccard
    result["ChgAud"] = change_team
    result["ChgRt"] = change_rate
    return result


def validate(firm_year: pd.DataFrame, correction: pd.DataFrame, pairs: pd.DataFrame) -> None:
    if len(firm_year) != EXPECTED_FIRM_YEARS:
        raise AssertionError(f"公司—年度数 {len(firm_year)} 与预期 {EXPECTED_FIRM_YEARS} 不符")
    if firm_year.duplicated(["证券代码", "时间"]).any():
        raise AssertionError("公司—年度键重复")
    if firm_year["CleanInstitutionID"].isna().any() or firm_year["ExpertiseIndustryCode"].eq("").any():
        raise AssertionError("行业或境内所代码存在空值")
    if len(correction) != EXPECTED_CORRECTIONS:
        raise AssertionError(f"清洗对照 {len(correction)} 条，预期 {EXPECTED_CORRECTIONS} 条")
    if (correction["清洗方式"] == "境内所字段缺失，回退原机构代码").sum() != 1:
        raise AssertionError("回退原机构代码的记录不是 1 条")
    if len(pairs) != EXPECTED_PAIRS:
        raise AssertionError(f"公司对数 {len(pairs)} 与预期 {EXPECTED_PAIRS} 不符")
    expected_pairs = firm_year.groupby(["时间", "ExpertiseIndustryCode"]).size()
    expected_pairs = int((expected_pairs * (expected_pairs - 1) / 2).sum())
    if expected_pairs != len(pairs):
        raise AssertionError("分组配对公式与输出行数不一致")
    if pairs.duplicated(["时间", "ExpertiseIndustryCode", "证券代码_i", "证券代码_j"]).any():
        raise AssertionError("公司对键重复")
    if pairs["证券代码_i"].ge(pairs["证券代码_j"]).any():
        raise AssertionError("公司对不是按证券代码升序的无序对")
    if not set(pairs["SameAF"].unique()).issubset({0, 1}):
        raise AssertionError("SameAF 超出 0/1")
    linked = pairs["SameAF"].eq(1)
    for column in ("SameBig4", "SameBig10"):
        if pairs.loc[~linked, column].notna().any() or pairs.loc[linked, column].isna().any():
            raise AssertionError(f"{column} 的缺失范围与 SameAF 不一致")
    auditor_ready = linked & pairs["SharedCPACount"].notna()
    if pairs.loc[~linked, "SameAud"].notna().any():
        raise AssertionError("不同所公司对不应有签字师重合变量")
    if len(auditor_ready) and pairs.loc[auditor_ready, "SimRt"].gt(0.5).any():
        raise AssertionError("SimRt 超过 0.5")
    if pairs.loc[auditor_ready & pairs["SameAud"].eq(1), "SimJaccard"].ne(1).any():
        raise AssertionError("完全相同的签字师团队 Jaccard 应为 1")
    filled_2024 = set(firm_year.loc[(firm_year["时间"] == 2024) & (firm_year["DomesticTop10_filled"] == 1), "CleanInstitutionID"])
    if filled_2024 != firm_year.attrs["top10_2023"]:
        raise AssertionError("2024 年十大名单与 2023 年不一致")


def describe_variable(series: pd.Series, name: str, sample: str, positive: str) -> dict[str, object]:
    valid = series.dropna()
    hits = int((valid > 0).sum()) if positive == "positive" else int((valid == 1).sum())
    return {
        "变量": name,
        "适用样本": sample,
        "观测数": int(valid.shape[0]),
        "均值": valid.mean(),
        "标准差": valid.std(ddof=1),
        "最小值": valid.min(),
        "P25": valid.quantile(0.25),
        "中位数": valid.median(),
        "P75": valid.quantile(0.75),
        "最大值": valid.max(),
        "取1或大于0的配对数": hits,
        "占比": hits / len(valid) if len(valid) else np.nan,
    }


def conditional_mean(series: pd.Series) -> tuple[int, float]:
    valid = series.dropna()
    return int(len(valid)), (np.nan if valid.empty else float(valid.mean()))


def build_descriptives(firm_year: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    definitions = [
        ("SameAF", "全部同行业—年度无序公司对", "binary"),
        ("ChgCo", "配对双方在上一年都有公司—年度记录", "binary"),
        ("ChgCo_FromDiff", "配对双方在上一年都有记录，且上一年由不同事务所审计", "binary"),
        ("BreakCo", "配对双方在上一年都有公司—年度记录", "binary"),
        ("SameBig4", "当年由同一家事务所审计的公司对", "binary"),
        ("SameBig10", "当年由同一家事务所审计的公司对", "binary"),
        ("ChgBig4", "当年同所，且双方上一年都有记录", "binary"),
        ("ChgBig10", "当年同所，且双方上一年都有记录", "binary"),
        ("SharedCPACount", "当年同所，且双方当年签字师均非空", "positive"),
        ("AnySharedAud", "当年同所，且双方当年签字师均非空", "binary"),
        ("SameAud", "当年同所，且双方当年签字师均非空", "binary"),
        ("SimRt", "当年同所，且双方当年签字师均非空", "positive"),
        ("SimJaccard", "当年同所，且双方当年签字师均非空", "positive"),
        ("ChgAud", "当年同所，且双方当年和上一年签字师均非空", "binary"),
        ("ChgRt", "当年同所，且双方当年和上一年签字师均非空", "binary"),
    ]
    overall = pd.DataFrame([
        describe_variable(pairs[name], name, sample, positive) for name, sample, positive in definitions
    ])

    yearly_rows = []
    for year, group in pairs.groupby("时间"):
        firms = pd.unique(np.concatenate([group["证券代码_i"].to_numpy(), group["证券代码_j"].to_numpy()]))
        row = {
            "时间": int(year),
            "公司数": int(len(firms)),
            "行业数": int(group["ExpertiseIndustryCode"].nunique()),
            "总配对数": int(len(group)),
            "同所配对数": int(group["SameAF"].sum()),
        }
        for name in ("SameAF", "ChgCo", "SameBig4", "SameBig10", "AnySharedAud", "SameAud", "SimRt"):
            count, mean = conditional_mean(group[name])
            row[f"{name}观测数"] = count
            row[f"{name}均值"] = mean
        yearly_rows.append(row)
    by_year = pd.DataFrame(yearly_rows)

    firm_counts = firm_year.groupby(["ExpertiseIndustryCode", "时间"]).size().rename("公司数").reset_index()
    industry_base = firm_counts.groupby("ExpertiseIndustryCode").agg(出现年数=("时间", "nunique"), 年均公司数=("公司数", "mean"))
    pair_counts = pairs.groupby("ExpertiseIndustryCode").size().rename("累计配对数")
    same_counts = pairs.loc[pairs["SameAF"].eq(1)].groupby("ExpertiseIndustryCode").size().rename("同所配对数")
    same_mean = pairs.groupby("ExpertiseIndustryCode")["SameAF"].mean().rename("SameAF均值")
    linked = pairs.loc[pairs["SameAF"].eq(1)]
    overlap_rate = linked.groupby("ExpertiseIndustryCode")["AnySharedAud"].mean().rename("同所签字师重合率")
    sim_mean = linked.groupby("ExpertiseIndustryCode")["SimRt"].mean().rename("同所SimRt均值")
    by_industry = industry_base.join(pair_counts).join(same_counts).join(same_mean).join(overlap_rate).join(sim_mean)
    by_industry["同所配对数"] = by_industry["同所配对数"].fillna(0).astype(int)
    by_industry["占全样本配对比例"] = by_industry["累计配对数"] / len(pairs)
    by_industry = by_industry.reset_index().rename(columns={"ExpertiseIndustryCode": "行业代码"})
    by_industry = by_industry.sort_values(["累计配对数", "行业代码"], ascending=[False, True]).reset_index(drop=True)
    by_industry = by_industry[[
        "行业代码", "出现年数", "年均公司数", "累计配对数", "占全样本配对比例", "同所配对数",
        "SameAF均值", "同所签字师重合率", "同所SimRt均值",
    ]]

    names = firm_year.groupby("CleanInstitutionID")["CleanFirmName"].agg(lambda s: s.value_counts().index[0])
    aliases = firm_year.groupby("CleanInstitutionID")["CleanFirmName"].agg(
        lambda s: " / ".join(sorted(set(s) - {s.value_counts().index[0]}))
    )
    profile = firm_year.groupby("CleanInstitutionID").agg(
        公司年度数=("证券代码", "size"),
        是否四大=("Big4", "max"),
        是否2023年前十=("DomesticTop10_filled", "max"),
    )
    # 2024 carry-forward would mark a 2023 non-top firm as top if it only appears in 2024 and is on the 2023 list.
    # 是否2023年前十 uses the carried 2023 set, which is the intended roster.
    profile["是否2023年前十"] = firm_year.groupby("CleanInstitutionID")["CleanInstitutionID"].first().isin(
        firm_year.attrs["top10_2023"]
    ).astype(int)
    linked_counts = linked.groupby("CleanInstitutionID_i").size().rename("同行业同年度客户对数量")
    sharing = linked.groupby("CleanInstitutionID_i")["AnySharedAud"].mean().rename("所内签字师共享率")
    by_firm = profile.join(names.rename("事务所名称")).join(aliases.rename("曾用名称"))
    by_firm = by_firm.join(linked_counts).join(sharing)
    by_firm["同行业同年度客户对数量"] = by_firm["同行业同年度客户对数量"].fillna(0).astype(int)
    by_firm["占同所配对比重"] = by_firm["同行业同年度客户对数量"] / linked_counts.sum()
    by_firm = by_firm.reset_index().rename(columns={"CleanInstitutionID": "CleanInstitutionID"})
    by_firm = by_firm.sort_values(["同行业同年度客户对数量", "公司年度数"], ascending=False).reset_index(drop=True)
    by_firm = by_firm[[
        "CleanInstitutionID", "事务所名称", "曾用名称", "是否四大", "是否2023年前十", "公司年度数",
        "同行业同年度客户对数量", "占同所配对比重", "所内签字师共享率",
    ]]
    return {"overall": overall, "by_year": by_year, "by_industry": by_industry, "by_firm": by_firm}


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value * 100:.2f}%"


def number(value: float, digits: int = 4) -> str:
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def write_summary(firm_year: pd.DataFrame, correction: pd.DataFrame, pairs: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    overall = tables["overall"].set_index("变量")
    by_year = tables["by_year"]
    by_industry = tables["by_industry"]
    by_firm = tables["by_firm"]
    names = firm_year.groupby("CleanInstitutionID")["CleanFirmName"].agg(lambda s: s.value_counts().index[0])
    top10_names = "、".join(names.loc[sorted(firm_year.attrs["top10_2023"])].tolist())
    big4 = by_firm.loc[by_firm["是否四大"].eq(1), ["事务所名称", "同行业同年度客户对数量"]]
    top_industry = by_industry.iloc[0]
    mapped = int((correction["清洗方式"] == "按境内主审所名称映射").sum())
    fallback = int((correction["清洗方式"] == "境内所字段缺失，回退原机构代码").sum())

    def row(name: str) -> pd.Series:
        return overall.loc[name]

    year_lines = []
    for record in by_year.to_dict("records"):
        year_lines.append(
            f"| {record['时间']} | {record['公司数']:,} | {record['行业数']} | {record['总配对数']:,} | "
            f"{record['同所配对数']:,} | {percent(record['SameAF均值'])} | {percent(record['ChgCo均值'])} | "
            f"{percent(record['SameBig4均值'])} | {percent(record['SameBig10均值'])} | "
            f"{percent(record['AnySharedAud均值'])} | {percent(record['SameAud均值'])} | {number(record['SimRt均值'])} |"
        )

    text = f"""# 公司配对自变量说明

样本来自 `Regression/data/input/combined_greenwashing_auditor_expertise_panel.csv`。在清洗境内主审会计师事务所之后，按同一年度、同一 `ExpertiseIndustryCode` 构造无序公司对。行业口径沿用面板中的既有字段：制造业保留证监会两位代码，其他行业保留一位门类代码。

## 样本与清洗

公司—年度观测 {len(firm_year):,} 条，公司 {firm_year['证券代码'].nunique():,} 家，年度为 {firm_year['时间'].min()}–{firm_year['时间'].max()}。行业—年度单元 {firm_year.groupby(['时间', 'ExpertiseIndustryCode']).ngroups} 个，无序公司对 {len(pairs):,} 个。每个行业—年度至少两家公司，因此全部公司都进入配对。

境内主审所使用 `CleanInstitutionID`。`TerritoryAccountingFirm` 与 `AccountingFirmName` 不一致的记录有 {len(correction)} 条：{mapped} 条按境内主审所名称映射回境内机构代码，{fallback} 条境内所字段缺失、回退原机构代码（302132 中航成飞 2024 年，大信 `106423`）。映射同时合并更名事务所：公证天业 `107053`、北京大华国际/北京德皓国际 `10818013`、深圳大华国际/政旦志远 `10981942`。清洗后境内所 {firm_year['CleanInstitutionID'].nunique()} 家。

国际四大共 4 家，标识在 2021–2024 年完整。国内十大在 2021–2023 年各有 10 家；2024 年原始字段整年缺失，`DomesticTop10_filled` 沿用 2023 年名单，来源记为 `carried_forward_2023`。2023 年并沿用至 2024 年的十家为：{top10_names}。其中普华永道中天在 2022 年属于这十家，2023 年不在名单内，因此 2023 和 2024 年它只计入四大、不计入国内十大。

## 变量定义

公司对在每个 `时间 × ExpertiseIndustryCode` 内按证券代码升序生成，只保留 i < j。

- `SameAF`：两家 `CleanInstitutionID` 相同取 1，否则取 0。全样本 {len(pairs):,} 对。
- `ChgCo`：双方上一年都有记录时，上一年不同所且本年同所取 1，否则取 0。
- `ChgCo_FromDiff`：只在上一年不同所的公司对中定义，本年变为同所取 1，仍不同所取 0。
- `BreakCo`：双方上一年都有记录时，上一年同所且本年不同所取 1，否则取 0。
- `SameBig4`、`SameBig10`：只在 `SameAF = 1` 中定义。共同的事务所是四大或国内十大取 1，是同一家非四大或非十大取 0。
- `ChgBig4`、`ChgBig10`：当年同所且双方上一年都有记录时，由上一年同一家非四大（或非十大）改为本年同一家四大（或十大）取 1，否则取 0。
- `SharedCPACount`、`AnySharedAud`、`SameAud`、`SimRt`、`SimJaccard`：只在当年同所且双方签字师都非空时计算。`SimRt` 为共享人数除以双方人数之和，完全相同的团队取 0.5。`SimJaccard` 为共享人数除以并集人数，完全相同取 1。
- `ChgAud`、`ChgRt`：当年同所且双方当年、上一年签字师都非空。上一年不同所时，上一年的同所重合度视为 0。`ChgAud` 表示签字师团队由不完全相同变为完全相同。`ChgRt` 表示 `SimRt` 高于上一年。

不同所公司对上，规模变量和签字师变量留空，不把“不同所”编成 0。

## 描述性结果

`SameAF` 均值为 {percent(row('SameAF')['均值'])}，同所公司对 {int(row('SameAF')['取1或大于0的配对数']):,} 个。可追踪到上一年的公司对中，`ChgCo` 均值为 {percent(row('ChgCo')['均值'])}（{int(row('ChgCo')['观测数']):,} 对），由不同所变为同所的条件概率 `ChgCo_FromDiff` 为 {percent(row('ChgCo_FromDiff')['均值'])}。`BreakCo` 均值为 {percent(row('BreakCo')['均值'])}。

同所公司对里，`SameBig4` 为 {percent(row('SameBig4')['均值'])}，`SameBig10` 为 {percent(row('SameBig10')['均值'])}。签字师至少重合一人的比例 `AnySharedAud` 为 {percent(row('AnySharedAud')['均值'])}，团队完全相同的比例 `SameAud` 为 {percent(row('SameAud')['均值'])}，`SimRt` 均值为 {number(row('SimRt')['均值'])}，`SimJaccard` 均值为 {number(row('SimJaccard')['均值'])}。

| 年度 | 公司数 | 行业数 | 总配对数 | 同所配对数 | SameAF | ChgCo | SameBig4 | SameBig10 | AnySharedAud | SameAud | SimRt |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(year_lines)}

2021 年没有上一年，`ChgCo` 为空。`SameBig4` 及以后各列是当年同所子样本的均值。配对数最多的行业是 {top_industry['行业代码']}，累计 {int(top_industry['累计配对数']):,} 对，占全部公司对的 {percent(top_industry['占全样本配对比例'])}，该行业 `SameAF` 为 {percent(top_industry['SameAF均值'])}。

同所客户对最多的五家事务所：

| 事务所 | 公司—年度 | 同所客户对 | 占同所配对 | 签字师共享率 |
| --- | ---: | ---: | ---: | ---: |
"""
    firm_lines = []
    for record in by_firm.head(5).to_dict("records"):
        firm_lines.append(
            f"| {record['事务所名称']} | {int(record['公司年度数']):,} | {int(record['同行业同年度客户对数量']):,} | "
            f"{percent(record['占同所配对比重'])} | {percent(record['所内签字师共享率'])} |"
        )
    big4_text = "、".join(f"{name}（{int(count):,} 对）" for name, count in big4.itertuples(index=False))
    text += "\n".join(firm_lines)
    text += f"""

四大形成的同行业—年度客户对为：{big4_text}。

## 与配对文献的对照

田高良、陈匡宇、齐保垒（2021，《会计研究》）在同一年度、同一行业内穷尽配对，制造业保留两位代码、其他行业保留一位代码，并用 `SameAuditor` 区分同所与不同所。本表的行业键和 `SameAF` 与该设计一致。他们 2016–2018 年关键审计事项样本有 1,767,801 对；本样本为 2021–2024 年环境披露面板，共 {len(pairs):,} 对。

张勇、殷健（2022，《审计研究》）的 `SameAF` 均值为 6.03%，`ChgCo` 均值为 1.19%；同所子样本中 `SameBig4` 为 1.48%，`SameBig10` 为 80.87%，`SameAud` 为 2.36%，`SimRt` 为 0.0192。本样本对应数值为 `SameAF` {percent(row('SameAF')['均值'])}、`ChgCo` {percent(row('ChgCo')['均值'])}、`SameBig4` {percent(row('SameBig4')['均值'])}、`SameBig10` {percent(row('SameBig10')['均值'])}、`SameAud` {percent(row('SameAud')['均值'])}、`SimRt` {number(row('SimRt')['均值'])}。样本期、行业分类版本和审计市场集中度都不同，数值只说明本次构造的量级，不拿来复现他们的回归。

## 输出文件

- `data/cleaning/auditor_id_correction_audit.csv`：{len(correction)} 条标识清洗对照。
- `data/cleaning/firm_year_cleaned_auditor.csv`：{len(firm_year):,} 条清洗后的公司—年度。
- `data/pairs/company_pair_x_variables.parquet`：{len(pairs):,} 条公司对自变量。
- `descriptive/desc_1_overall_x_variables.csv` 至 `desc_4_by_accounting_firm.csv`：总体、分年度、分行业、分事务所描述性统计。
"""
    (DESCRIPTIVE / "X_variable_descriptive_summary.md").write_text(text, encoding="utf-8")


def main() -> None:
    for folder in (CLEANING, PAIRS, DESCRIPTIVE):
        folder.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    firm_year, correction, _ = clean_firm_year(panel)
    correction.to_csv(CLEANING / "auditor_id_correction_audit.csv", index=False, encoding="utf-8-sig")
    firm_year.drop(columns=[], errors="ignore").to_csv(
        CLEANING / "firm_year_cleaned_auditor.csv", index=False, encoding="utf-8-sig"
    )
    print(f"cleaned firm-years={len(firm_year)} corrections={len(correction)}")

    pairs = build_static_pairs(firm_year)
    pairs = add_dynamic_and_auditor_variables(pairs, firm_year)
    output_columns = [
        "时间", "ExpertiseIndustryCode", "证券代码_i", "证券代码_j", "CleanInstitutionID_i", "CleanInstitutionID_j",
        "SameAF", "ChgCo", "ChgCo_FromDiff", "BreakCo", "SameBig4", "SameBig10", "ChgBig4", "ChgBig10",
        "SharedCPACount", "AnySharedAud", "SameAud", "SimRt", "SimJaccard", "ChgAud", "ChgRt",
    ]
    pairs = pairs[output_columns]
    validate(firm_year, correction, pairs)
    pair_path = PAIRS / "company_pair_x_variables.parquet"
    pairs.to_parquet(pair_path, index=False)
    print(f"pairs={len(pairs)} same={int(pairs['SameAF'].sum())} -> {pair_path}")

    tables = build_descriptives(firm_year, pairs)
    tables["overall"].to_csv(DESCRIPTIVE / "desc_1_overall_x_variables.csv", index=False, encoding="utf-8-sig")
    tables["by_year"].to_csv(DESCRIPTIVE / "desc_2_by_year.csv", index=False, encoding="utf-8-sig")
    tables["by_industry"].to_csv(DESCRIPTIVE / "desc_3_by_industry.csv", index=False, encoding="utf-8-sig")
    tables["by_firm"].to_csv(DESCRIPTIVE / "desc_4_by_accounting_firm.csv", index=False, encoding="utf-8-sig")
    if int(tables["by_year"]["总配对数"].sum()) != len(pairs):
        raise AssertionError("分年度配对数之和与总对数不一致")
    if int(tables["by_industry"]["累计配对数"].sum()) != len(pairs):
        raise AssertionError("分行业配对数之和与总对数不一致")
    if int(tables["by_firm"]["同行业同年度客户对数量"].sum()) != int(pairs["SameAF"].sum()):
        raise AssertionError("分事务所同所对数之和与 SameAF 不一致")
    write_summary(firm_year, correction, pairs, tables)
    print("descriptives written")


if __name__ == "__main__":
    main()
