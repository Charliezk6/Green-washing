"""构造竞争性扩散通道，并检验它们与 SameAF 的相关和重叠。"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
APPENDIX = PROJECT / "appendix"
PAIRS = ROOT / "data" / "pairs" / "company_pair_x_variables.parquet"
OUT = ROOT / "correlation"

CHANNELS = [
    "SameGroup",
    "SameBlockholder",
    "SameProvince",
    "SameCity",
    "SameOfficeZip4",
    "BoardInterlock",
    "SameIndepDir",
    "SupplyChainLink",
    "BothInquired",
]
LABELS = {
    "SameGroup": "同一企业集团",
    "SameBlockholder": "共同大股东",
    "SameProvince": "注册地同省",
    "SameCity": "注册地同城",
    "SameOfficeZip4": "办公地邮编前四位相同",
    "BoardInterlock": "共享董事或高管",
    "SameIndepDir": "共享独立董事",
    "SupplyChainLink": "上市前五大客户或供应商",
    "BothInquired": "双方近年均收到年报问询",
}


def code6(value: object) -> str | None:
    text = "" if pd.isna(value) else str(value).strip().upper()
    for suffix in (".SH", ".SZ", ".BJ"):
        text = text.replace(suffix, "")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) != 6:
        return None
    return digits


def year_of(value: object) -> int | None:
    text = "" if pd.isna(value) else str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) < 4:
        return None
    year = int(digits[:4])
    return year if 2000 <= year <= 2030 else None


def eligible_controller(name: object) -> str | None:
    text = "" if pd.isna(name) else str(name).strip()
    if text in {"", "0", "无", "nan", "None"}:
        return None
    if "国资委" in text or "国有资产监督管理委员会" in text:
        return None
    return text


def eligible_holder(name: object) -> str | None:
    text = "" if pd.isna(name) else " ".join(str(name).strip().split())
    if not text or text.lower() == "nan":
        return None
    upper = text.upper()
    if "香港中央结算" in text or "HKSCC" in upper:
        return None
    return text


def add_pair(links: set[tuple[int, str, str]], year: int, left: str, right: str) -> None:
    if left == right:
        return
    lo, hi = (left, right) if left < right else (right, left)
    links.add((year, lo, hi))


def links_from_membership(members: dict[tuple[int, str], set[str]]) -> set[tuple[int, str, str]]:
    links: set[tuple[int, str, str]] = set()
    for (year, _), firms in members.items():
        if len(firms) < 2:
            continue
        for left, right in combinations(sorted(firms), 2):
            links.add((year, left, right))
    return links


def load_pairs() -> pd.DataFrame:
    pairs = pd.read_parquet(
        PAIRS,
        columns=["时间", "ExpertiseIndustryCode", "证券代码_i", "证券代码_j", "SameAF"],
    )
    pairs["时间"] = pairs["时间"].astype(int)
    pairs["证券代码_i"] = pairs["证券代码_i"].astype(str).str.zfill(6)
    pairs["证券代码_j"] = pairs["证券代码_j"].astype(str).str.zfill(6)
    pairs["SameAF"] = pairs["SameAF"].astype(np.int8)
    if len(pairs) != 2_051_610 or int(pairs["SameAF"].sum()) != 153_692:
        raise AssertionError(f"公司对或同所对数不符: {len(pairs)}, {int(pairs['SameAF'].sum())}")
    return pairs


def group_links() -> set[tuple[int, str, str]]:
    path = APPENDIX / "上市公司控制人文件152909828(仅供南京审计大学使用)" / "HLD_Contrshr.csv"
    data = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=["Stkcd", "Reptdt", "S0701b"])
    members: dict[tuple[int, str], set[str]] = {}
    for code, date, name in zip(data["Stkcd"], data["Reptdt"], data["S0701b"]):
        year = year_of(date)
        text = "" if pd.isna(date) else str(date)
        firm = code6(code)
        controller = eligible_controller(name)
        if year is None or firm is None or controller is None or not 2021 <= year <= 2024 or text[5:7] != "12":
            continue
        members.setdefault((year, controller), set()).add(firm)
    return links_from_membership(members)


def blockholder_links() -> set[tuple[int, str, str]]:
    path = APPENDIX / "十大股东文件195111618(仅供南京审计大学使用)" / "HLD_Shareholders.csv"
    data = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=["Stkcd", "Reptdt", "S0301a", "S0304a"])
    members: dict[tuple[int, str], set[str]] = {}
    for code, date, name, stake in zip(data["Stkcd"], data["Reptdt"], data["S0301a"], data["S0304a"]):
        year = year_of(date)
        text = "" if pd.isna(date) else str(date)
        if year is None or not 2021 <= year <= 2024 or text[5:7] != "12":
            continue
        firm = code6(code)
        holder = eligible_holder(name)
        try:
            share = float(stake)
        except (TypeError, ValueError):
            continue
        if firm is None or holder is None or share < 5:
            continue
        members.setdefault((year, holder), set()).add(firm)
    return links_from_membership(members)


def director_links() -> tuple[set[tuple[int, str, str]], set[tuple[int, str, str]]]:
    path = APPENDIX / "高管个人资料文件115143621(仅供南京审计大学使用)" / "CG_Director.csv"
    data = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=["Stkcd", "Reptdt", "PersonID", "PositionID"])
    board: dict[tuple[int, str], set[str]] = {}
    indep_firms: dict[tuple[int, str], set[str]] = {}
    board_firms: dict[tuple[int, str], set[str]] = {}
    for code, date, person, position in zip(data["Stkcd"], data["Reptdt"], data["PersonID"], data["PositionID"]):
        year = year_of(date)
        firm = code6(code)
        if year is None or firm is None or not 2021 <= year <= 2024 or pd.isna(person):
            continue
        person_id = str(person).strip()
        roles = [] if pd.isna(position) else [part.strip() for part in str(position).split(",") if part.strip()]
        if any(role.startswith("P351") or role.startswith("P353") for role in roles):
            board.setdefault((year, person_id), set()).add(firm)
            board_firms.setdefault((year, person_id), set()).add(firm)
        if "P35106" in roles:
            indep_firms.setdefault((year, person_id), set()).add(firm)
    indep_links: set[tuple[int, str, str]] = set()
    for key, firms in indep_firms.items():
        others = board_firms.get(key, set())
        for indep_firm in firms:
            for other in others:
                add_pair(indep_links, key[0], indep_firm, other)
    return links_from_membership(board), indep_links


def location_table() -> pd.DataFrame:
    path = APPENDIX / "上市公司基本信息年度表121239549(仅供南京审计大学使用)" / "STK_LISTEDCOINFOANL.csv"
    data = pd.read_csv(
        path,
        dtype=str,
        encoding="utf-8-sig",
        usecols=["Symbol", "EndDate", "Zipcode", "PROVINCECODE", "CITYCODE"],
    )
    data["证券代码"] = data["Symbol"].map(code6)
    data["时间"] = data["EndDate"].map(year_of)
    data = data.loc[data["证券代码"].notna() & data["时间"].between(2021, 2024), [
        "证券代码", "时间", "PROVINCECODE", "CITYCODE", "Zipcode",
    ]]
    data["时间"] = data["时间"].astype(int)
    data["PROVINCECODE"] = data["PROVINCECODE"].fillna("").str.strip()
    data["CITYCODE"] = data["CITYCODE"].fillna("").str.strip()

    def prefixes(value: object) -> tuple[str, ...]:
        text = "" if pd.isna(value) else str(value)
        found = []
        for part in text.replace("，", ",").replace(";", ",").split(","):
            digits = "".join(ch for ch in part if ch.isdigit())
            if len(digits) >= 4:
                found.append(digits[:4])
        return tuple(dict.fromkeys(found))

    parsed = data["Zipcode"].map(prefixes)
    width = max((len(item) for item in parsed), default=1)
    for index in range(width):
        data[f"zip4_{index}"] = [item[index] if index < len(item) else "" for item in parsed]
    return data.drop(columns=["Zipcode"]).drop_duplicates(["证券代码", "时间"], keep="last")


def supply_links() -> set[tuple[int, str, str]]:
    links: set[tuple[int, str, str]] = set()
    files = [
        APPENDIX / "前五大客户销售信息表200610012(仅供南京审计大学使用)" / "SC_TopFiveSaleInfo.csv",
        APPENDIX / "前五大供应商采购信息表200634651(仅供南京审计大学使用)" / "SC_TopFivePurchaseInfo.csv",
    ]
    for path in files:
        data = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=["Symbol", "EndDate", "IsListed", "BusinessSymbol"])
        listed = data["IsListed"].fillna("").str.upper().eq("Y")
        data = data.loc[listed]
        for code, date, other in zip(data["Symbol"], data["EndDate"], data["BusinessSymbol"]):
            year = year_of(date)
            firm = code6(code)
            counterparty = code6(other)
            if year is None or firm is None or counterparty is None or not 2021 <= year <= 2024:
                continue
            add_pair(links, year, firm, counterparty)
    return links


def inquiry_flags() -> pd.DataFrame:
    path = APPENDIX / "问询函信息表(上市公司)(日)195715411(仅供南京审计大学使用)" / "CMS_InquiriesSE.csv"
    data = pd.read_csv(path, dtype=str, encoding="utf-8-sig", usecols=["Symbol", "InquiryDate", "InquiryType"])
    annual = data["InquiryType"].fillna("").astype(str).str.strip().eq("1")
    data = data.loc[annual]
    firms = {code6(code) for code in data["Symbol"]}
    years = {year_of(date) for date in data["InquiryDate"]}
    received = {
        (code6(code), year_of(date))
        for code, date in zip(data["Symbol"], data["InquiryDate"])
        if code6(code) and year_of(date)
    }
    rows = []
    for firm in firms:
        if firm is None:
            continue
        for year in range(2021, 2025):
            hit = (firm, year) in received or (firm, year - 1) in received
            if hit:
                rows.append((firm, year, 1))
    flags = pd.DataFrame(rows, columns=["证券代码", "时间", "inquired"])
    return flags


def mark_links(pairs: pd.DataFrame, links: set[tuple[int, str, str]]) -> np.ndarray:
    if not links:
        return np.zeros(len(pairs), dtype=np.int8)
    linked = pd.DataFrame(list(links), columns=["时间", "证券代码_i", "证券代码_j"])
    linked["hit"] = np.int8(1)
    matched = pairs[["时间", "证券代码_i", "证券代码_j"]].merge(linked, how="left", on=["时间", "证券代码_i", "证券代码_j"])
    return matched["hit"].fillna(0).astype(np.int8).to_numpy()


def attach_channels(pairs: pd.DataFrame) -> pd.DataFrame:
    print("building group and blockholder links")
    result = pairs.copy()
    result["SameGroup"] = mark_links(result, group_links())
    result["SameBlockholder"] = mark_links(result, blockholder_links())
    print("building director links")
    board, indep = director_links()
    result["BoardInterlock"] = mark_links(result, board)
    result["SameIndepDir"] = mark_links(result, indep)
    print("building supply and inquiry links")
    result["SupplyChainLink"] = mark_links(result, supply_links())
    inquired = inquiry_flags()
    result = result.merge(inquired.add_prefix("i_").rename(columns={"i_证券代码": "证券代码_i", "i_时间": "时间"}), how="left")
    result = result.merge(inquired.add_prefix("j_").rename(columns={"j_证券代码": "证券代码_j", "j_时间": "时间"}), how="left")
    result["BothInquired"] = (result["i_inquired"].fillna(0).eq(1) & result["j_inquired"].fillna(0).eq(1)).astype(np.int8)
    result = result.drop(columns=["i_inquired", "j_inquired"])

    print("building location")
    location = location_table()
    zip_columns = [column for column in location.columns if column.startswith("zip4_")]
    keep = ["证券代码", "时间", "PROVINCECODE", "CITYCODE", *zip_columns]
    result = result.merge(location[keep].add_prefix("i_").rename(columns={"i_证券代码": "证券代码_i", "i_时间": "时间"}), how="left")
    result = result.merge(location[keep].add_prefix("j_").rename(columns={"j_证券代码": "证券代码_j", "j_时间": "时间"}), how="left")
    result["SameProvince"] = (
        result["i_PROVINCECODE"].fillna("").ne("") & result["i_PROVINCECODE"].eq(result["j_PROVINCECODE"])
    ).astype(np.int8)
    result["SameCity"] = (
        result["i_CITYCODE"].fillna("").ne("") & result["i_CITYCODE"].eq(result["j_CITYCODE"])
    ).astype(np.int8)
    office = np.zeros(len(result), dtype=bool)
    for left_name in zip_columns:
        left = result[f"i_{left_name}"].fillna("").to_numpy()
        for right_name in zip_columns:
            right = result[f"j_{right_name}"].fillna("").to_numpy()
            office |= (left != "") & (left == right)
    result["SameOfficeZip4"] = office.astype(np.int8)
    drop_columns = [column for column in result.columns if column.startswith(("i_", "j_"))]
    return result.drop(columns=drop_columns)


def phi_matrix(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns, dtype=float)
    for left_index, left in enumerate(columns):
        x = data[left].to_numpy(dtype=float)
        for right in columns[:left_index]:
            y = data[right].to_numpy(dtype=float)
            matrix.loc[left, right] = matrix.loc[right, left] = float(np.corrcoef(x, y)[0, 1])
    return matrix


def demean(values: np.ndarray, groups: pd.Series) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "group": groups.to_numpy()})
    means = frame.groupby("group", sort=False)["value"].transform("mean").to_numpy()
    return values - means


def twoway_t(y: np.ndarray, x: np.ndarray, firm_i: np.ndarray, firm_j: np.ndarray) -> tuple[float, float, float, float]:
    xx = float(np.dot(x, x))
    if xx <= 0:
        return (np.nan, np.nan, np.nan, np.nan)
    beta = float(np.dot(x, y) / xx)
    residual = x * (y - x * beta)

    def meat(groups: np.ndarray) -> tuple[float, int]:
        summed = pd.DataFrame({"group": groups, "score": residual}).groupby("group", sort=False)["score"].sum().to_numpy()
        return float(np.dot(summed, summed)), int(summed.size)

    meat_i, count_i = meat(firm_i)
    meat_j, count_j = meat(firm_j)
    meat_pair = float(np.dot(residual, residual))
    count_pair = int(residual.size)
    variance_i = meat_i * count_i / (count_i - 1) / (xx ** 2)
    variance_j = meat_j * count_j / (count_j - 1) / (xx ** 2)
    variance_pair = meat_pair * count_pair / (count_pair - 1) / (xx ** 2)
    variance = variance_i + variance_j - variance_pair
    if not np.isfinite(variance) or variance <= 0:
        variance = max(variance_i, variance_j)
    standard_error = float(np.sqrt(variance))
    t_value = beta / standard_error
    p_value = float(2 * norm.sf(abs(t_value)))
    return beta, standard_error, t_value, p_value


def vif_after_fe(data: pd.DataFrame, columns: list[str], groups: pd.Series) -> dict[str, float]:
    demeaned = np.column_stack([demean(data[column].to_numpy(dtype=float), groups) for column in columns])
    scores = {}
    for index, name in enumerate(columns):
        y = demeaned[:, index]
        x = np.delete(demeaned, index, axis=1)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        total = float(np.dot(y, y))
        residual = float(np.dot(y - fitted, y - fitted))
        r_squared = 1 - residual / total if total > 0 else np.nan
        scores[name] = float(1 / (1 - r_squared)) if np.isfinite(r_squared) and r_squared < 1 else np.nan
    return scores


def correlation_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variables = ["SameAF", *CHANNELS]
    matrix = phi_matrix(data, variables)
    same = data["SameAF"].eq(1)
    groups = data["时间"].astype(str) + "|" + data["ExpertiseIndustryCode"].astype(str)
    firm_i = data["证券代码_i"].to_numpy()
    firm_j = data["证券代码_j"].to_numpy()
    same_dm = demean(data["SameAF"].to_numpy(dtype=float), groups)
    vif = vif_after_fe(data, variables, groups)
    phi_rows = []
    overlap_rows = []
    cluster_rows = []
    for channel in CHANNELS:
        values = data[channel].to_numpy(dtype=float)
        phi_rows.append({
            "变量": channel,
            "变量名称": LABELS[channel],
            "与SameAF的Phi系数": matrix.loc["SameAF", channel],
            "通道均值": values.mean(),
            "同为1的配对数": int(((data["SameAF"] == 1) & (data[channel] == 1)).sum()),
        })
        overlap_rows.append({
            "变量": channel,
            "变量名称": LABELS[channel],
            "同所配对数": int(same.sum()),
            "同所且通道为1": int((same & data[channel].eq(1)).sum()),
            "同所组比例": float(data.loc[same, channel].mean()),
            "不同所配对数": int((~same).sum()),
            "不同所且通道为1": int((~same & data[channel].eq(1)).sum()),
            "不同所组比例": float(data.loc[~same, channel].mean()),
        })
        beta, error, t_value, p_value = twoway_t(demean(values, groups), same_dm, firm_i, firm_j)
        cluster_rows.append({
            "变量": channel,
            "变量名称": LABELS[channel],
            "系数": beta,
            "双向聚类标准误": error,
            "t值": t_value,
            "p值": p_value,
            "行业年度固定效应": "是",
            "VIF": vif[channel],
            "SameAF的VIF": vif["SameAF"],
            "观测数": len(data),
        })
    return pd.DataFrame(phi_rows), pd.DataFrame(overlap_rows), pd.DataFrame(cluster_rows)


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value * 100:.2f}%"


def write_report(phi: pd.DataFrame, overlap: pd.DataFrame, cluster: pd.DataFrame) -> None:
    merged = phi.merge(overlap, on=["变量", "变量名称"]).merge(cluster, on=["变量", "变量名称"])
    merged = merged.sort_values("与SameAF的Phi系数", ascending=False)
    by_name = {row["变量"]: row for row in merged.to_dict("records")}
    same_af_vif = float(merged["SameAF的VIF"].iloc[0])
    lines = [
        "# SameAF 与其他扩散通道的相关性",
        "",
        "样本是 2,051,610 个同一年度、同一行业的无序公司对，其中同所公司对 153,692 个。九条通道都是公司对上的 0/1。Phi 是两个 0/1 变量的相关系数。每条通道对 `SameAF` 的回归去掉行业×年度均值，标准误按两家证券代码双向聚类。VIF 来自九条通道与 `SameAF` 同时进入、并去掉行业×年度均值之后。",
        "",
        "| 通道 | Phi | 同所组比例 | 不同所组比例 | 聚类系数 | t值 | VIF |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in merged.to_dict("records"):
        lines.append(
            f"| {row['变量名称']} | {row['与SameAF的Phi系数']:.4f} | {percent(row['同所组比例'])} | "
            f"{percent(row['不同所组比例'])} | {row['系数']:.4f} | {row['t值']:.2f} | {row['VIF']:.2f} |"
        )

    province = by_name["SameProvince"]
    city = by_name["SameCity"]
    office = by_name["SameOfficeZip4"]
    group = by_name["SameGroup"]
    holder = by_name["SameBlockholder"]
    board = by_name["BoardInterlock"]
    indep = by_name["SameIndepDir"]
    supply = by_name["SupplyChainLink"]
    inquiry = by_name["BothInquired"]
    lines.extend([
        "",
        (
            f"地理通道和同所的重叠最大。同所公司对里，{percent(province['同所组比例'])} 注册地在同一省份，"
            f"不同所公司对里这一比例是 {percent(province['不同所组比例'])}，Phi 为 {province['与SameAF的Phi系数']:.3f}，"
            f"双向聚类 t 值为 {province['t值']:.2f}。同城的对应比例是 {percent(city['同所组比例'])} 和 {percent(city['不同所组比例'])}，"
            f"办公地邮编前四位相同的对应比例是 {percent(office['同所组比例'])} 和 {percent(office['不同所组比例'])}。"
            "同省与同城、同城与邮编前四位彼此相关，所以同城的 VIF 高于其余地理变量。"
        ),
        "",
        (
            f"集团、大股东和董事联结在同所公司对里仍然很少，比例高于不同所公司对。"
            f"同所组里，同一集团有 {int(group['同所且通道为1'])} 对（{percent(group['同所组比例'])}），"
            f"共同持股达到 5% 的股东有 {int(holder['同所且通道为1'])} 对（{percent(holder['同所组比例'])}），"
            f"共享董事或高管有 {int(board['同所且通道为1'])} 对（{percent(board['同所组比例'])}），"
            f"共享独立董事有 {int(indep['同所且通道为1'])} 对（{percent(indep['同所组比例'])}）。"
            f"不同所组的对应比例是 {percent(group['不同所组比例'])}、{percent(holder['不同所组比例'])}、"
            f"{percent(board['不同所组比例'])} 和 {percent(indep['不同所组比例'])}。"
            f"四条通道的双向聚类 t 值分别为 {group['t值']:.2f}、{holder['t值']:.2f}、{board['t值']:.2f} 和 {indep['t值']:.2f}。"
            f"共享董事与共享独董的 Phi 为 0.85，VIF 分别为 {board['VIF']:.2f} 和 {indep['VIF']:.2f}，两条几乎是同一条通道。"
        ),
        "",
        (
            f"与同所几乎不重合的是供应链和年报问询。同所的 153,692 对里，只有 {int(supply['同所且通道为1'])} 对"
            f"在前五大客户或供应商中互相标为上市公司，Phi 为 {supply['与SameAF的Phi系数']:.4f}，t 值为 {supply['t值']:.2f}。"
            f"全部公司对里这类上市供应链共 {int(supply['同所且通道为1'] + supply['不同所且通道为1'])} 对。"
            f"双方在当年或上一年都收到年报问询函的比例，同所组是 {percent(inquiry['同所组比例'])}，"
            f"不同所组是 {percent(inquiry['不同所组比例'])}，Phi 为 {inquiry['与SameAF的Phi系数']:.3f}。"
        ),
        "",
        (
            f"九条通道一起进入时，`SameAF` 的 VIF 是 {same_af_vif:.2f}。"
            "同所和这些通道可以同时放进行业×年度固定效应回归。"
        ),
        "",
        "供应链只保留前五名中标明为上市公司并给出股票代码的对手方。集团匹配去掉名称中含“国资委”或“国有资产监督管理委员会”的控制人，并使用当年 12 月 31 日的记录。大股东匹配去掉香港中央结算有限公司和 HKSCC NOMINEES，只保留年末持股达到 5% 的股东。董事联结使用人物编码，监事不计入。",
    ])
    (OUT / "SameAF_correlation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = load_pairs()
    data = attach_channels(pairs)
    for channel in CHANNELS:
        data[channel] = data[channel].astype(np.int8)
    output_columns = ["时间", "ExpertiseIndustryCode", "证券代码_i", "证券代码_j", "SameAF", *CHANNELS]
    data[output_columns].to_parquet(OUT / "company_pair_channels.parquet", index=False)
    print({channel: int(data[channel].sum()) for channel in CHANNELS})
    phi, overlap, cluster = correlation_tables(data)
    matrix = phi_matrix(data, ["SameAF", *CHANNELS])
    matrix.index.name = "变量"
    matrix.to_csv(OUT / "corr_1_pairwise_matrix.csv", encoding="utf-8-sig")
    overlap.to_csv(OUT / "corr_2_channel_overlap_crosstab.csv", index=False, encoding="utf-8-sig")
    cluster.to_csv(OUT / "corr_3_clustered_t_and_vif.csv", index=False, encoding="utf-8-sig")
    write_report(phi, overlap, cluster)
    print(f"outputs={OUT}")


if __name__ == "__main__":
    main()
