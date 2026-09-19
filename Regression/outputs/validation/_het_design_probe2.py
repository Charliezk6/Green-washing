# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(r"d:\Charlie 应用\Documents\zkk\Research Project\greenwashing")
OUT = ROOT / "Regression" / "outputs" / "validation" / "_het_design_probe2.txt"
JSONL = ROOT / "Y_variable_reproduction" / "data" / "input" / "expansion" / "layer_b_results.jsonl"
COMMON = ROOT / "Regression" / "outputs" / "paper_tables" / "common_sample_15381.csv"
APP = ROOT / "appendix"

lines = []
def log(m=""):
    lines.append(str(m))

# ---- coding rules ----
EXCLUDE = [
    "是否属于环境保护部门公布的重点排污单位",
    "报告期内因环境问题受到行政处罚的情况",
    "对上市公司生产经营的影响公司的整改措施其他应当公开的环境信息",
    "公司或子公司名称处罚原因违规情形处罚结果",
]
# 在建 as false positive in 在建设
CIP_POS = re.compile(r"在建工程|技改在建|工程在建|尚在建|正在建设|在建项目")
CIP_NEG = re.compile(r"在建设")  # often 在建设项目环境影响评价 without 在建工程

AUDIT_PATTERNS = {
    "capex": re.compile(r"环保投资|环境投资|环保投入|环境投入|环保支出|污染治理投资|脱硫脱硝投资|环保技改.{0,8}(投入|投资|支出)|资本性支出"),
    "equipment": re.compile(r"环保设施|环保设备|污染治理设施|污水处理设施|废气处理设施|除尘(?:设备|设施)|脱硫(?:装置|设施|设备)|脱硝(?:装置|设施)|在线监测(?:设备|设施)|废水处理设施"),
    "penalty": re.compile(r"(?:环保|环境|生态).{0,12}(?:处罚|罚款)|行政处罚.{0,20}(?:万|元)|罚款.{0,12}(?:万|元)"),
    "rectify": re.compile(r"(?:完成|实施|落实|开展).{0,8}整改|整改(?:措施|完成|完毕|到位)|限期治理"),
    "remediate": re.compile(r"生态修复|土壤修复|环境修复|修复治理|土地修复|矿山修复"),
    "cip": re.compile(r"在建工程|环保工程.{0,8}在建|技改在建"),
    "liab": re.compile(r"预计负债|或有负债|弃置费用|环境负债|环境治理义务"),
}

PLACEBO_PAT = re.compile(r"绿水青山|绿色发展理念|可持续发展理念|高度重视环境保护|积极履行(?:环保|环境|社会责任)|倡导绿色|碳达峰碳中和目标|美丽中国|生态文明理念")

def is_template(sent: str) -> bool:
    return any(x in sent for x in EXCLUDE)

def classify_auditable(sent: str) -> list[str]:
    if is_template(sent):
        return []
    hits = []
    for name, pat in AUDIT_PATTERNS.items():
        if pat.search(sent):
            hits.append(name)
    # 设备/投资 leftover noise filters
    if "电子设备制造" in sent or "资产负债管理" in sent:
        hits = [h for h in hits if h not in ("equipment", "liab")]
    return hits

# unique sentences
unique = {}  # sent -> rec meta
n_pairs = 0
n_unique = 0
cat_unique = Counter()
cat_pairs = Counter()
placebo_unique = 0
placebo_pairs = 0
fy_total = Counter()
fy_audit = Counter()
fy_placebo = Counter()
fy_audit_cats = defaultdict(Counter)
examples = defaultdict(list)
noise_examples = defaultdict(list)
raw_kw_unique = {k: set() for k in ["设备", "投资", "处罚", "整改", "负债", "修复", "在建", "环保设施", "资本开支"]}

with JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        n_pairs += 1
        sent = rec.get("sentence") or ""
        code = str(rec.get("证券代码"))
        year = rec.get("时间")
        judg = rec.get("judgment")
        fy = (code.zfill(6), int(year) if year is not None else None)
        fy_total[fy] += 1
        for k in raw_kw_unique:
            if k in sent:
                raw_kw_unique[k].add(sent)
        hits = classify_auditable(sent)
        if hits:
            fy_audit[fy] += 1
            for h in hits:
                cat_pairs[h] += 1
                fy_audit_cats[h][fy] += 1
        if PLACEBO_PAT.search(sent) and not hits:
            fy_placebo[fy] += 1
            placebo_pairs += 1
        if sent not in unique:
            unique[sent] = {"code": code, "year": year, "word": rec.get("word"), "judgment": judg}
            n_unique += 1
            if hits:
                for h in hits:
                    cat_unique[h] += 1
                    if len(examples[h]) < 6:
                        examples[h].append((judg, rec.get("word"), f"{code}-{year}", sent[:320]))
            else:
                # collect noise from raw keywords
                for k in ["设备", "投资", "处罚", "整改", "负债", "修复", "在建"]:
                    if k in sent and len(noise_examples[k]) < 5 and not is_template(sent):
                        noise_examples[k].append((judg, rec.get("word"), f"{code}-{year}", sent[:280]))
            if PLACEBO_PAT.search(sent) and not hits:
                placebo_unique += 1
                if len(examples["placebo"]) < 6:
                    examples["placebo"].append((judg, rec.get("word"), f"{code}-{year}", sent[:280]))

log(f"pairs={n_pairs} unique_sentences={n_unique}")
log(f"raw unique sentences containing keyword:")
for k, s in raw_kw_unique.items():
    log(f"  {k}: unique_sents={len(s)}")
log(f"\nAUDitable unique sentences by rule (after noise filter): {dict(cat_unique)}")
log(f"AUDitable pairs by rule: {dict(cat_pairs)}")
log(f"placebo unique={placebo_unique} pairs={placebo_pairs}")
log(f"firm-years with any Layer B pairs={len(fy_total)}")
log(f"firm-years with >=1 auditable pair={sum(1 for v in fy_audit.values() if v>0)}")
log(f"firm-years with >=1 placebo pair={sum(1 for v in fy_placebo.values() if v>0)}")

# shares
shares = []
for fy, tot in fy_total.items():
    a = fy_audit.get(fy, 0)
    p = fy_placebo.get(fy, 0)
    shares.append((a / tot, p / tot, tot, a, p, fy))
s = pd.Series([x[0] for x in shares])
log(f"AudShare describe:\n{s.describe(percentiles=[0.1,0.25,0.5,0.75,0.9])}")
log(f"AudShare>0 share of FY={ (s>0).mean():.3f} median among >0={s[s>0].median() if (s>0).any() else None}")
ps = pd.Series([x[1] for x in shares])
log(f"PlaceboShare describe:\n{ps.describe(percentiles=[0.5,0.75,0.9])}")

log("\n=== AUDITABLE EXAMPLES ===")
for k, exs in examples.items():
    log(f"\n-- {k} --")
    for judg, word, fy, sent in exs:
        log(f"  j={judg} word={word} {fy}: {sent}")

OUT.write_text("\n".join(lines), encoding="utf-8")

log("\n=== RAW-KEYWORD NOISE (not captured by tight rules, non-template) ===")
for k, exs in noise_examples.items():
    log(f"\n-- {k} --")
    for judg, word, fy, sent in exs:
        log(f"  j={judg} word={word} {fy}: {sent}")

# merge to common sample
common = pd.read_csv(COMMON, dtype={"证券代码": str}, encoding="utf-8-sig")
common["证券代码"] = common["证券代码"].str.zfill(6)
rows = []
for fy, tot in fy_total.items():
    rows.append({
        "证券代码": fy[0], "时间": fy[1],
        "n_pairs": tot,
        "n_audit": fy_audit.get(fy, 0),
        "n_placebo": fy_placebo.get(fy, 0),
        "AudShare": fy_audit.get(fy, 0) / tot if tot else 0,
        "PlaceboShare": fy_placebo.get(fy, 0) / tot if tot else 0,
    })
text = pd.DataFrame(rows)
m = common.merge(text, on=["证券代码", "时间"], how="left")
log(f"\ncommon matched LayerB text n={m.n_pairs.notna().sum()} missing={m.n_pairs.isna().sum()}")
log(f"  AudShare>0={(m.AudShare.fillna(0)>0).mean():.3f} median={m.AudShare.median()} mean={m.AudShare.mean()}")
log(f"  HighAudit median split: {(m.AudShare >= m.AudShare.median()).mean()}")
log(f"  PlaceboShare>0={(m.PlaceboShare.fillna(0)>0).mean():.3f}")

# penalty coverage
pen_path = APP / "上市公司环保处罚总表.xlsx"
pen = pd.read_excel(pen_path, dtype={"证券代码": str})
pen["证券代码"] = pen["证券代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
# year from 处罚日期 else 公告日期
def to_year(x):
    if pd.isna(x):
        return None
    s = str(int(x)) if isinstance(x, float) and x == int(x) else str(x)
    s = re.sub(r"\D", "", s)
    if len(s) >= 4:
        y = int(s[:4])
        if 1990 <= y <= 2026:
            return y
    return None
pen["year_penalty"] = pen["处罚日期"].map(to_year)
pen["year_ann"] = pen["公告日期"].map(to_year)
pen["时间"] = pen["year_penalty"].fillna(pen["year_ann"])
log(f"\nPENALTY n={len(pen)} firms={pen['证券代码'].nunique()}")
yd = pen["时间"].value_counts().sort_index()
log("  year dist:\n" + yd.to_string())
sub = pen[pen["时间"].between(2021, 2024)]
log(f"  2021-2024 events={len(sub)} firm-years={sub.groupby(['证券代码','时间']).ngroups} firms={sub['证券代码'].nunique()}")
log(f"  处罚金额 describe:\n{pd.to_numeric(sub['处罚金额（万元）'], errors='coerce').describe()}")
log(f"  与上市公司关系:\n{sub['与上市公司关系'].value_counts().head(10).to_string()}")
pen_fy = sub.groupby(["证券代码", "时间"]).agg(
    n_penalty=("证券代码", "size"),
    penalty_amt=("处罚金额（万元）", lambda s: pd.to_numeric(s, errors="coerce").sum()),
).reset_index()
m2 = m.merge(pen_fy, on=["证券代码", "时间"], how="left")
log(f"  common HasPenalty={(m2.n_penalty.fillna(0)>0).sum()} ({(m2.n_penalty.fillna(0)>0).mean():.3f})")

# investment xlsx vs ENV
invx = pd.read_excel(APP / "上市公司环保投资额（1999-2024年）.xlsx", dtype={"股票代码": str})
invx["证券代码"] = invx["股票代码"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
invx["时间"] = pd.to_numeric(invx["年份"], errors="coerce")
invx = invx[invx["时间"].between(2021, 2024)]
log(f"\nINV XLSX 2021-2024 n={len(invx)} firms={invx['证券代码'].nunique()} amt>0={(pd.to_numeric(invx['环保投资金额_元'], errors='coerce')>0).sum()}")
m3 = m.merge(invx[["证券代码", "时间", "环保投资金额_元"]], on=["证券代码", "时间"], how="left")
log(f"  common matched inv xlsx non-null={m3['环保投资金额_元'].notna().sum()} ({m3['环保投资金额_元'].notna().mean():.3f})")
log(f"  amt>0={(pd.to_numeric(m3['环保投资金额_元'], errors='coerce')>0).sum()}")

env_inv = pd.read_csv(APP / "上市公司环境投资统计表155903146(仅供南京审计大学使用)" / "ENV_InvestStats.csv", dtype={"Symbol": str})
env_inv["证券代码"] = env_inv.Symbol.str.zfill(6)
env_inv["时间"] = env_inv.EndDate.astype(str).str[:4].astype(int)
cmp = m3.merge(env_inv[env_inv["时间"].between(2021,2024)][["证券代码","时间","EPInvest"]], on=["证券代码","时间"], how="left")
both = cmp["环保投资金额_元"].notna() & cmp["EPInvest"].notna()
only_x = cmp["环保投资金额_元"].notna() & cmp["EPInvest"].isna()
only_e = cmp["环保投资金额_元"].isna() & cmp["EPInvest"].notna()
log(f"  overlap both={both.sum()} only_xlsx={only_x.sum()} only_ENV={only_e.sum()}")

# KeyPoll already known
sup = pd.read_csv(APP / "上市公司环境监管与认证披露情况表155713542(仅供南京审计大学使用)" / "ENV_SupCertDisclose.csv", dtype={"Symbol": str})
sup["证券代码"] = sup.Symbol.str.zfill(6)
sup["时间"] = sup.EndDate.astype(str).str[:4].astype(int)
m4 = m.merge(sup[sup["时间"].between(2021,2024)][["证券代码","时间","KeyPollMonUnit","IsPassISO14001"]], on=["证券代码","时间"], how="left")
log(f"\nKeyPoll in common=1: {(pd.to_numeric(m4.KeyPollMonUnit, errors='coerce')==1).sum()}")
log(f"ISO14001=1: {(pd.to_numeric(m4.IsPassISO14001, errors='coerce')==1).sum()}")

# carbon
carb = pd.read_csv(APP / "CSMAR碳排放量（2021-2024）.csv", dtype={"证券代码": str})
carb["证券代码"] = carb["证券代码"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
carb["时间"] = pd.to_datetime(carb["统计截止日期"], errors="coerce").dt.year
log(f"carbon n={len(carb)} unique fy={carb.groupby(['证券代码','时间']).ngroups}")
m5 = common.merge(carb, on=["证券代码","时间"], how="left")
log(f"  common matched any carbon field={m5['温室气体排放量'].notna().sum()} + {m5['碳排放量'].notna().sum()} + {m5['二氧化碳排放量'].notna().sum()}")

# patents 2021-2024
pat = pd.read_excel(APP / "上市公司绿色专利数据（1999-2024年）.xlsx", dtype={"股票代码": str})
pat["证券代码"] = pat["股票代码"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(6)
pat["时间"] = pd.to_numeric(pat["年份"], errors="coerce")
pat = pat[pat["时间"].between(2021, 2024)]
m6 = common.merge(pat[["证券代码","时间","企业绿色专利申请总量"]], on=["证券代码","时间"], how="left")
log(f"green patents matched={m6['企业绿色专利申请总量'].notna().sum()} >0={(pd.to_numeric(m6['企业绿色专利申请总量'], errors='coerce')>0).sum()}")

OUT.write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT, "size", OUT.stat().st_size)
