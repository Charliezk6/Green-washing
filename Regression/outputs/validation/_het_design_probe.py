# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(r"d:\Charlie 应用\Documents\zkk\Research Project\greenwashing")
OUT = ROOT / "Regression" / "outputs" / "validation" / "_het_design_probe.txt"
JSONL = ROOT / "Y_variable_reproduction" / "data" / "input" / "expansion" / "layer_b_results.jsonl"
PANEL = ROOT / "Regression" / "data" / "input" / "combined_greenwashing_auditor_expertise_panel.csv"
COMMON = ROOT / "Regression" / "outputs" / "paper_tables" / "common_sample_15381.csv"

KEYWORDS = ["设备", "投资", "处罚", "整改", "负债", "修复", "在建", "环保设施", "资本开支"]
COMPOUND = {
    "环保投资": ["环保投资", "环境投资", "环保投入", "环境投入"],
    "环保设施": ["环保设施", "环保设备", "污染治理设施", "脱硫", "脱硝", "除尘设备"],
    "处罚罚款": ["处罚", "罚款", "罚金"],
    "整改": ["整改", "限期治理"],
    "在建": ["在建", "在建工程"],
    "负债准备": ["负债", "预计负债", "或有负债", "环境负债"],
    "修复": ["修复", "修复治理", "生态修复", "土壤修复"],
    "资本开支": ["资本开支", "资本支出", "资本性支出"],
}

PLACEBO = ["倡导", "理念", "愿景", "承诺", "碳中和", "绿水青山", "可持续发展", "社会责任", "积极响应", "高度重视"]

rng = random.Random(42)
lines_out = []

def log(msg=""):
    lines_out.append(str(msg))


# ---------- appendix xlsx ----------
log("=== APPENDIX FILES ===")
app = ROOT / "appendix"
xlsx_files = []
for f in app.iterdir():
    if f.is_file() and f.suffix.lower() in {".xlsx", ".xls", ".csv"}:
        xlsx_files.append(f)
        log(f"FILE\t{f.name}\t{f.stat().st_size}")

for f in xlsx_files:
    if f.suffix.lower() != ".xlsx":
        continue
    try:
        xl = pd.ExcelFile(f)
        log(f"\nXLSX {f.name}")
        log(f"  sheets={xl.sheet_names[:10]}")
        for sheet in xl.sheet_names[:3]:
            df = pd.read_excel(f, sheet_name=sheet, nrows=5)
            log(f"  sheet={sheet} cols={list(df.columns)}")
            log(f"  head=\n{df.head(3).to_string()}")
            full = pd.read_excel(f, sheet_name=sheet)
            log(f"  nrows={len(full)}")
            # year-like cols
            for c in full.columns:
                s = str(c)
                if any(k in s.lower() for k in ["year", "date", "时间", "年份", "年度", "end"]):
                    log(f"  col {c} sample={full[c].dropna().astype(str).head(8).tolist()} nunique={full[c].nunique()}")
            for c in full.columns:
                s = str(c)
                if any(k in s for k in ["代码", "Symbol", "证券", "股票", "stk", "code", "Code"]):
                    vals = full[c].dropna().astype(str).head(8).tolist()
                    log(f"  code col {c} sample={vals} nunique={full[c].nunique()}")
    except Exception as e:
        log(f"XLSX FAIL {f.name}: {e}")


# ---------- ENV coverage ----------
log("\n=== ENV TABLES ===")
env_inv = pd.read_csv(app / "上市公司环境投资统计表155903146(仅供南京审计大学使用)" / "ENV_InvestStats.csv", dtype={"Symbol": str})
env_sup = pd.read_csv(app / "上市公司环境监管与认证披露情况表155713542(仅供南京审计大学使用)" / "ENV_SupCertDisclose.csv", dtype={"Symbol": str})
log(f"ENV_InvestStats n={len(env_inv)} firms={env_inv.Symbol.nunique()} years={sorted(env_inv.EndDate.astype(str).str[:4].unique())[:20]}")
log(f"  EPInvest describe=\n{pd.to_numeric(env_inv.EPInvest, errors='coerce').describe()}")
log(f"  2021-2024 n={(env_inv.EndDate.astype(str).str[:4].isin(['2021','2022','2023','2024'])).sum()}")
log(f"  Symbol sample={env_inv.Symbol.head(8).tolist()} lens={env_inv.Symbol.str.len().value_counts().to_dict()}")

log(f"ENV_SupCertDisclose n={len(env_sup)}")
sup_years = env_sup.EndDate.astype(str).str[:4]
log(f"  years 2021-2024 n={sup_years.isin(['2021','2022','2023','2024']).sum()}")
for col in ["KeyPollMonUnit", "PollEmissStandard", "IsPassISO14001", "IsPassISO9001"]:
    sub = env_sup[sup_years.isin(["2021", "2022", "2023", "2024"])]
    log(f"  {col} value_counts=\n{sub[col].value_counts(dropna=False).head(10)}")

env_perf = pd.read_csv(app / "上市公司环境绩效明细表155529021(仅供南京审计大学使用)" / "ENV_PerfDetail.csv", dtype={"Symbol": str}, nrows=0)
log(f"ENV_PerfDetail cols={list(env_perf.columns)}")
# project name counts
perf = pd.read_csv(app / "上市公司环境绩效明细表155529021(仅供南京审计大学使用)" / "ENV_PerfDetail.csv", dtype={"Symbol": str}, usecols=["EndDate", "Symbol", "PerfProjectName"])
perf["year"] = perf.EndDate.astype(str).str[:4]
perf = perf[perf.year.isin(["2021", "2022", "2023", "2024"])]
log(f"ENV_PerfDetail 2021-2024 rows={len(perf)} firms={perf.Symbol.nunique()}")
names = perf.PerfProjectName.astype(str)
for kw in ["投资", "处罚", "罚款", "整改", "设施", "设备", "在建", "负债", "修复", "排污费", "环保"]:
    log(f"  PerfProjectName contains {kw}: {(names.str.contains(kw, na=False)).sum()}")
log("  top project names:")
log(perf.PerfProjectName.value_counts().head(25).to_string())

# carbon
carb = pd.read_csv(app / "CSMAR碳排放量（2021-2024）.csv", dtype=str, nrows=8)
log(f"\nCSMAR carbon cols={list(carb.columns)}")
log(carb.head(3).to_string())
carb_full = pd.read_csv(app / "CSMAR碳排放量（2021-2024）.csv", dtype={"证券代码": str})
log(f"  n={len(carb_full)} code sample={carb_full['证券代码'].head(10).tolist()} lens={carb_full['证券代码'].astype(str).str.len().value_counts().to_dict()}")


# ---------- panel keys ----------
log("\n=== PANEL KEYS ===")
panel = pd.read_csv(PANEL, dtype={"证券代码": str}, encoding="utf-8-sig", usecols=["证券代码", "时间", "GI", "X_实质性披露", "Y_象征性披露", "绿色披露总量", "E得分", "Greenwashing连续_基准", "IMS"])
log(f"panel n={len(panel)} code lens={panel['证券代码'].str.len().value_counts().to_dict()} years={sorted(panel['时间'].unique())}")
log(f"  code sample={panel['证券代码'].head(8).tolist()}")
common = pd.read_csv(COMMON, dtype={"证券代码": str}, encoding="utf-8-sig", usecols=["证券代码", "时间"])
log(f"common 15381 n={len(common)}")

# merge coverage
inv = env_inv.copy()
inv["证券代码"] = inv.Symbol.str.zfill(6)
inv["时间"] = inv.EndDate.astype(str).str[:4].astype(int)
inv = inv[inv["时间"].between(2021, 2024)]
m = common.merge(inv[["证券代码", "时间", "EPInvest"]], on=["证券代码", "时间"], how="left")
log(f"common matched ENV_InvestStats non-null EPInvest={m.EPInvest.notna().sum()} ({m.EPInvest.notna().mean():.3f})")
log(f"  EPInvest>0={(pd.to_numeric(m.EPInvest, errors='coerce')>0).sum()}")

sup = env_sup.copy()
sup["证券代码"] = sup.Symbol.str.zfill(6)
sup["时间"] = sup.EndDate.astype(str).str[:4].astype(int)
sup = sup[sup["时间"].between(2021, 2024)]
m2 = common.merge(sup[["证券代码", "时间", "KeyPollMonUnit", "PollEmissStandard", "IsPassISO14001"]], on=["证券代码", "时间"], how="left")
log(f"common matched ENV_SupCert={m2.KeyPollMonUnit.notna().sum()} ({m2.KeyPollMonUnit.notna().mean():.3f})")
log(f"  KeyPollMonUnit=1={(pd.to_numeric(m2.KeyPollMonUnit, errors='coerce')==1).sum()}")
log(f"  unmatched KeyPoll={m2.KeyPollMonUnit.isna().sum()}")


# ---------- Layer B keyword sampling ----------
log("\n=== LAYER B STREAM ===")
kw_hits = {k: 0 for k in KEYWORDS}
kw_judg = {k: Counter() for k in KEYWORDS}
kw_examples = {k: [] for k in KEYWORDS}
compound_hits = {k: 0 for k in COMPOUND}
compound_judg = {k: Counter() for k in COMPOUND}
compound_ex = {k: [] for k in COMPOUND}
placebo_hits = {k: 0 for k in PLACEBO}
placebo_judg = {k: Counter() for k in PLACEBO}
placebo_ex = {k: [] for k in PLACEBO}

# noise flags
noise_templates = [
    "是否属于环境保护部门公布的重点排污单位",
    "公司是否披露环境信息",
]
n_rows = 0
n_success = 0
word_counter = Counter()
seen_sent_for_kw = {k: set() for k in KEYWORDS}

with JSONL.open("r", encoding="utf-8") as f:
    for line in f:
        n_rows += 1
        rec = json.loads(line)
        sent = rec.get("sentence") or ""
        word = rec.get("word") or ""
        judg = rec.get("judgment")
        status = rec.get("status")
        if status == "success":
            n_success += 1
        word_counter[word] += 1
        for k in KEYWORDS:
            if k in sent:
                kw_hits[k] += 1
                kw_judg[k][judg] += 1
                key = (rec.get("证券代码"), rec.get("时间"), sent[:80])
                if key not in seen_sent_for_kw[k] and len(kw_examples[k]) < 8:
                    seen_sent_for_kw[k].add(key)
                    # skip obvious template after collecting 2
                    kw_examples[k].append({
                        "code": rec.get("证券代码"),
                        "year": rec.get("时间"),
                        "word": word,
                        "judgment": judg,
                        "sentence": sent[:280],
                    })
        for k, variants in COMPOUND.items():
            if any(v in sent for v in variants):
                compound_hits[k] += 1
                compound_judg[k][judg] += 1
                if len(compound_ex[k]) < 6:
                    compound_ex[k].append({
                        "code": rec.get("证券代码"),
                        "year": rec.get("时间"),
                        "word": word,
                        "judgment": judg,
                        "sentence": sent[:280],
                    })
        for k in PLACEBO:
            if k in sent:
                placebo_hits[k] += 1
                placebo_judg[k][judg] += 1
                if len(placebo_ex[k]) < 4:
                    placebo_ex[k].append({
                        "code": rec.get("证券代码"),
                        "year": rec.get("时间"),
                        "word": word,
                        "judgment": judg,
                        "sentence": sent[:220],
                    })

log(f"n_rows={n_rows} n_success={n_success}")
log("keyword hits (sentence contains, keyword-sentence pairs):")
for k in KEYWORDS:
    tot = kw_hits[k]
    j = kw_judg[k]
    log(f"  {k}: n={tot} judgment={dict(j)} share_subst={j.get(1,0)/tot if tot else None}")
    for ex in kw_examples[k][:5]:
        log(f"    EX j={ex['judgment']} word={ex['word']} {ex['code']}-{ex['year']}: {ex['sentence']}")

log("\ncompound hits:")
for k, tot in compound_hits.items():
    j = compound_judg[k]
    log(f"  {k}: n={tot} judgment={dict(j)} share_subst={j.get(1,0)/tot if tot else None}")
    for ex in compound_ex[k][:4]:
        log(f"    EX j={ex['judgment']} word={ex['word']} {ex['code']}-{ex['year']}: {ex['sentence']}")

log("\nplacebo hits:")
for k in PLACEBO:
    tot = placebo_hits[k]
    j = placebo_judg[k]
    log(f"  {k}: n={tot} judgment={dict(j)} share_subst={j.get(1,0)/tot if tot else None}")
    for ex in placebo_ex[k][:2]:
        log(f"    EX j={ex['judgment']} {ex['code']}-{ex['year']}: {ex['sentence']}")

log("\nLayer A/B word counter for related words:")
for w, c in word_counter.most_common():
    if any(k in w for k in ["投资", "处罚", "设备", "整改", "负债", "修复", "在建", "设施", "罚款", "工程", "治理"]):
        log(f"  word={w} n={c}")

OUT.write_text("\n".join(lines_out), encoding="utf-8")
print(f"wrote {OUT} chars={OUT.stat().st_size}")
