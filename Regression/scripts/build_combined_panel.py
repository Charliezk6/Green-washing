from __future__ import annotations

import csv
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
PROJECT = PACK.parent
OLD = PACK / "data/input/original_greenwashing_auditor_expertise_panel.csv"
NEW = PROJECT / "X_reproduction2/outputs/company_year/greenwashing_expansion_with_auditor_expertise.csv"
APP = PROJECT / "appendix"
OUT = PACK / "data/input/combined_greenwashing_auditor_expertise_panel.csv"
SOE_CODES = {1100, 2000, 2100, 2120}


def soe_from_s0702b(value: object) -> int:
    text = "" if value is None else str(value).strip()
    if not text:
        return 0
    try:
        code = int(float(text))
    except ValueError:
        return 0
    return 1 if code in SOE_CODES else 0


def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row): return (row["证券代码"].zfill(6), row["时间"])
def date_key(row, code="Symbol", date_col="EndDate"):
    return (row[code].zfill(6), row[date_col][:4])


def index(path, code="Symbol", date_col="EndDate"):
    return {date_key(r, code, date_col): r for r in rows(path) if r.get(code) and r.get(date_col)}


def main():
    old = rows(OLD)
    new = rows(NEW)
    market = index(PROJECT / "X_variable_reproduction/data/input/full_a_share_audit_market_2021_2024.csv", code="证券代码")
    audit = index(APP / "上市公司审计机构列表153957251(仅供南京审计大学使用)/AR_LISTCOMPAUDIT.csv")
    opinion = index(APP / "财务报告审计意见信息表(年)151014682(仅供南京审计大学使用)/AIQ_FinAuditOpinY.csv")
    manager = index(APP / "高管人数、持股及薪酬情况表153539082(仅供南京审计大学使用)/CG_ManagerShareSalary.csv", date_col="Enddate")
    governance = index(APP / "治理综合信息文件153624176(仅供南京审计大学使用)/CG_Ybasic.csv", code="Stkcd", date_col="Reptdt")
    holder = index(APP / "上市公司控制人文件152909828(仅供南京审计大学使用)/HLD_Contrshr.csv", code="Stkcd", date_col="Reptdt")
    company = {r["Stkcd"].zfill(6): r for r in rows(APP / "公司基本情况文件154741304(仅供南京审计大学使用)/CG_Co.csv")}
    fields = list(old[0])
    out = old.copy(); old_keys = {key(r) for r in old}; added=0; incomplete=0
    for n in new:
        k=key(n)
        if k in old_keys or n.get("X变量匹配状态")!="complete": continue
        m,a,o,g,h,co = market.get(k,{}),audit.get(k,{}),opinion.get(k,{}),governance.get(k,{}),holder.get(k,{}),company.get(k[0],{})
        d=manager.get(k,{})
        try: age=int(k[1])-int(co["ListedDate"][:4])
        except: age=""
        director=d.get("DirectorNumber",""); indep=d.get("IndependentDirectorNumber","")
        try: ratio=float(indep)/float(director)
        except: ratio=""
        y=g.get("Y1001b","")
        record={f:"" for f in fields}; record.update({f:n.get(f,"") for f in fields})
        record.update({"IndustryName":m.get("IndustryName",n.get("证监会行业新","")),"IndustryCode":m.get("IndustryCode",""),"IndustryNameD":m.get("IndustryName",""),"IndustryCodeD":m.get("IndustryCode",""),"Greenwashing连续_基准":n.get("Greenwashing连续",""),"TotalAssets":m.get("TotalAssets",""),"TotalLiabilities":m.get("TotalLiabilities",""),"OperatingRevenue":m.get("OperatingRevenue",""),"NetProfit":m.get("NetProfit",""),"OperatingNetCashFlow":m.get("OperatingNetCashFlow",""),"AuditDate":o.get("AuditDate",""),"TypeAuditOpin":o.get("TypeAuditOpin",""),"Auditor":o.get("Auditor",""),"TerritoryAccountingFirm":o.get("TerritoryAccountingFirm",""),"InternationalBig4":o.get("InternationalBig4",""),"DomesticTop10":o.get("DomesticTop10",""),"TotalAuditFee":o.get("TotalAuditFee",""),"AccountingFirmName":a.get("AccountingFirmName",n.get("AccountingFirmName","")),"InstitutionID":a.get("InstitutionID",n.get("InstitutionID","")),"IsChangeFirm":a.get("IsChangeFirm",""),"AuditorTenure":a.get("AuditorTenure",""),"IsChangeCPA":a.get("IsChangeCPA",""),"S0701b":h.get("S0701b",""),"S0702b":h.get("S0702b",""),"Y1001b":y,"DirectorNumber":director,"IndependentDirectorNumber":indep,"SOE":soe_from_s0702b(h.get("S0702b","")),"Dual":1 if y=="1" else 0 if y=="2" else "","IndependentDirectorRatio":ratio,"Big4":1 if o.get("InternationalBig4")=="1" else 0 if o.get("InternationalBig4") else "","ListedDate":co.get("ListedDate",""),"ListingYear":co.get("ListedDate","")[:4],"FirmAge":age,"IMS":n.get("IMS",""),"IMS_D":n.get("IMS_D",""),"IPS":n.get("IPS",""),"IPS_D":n.get("IPS_D",""),"ExpertiseIndustryCode":n.get("ExpertiseIndustryCode","")})
        out.append(record); added+=1
        if any(record.get(x,"")=="" for x in ["Greenwashing连续_基准","IMS","TotalAuditFee","AuditorTenure","IsChangeFirm","DirectorNumber","SOE","Dual","IndependentDirectorRatio","FirmAge"]): incomplete+=1
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(out)
    print({"original":len(old),"added":added,"added_with_missing_controls":incomplete,"combined":len(out)})

if __name__=="__main__": main()
