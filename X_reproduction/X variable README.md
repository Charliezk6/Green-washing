# 审计师行业专长 X 变量复现包

本目录独立复现会计师事务所行业专长（IMS / IPS 及 20% 二元指标），供扩展漂绿面板与现行回归使用。计算逻辑已全部写在 `scripts/reproduce_x.py` 中，运行时不依赖 `archive`。

## 口径

- 以全 A 股公司—年度审计市场计算事务所行业专长；
- 事务所使用 `InstitutionID` 识别；
- 规模变量为 `OperatingRevenue`，仅保留大于零的观测；
- 制造业保留三位 `C` 类行业代码，非制造业使用行业代码首字母，排除 `O` 类；
- `IMS = FirmIndustryRevenue / IndustryYearRevenue`；
- `IPS = FirmIndustryRevenue / FirmYearRevenue`；
- `IMS_D` 与 `IPS_D` 在相应连续指标大于等于 20% 时取 1。

## 输入与输出

输入：

- `data/input/full_a_share_audit_market_2021_2024.csv`：全 A 股审计市场；
- `data/input/greenwashing_expansion_panel.csv`：扩展漂绿面板。

输出：

- `outputs/expertise/auditor_industry_expertise_2021_2024.csv`；
- `outputs/company_year/greenwashing_expansion_with_auditor_expertise.csv`；
- `outputs/reports/summary.json`；
- `outputs/validation/summary.json`。

扩展面板有 8,982 条公司—年度样本，其中 8,838 条成功匹配 X；其余 144 条保留缺失，而不填充为零。

## 运行

```powershell
python scripts/reproduce_x.py
python scripts/validate_reproduction.py
python tests/test_reproduction_smoke.py
```
