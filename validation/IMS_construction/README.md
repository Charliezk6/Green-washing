# IMS 数据构造检查

IMS 在全 A 股审计市场上计算，不在回归面板内部计算。一家事务所在某个专长行业、某一年的 IMS，等于它在该行业—年审计客户的营业收入之和，除以该行业—年全部正收入公司的营业收入之和。回归面板只保存匹配后的 `IMS`。

## 口径

市场文件：`X_reproduction/data/input/full_a_share_audit_market_2021_2024.csv`。

规则与 `X_reproduction/scripts/reproduce_x.py` 相同：

- 年份为 2021–2024
- 只保留 `OperatingRevenue > 0`
- 制造业保留三位 `C` 类行业代码，其余行业取行业代码首字母
- 排除 `O` 类
- 事务所用 `InstitutionID`

回归面板 `Regression/data/input/combined_greenwashing_auditor_expertise_panel.csv` 只用来标记公司—年是否进入样本。`in_regression_panel = 1` 表示该公司—年出现在回归面板中。

## 三张表

`outputs/firm_industry_year_clients.csv`

一行是一家事务所在某年、某个专长行业。`n_companies` 是全市场客户数，`n_companies_in_panel` 是其中进入回归面板的公司数。`IMS` 按上面的公式重算。

`outputs/company_industry_revenue_share.csv`

一行是一家公司—年。`revenue_share` 是该公司营业收入占该行业—年总收入的比例。同一事务所在同一行业—年的客户份额加总，等于该格的 `IMS`。

`outputs/industry_year_structure.csv`

一行是一个专长行业—年。`n_companies` 是全市场公司数，`n_audit_firms` 是参与审计的事务所数，`industry_revenue` 是行业总收入，`n_companies_in_panel` 是进入回归面板的公司数。

`outputs/construction_summary.csv` 记录行数，以及下面三条核对。

## 核对

- 每个行业—年，公司 `revenue_share` 之和与 1 的绝对差距不超过 `1e-8`。本次全部行业—年都满足，最大绝对差距约为 `2.2e-16`。
- 每个事务所—行业—年，客户 `revenue_share` 之和与该格 `IMS` 的绝对差距不超过 `1e-8`。本次全部格子都满足。
- 回归面板中能对上 `InstitutionID`—行业—年、且 `IMS` 非缺失的观测，重算 `IMS` 与面板 `IMS` 的相对误差不超过 `1e-8`。17,890 条里有 17,889 条满足。唯一例外是 `601866|2021|107023|G`：面板 `InstitutionID` 是 107023，重算 IMS 约为 0.0093；面板保存的 IMS 约为 0.3236，等于全市场文件里该公司当年审计机构 107713 在行业 G 的 IMS。

有效市场观测 19,874 条，事务所—行业—年 4,089 格，行业—年 188 格。回归面板 18,077 个公司—年里，17,951 个落在有效市场内。

## 运行

在项目根目录执行：

```powershell
python validation/IMS_construction/code/check_ims_construction.py
```
