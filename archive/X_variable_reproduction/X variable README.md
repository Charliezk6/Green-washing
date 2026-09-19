# 审计师行业专长 X 变量复现包

参考核心文献：[1]蔡春,鲜文铎.会计师事务所行业专长与审计质量相关性的检验——来自中国上市公司审计市场的经验证据[J].会计研究,2007,(6):41-47+95.

## 1. 用途

本文件夹用于独立复现研究的核心解释变量——会计师事务所行业专长。只需保留本文件夹，即可从 2021—2024 年全 A 股公司—年度审计市场数据重新计算事务所—行业—年度层面的 `IMS`、`IMS_D`、`IPS`、`IPS_D`，再连接到具有漂绿 Y 变量的公司年度研究样本。

本包集中保存核心输入、当时实际使用的参考结果、统一复现脚本和测试，因此以后不需要依赖历史对话来理解 X 的形成过程。

## 2. 为什么使用审计师行业专长

审计机构在某行业拥有更多客户和业务规模时，通常更熟悉该行业的经营模式、监管要求、会计估计、环境风险和披露惯例，也更容易发现企业披露与实际表现之间的不一致。本研究因此用会计师事务所层面的行业专长解释企业漂绿程度。

这里的“审计师”实际指公司年度审计报告对应的**会计师事务所**，识别主键为 `InstitutionID`；不是签字注册会计师个人专长。

行业专长必须先在全 A 股审计市场中计算，再连接到漂绿样本。若只用拥有 Y 变量的公司计算市场份额，X 会受到最终研究样本选择的影响，不再代表事务所面对的完整市场。

## 3. 数据流

```text
全 A 股公司年度审计市场
→ 清理有效事务所、行业与营业收入记录
→ 构造混合行业代码 ExpertiseIndustryCode
→ 汇总事务所—行业—年度客户收入
→ 计算行业年度总收入与事务所年度客户总收入
→ 计算 IMS、IPS 及 10% 二元指标
→ 按年度、事务所和行业连接到漂绿公司年度面板
→ 输出公司年度 X 变量并与历史结果校验
```



## 4. 核心输入



### `data/input/full_a_share_audit_market_2021_2024.csv`

计算行业专长的全市场基础。每行是一家公司年度，主要字段为：

- `证券代码`、`时间`
- `AccountingFirmName`、`InstitutionID`
- `IndustryName`、`IndustryCode`
- `OperatingRevenue`

虽然文件还保留资产、负债和利润等字段，本次专长计算使用的是 `OperatingRevenue`。

### `data/input/greenwashing_auditor_regression_panel_2021_2024.csv`

尚未加入行业专长指标的漂绿研究样本，包含 Y 变量、公司审计信息和后续回归所需控制变量。复现脚本将事务所行业年度指标连接到该面板。

### `data/final_reference/`

保存当时实际使用的两层结果：

- `auditor_industry_expertise_2021_2024.csv`：事务所—行业—年度专长表；
- `greenwashing_panel_with_auditor_expertise_2021_2024.csv`：加入 X 后的公司年度面板。

参考文件只用于验收，不参与 X 指标计算。

## 5. 样本清理口径

从全市场文件构造 X 时，只保留：

1. 年份为 2021—2024；
2. `InstitutionID`、行业代码和营业收入可用；
3. `OperatingRevenue > 0`；
4. 行业代码不属于 `O`。

营业收入为零、负数或缺失的记录不参与分子和分母。历史数据中存在一条负营业收入记录；排除该记录后才能精确重建原专长结果。`O` 类仅有少量观察，历史专长表未纳入，因此本包明确延续这一口径。

每家公司年度在全市场输入中必须唯一，事务所以稳定的 `InstitutionID` 识别，名称主要用于展示。

## 6. 混合行业口径

行业专长使用 `ExpertiseIndustryCode`：

```text
制造业：保留三位行业代码，如 C13、C27、C37
非制造业：取 IndustryCode 首字母，如 A、B、D、E、K
O 类：排除
```

这意味着制造业按细分行业衡量专长，非制造业按行业门类衡量专长。该规则与当前研究面板中的“混合行业口径”一致。改变行业层级会改变市场分母和事务所专长，必须作为新版本处理，不能静默覆盖当前结果。

## 7. 变量公式

令公司为 i，事务所为 a，行业为 j，年度为 t，营业收入为 Revenue。

### 7.1 事务所行业客户收入

```text
FirmIndustryRevenue(a,j,t)
= 事务所 a 在行业 j、年度 t 的所有有效 A 股审计客户营业收入之和
```



### 7.2 行业年度市场收入

```text
IndustryYearRevenue(j,t)
= 行业 j 在年度 t 的所有有效 A 股公司营业收入之和
```



### 7.3 事务所年度客户收入

```text
FirmYearRevenue(a,t)
= 事务所 a 在年度 t 的所有有效 A 股审计客户营业收入之和
```



### 7.4 IMS：行业市场份额


IMS_{a,j,t}=\frac{FirmIndustryRevenue_{a,j,t}}{IndustryYearRevenue_{j,t}}


`IMS` 从行业市场角度衡量事务所在该行业中的影响力。值越大，表示该事务所审计的客户营业收入占该行业市场的份额越高。

### 7.5 IPS：事务所业务组合份额


IPS_{a,j,t}=\frac{FirmIndustryRevenue_{a,j,t}}{FirmYearRevenue_{a,t}}


`IPS` 从事务所自身角度衡量该行业在其全部客户组合中的重要性。

### 7.6 二元专长指标

现有结果反推并逐条验证后的真实规则为固定 10% 阈值：

```text
IMS_D = 1，当且仅当 IMS >= 0.10；否则为 0
IPS_D = 1，当且仅当 IPS >= 0.10；否则为 0
```

该规则不是行业第一名、分位数或四大差异阈值。当前 4,089 条事务所行业年度记录中，`IMS_D=1` 有 531 条，`IPS_D=1` 有 500 条。

## 8. 公司年度连接规则

计算完事务所层面指标后，使用以下三个键连接到研究样本：

```text
时间
InstitutionID
ExpertiseIndustryCode
```

只有成功连接行业专长的公司年度进入最终 X 面板。输入研究样本有 9,370 条，最终成功连接 9,239 条；其余 131 条没有有效专长匹配，不被强行填为零。

最终核心 X 字段为：


| 字段                      | 含义                   |
| ----------------------- | -------------------- |
| `ExpertiseIndustryCode` | 行业专长计算使用的混合行业代码      |
| `IMS`                   | 事务所在行业年度市场中的客户营业收入份额 |
| `IMS_D`                 | `IMS >= 10%` 的二元指标   |
| `IPS`                   | 该行业占事务所年度全部客户营业收入的份额 |
| `IPS_D`                 | `IPS >= 10%` 的二元指标   |


当前回归代码主要使用 `IMS` 和 `IMS_D` 作为核心解释变量；`IPS` 和 `IPS_D` 保留为替代专长度量。

## 9. 文件结构

```text
X_variable_reproduction/
├── README.md
├── config/reproduction_config.json
├── data/
│   ├── input/
│   └── final_reference/
├── scripts/
│   ├── reproduce_x.py
│   └── validate_reproduction.py
├── outputs/
│   ├── expertise/
│   ├── company_year/
│   └── validation/
└── tests/test_reproduction_smoke.py
```



## 10. 一键复现

要求 Python 3.10 或更高版本。核心脚本只使用 Python 标准库，无须安装 pandas。

在本文件夹运行：

```bash
python scripts/reproduce_x.py
```

输出：

```text
outputs/expertise/auditor_industry_expertise_2021_2024.csv
outputs/company_year/greenwashing_panel_with_auditor_expertise_2021_2024.csv
outputs/validation/summary.json
```

只生成结果、不与参考文件比较：

```bash
python scripts/reproduce_x.py --skip-validation
```

单独验证已有输出：

```bash
python scripts/validate_reproduction.py
```



## 11. 测试

运行：

```bash
python tests/test_reproduction_smoke.py
```

测试会确认：

- 主脚本能够完整运行；
- 事务所—行业—年度主键唯一；
- 公司—年度主键唯一；
- `IMS`、`IPS` 位于 `[0,1]`；
- `IMS_D`、`IPS_D` 只包含 0/1；
- 两层输出与历史参考结果一致。

营业收入合计涉及大额浮点加法，遍历顺序不同可造成小于 0.02 元的累计误差；校验对收入合计使用 0.02 元容差，对连续专长指标和公司年度结果使用近乎零的严格容差。该误差不会改变 IMS、IPS 或二元分类。

## 12. 不可随意修改的口径

1. 必须使用全 A 股审计市场计算 X，不能只使用漂绿研究样本。
2. 规模变量是营业收入，不是总资产、审计费用或客户数量。
3. 只纳入严格大于零的营业收入，缺失或非正数不能填零。
4. 制造业使用三位细分代码，非制造业使用首字母门类代码，并排除 O 类。
5. 事务所主键是 `InstitutionID`，不能仅按事务所名称聚合。
6. `IMS_D` 和 `IPS_D` 的阈值均为 `>= 0.10`。
7. 无法匹配专长的公司年度保持缺失，不能编码为零专长。
8. 如果改变规模变量、行业层级、阈值或样本边界，应建立新的 X 版本并重新进行稳健性分析。

