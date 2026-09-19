# 审计师行业专长与企业漂绿回归包

本目录是当前研究的**独立、完整且可直接使用的回归复现包**。后续回归、描述统计、相关性分析、论文格式表格和校验均以 `Regression/` 为唯一入口。

## 研究样本与数据来源

本包的输入面板由两部分公司—年度样本合并而成：

- 原研究面板：9,239 条公司—年度记录；
- DeepGreen 扩展面板：8,838 条已成功匹配审计师行业专长的公司—年度记录；
- 合并输入面板：18,077 条唯一公司—年度记录；
- 完整控制变量共同样本：15,381 条公司—年度记录（4,604 家公司，2021–2024，47 个行业）。

扩展面板的绿色披露变量来自 `Y_variable_reproduction/LLM_classification/deepgreen_reproduction3/`，审计师行业专长来自 `X_reproduction2/`。全市场收入增长率输入已复制至 `data/input/`。

合并后面板经 `scripts/correct_combined_panel.py` 统一校正：SOE 按实际控制人类型、同年全部控制人记录补全后重编，连续漂绿在合并样本上按同一行业—年度口径重算。旧面板备份在 `validation/panel_version_audit/data/combined_panel_pre_correction.csv`；补全前面板备份在 `validation/panel_version_audit/data/combined_panel_pre_soe_impute.csv`。

## 变量与口径

### 因变量

`Greenwashing连续_基准` 在**合并后的公司—年样本**上构造：行业—年度内 `GI` 的标准分减去环境 E 得分的标准分。分组为 `时间 × ExpertiseIndustryCode`（制造业证监会二级、非制造业一级），使用样本标准差 `ddof=1`。组内有效观测不超过 1 或标准差为 0 时记为缺失。

### 核心解释变量

- `IMS`：事务所在其客户所在行业—年度的营业收入市场份额；
- `IMS_D`：`IMS >= 0.20` 时取 1，否则取 0（主结果只使用该 20% 切点，不报告 10% 切点）；
- `IPS` 与 `IPS_D` 作为备选指标保留于输入面板。

### 行业口径

- 制造业：证监会二级行业代码；
- 非制造业：证监会一级行业代码；
- 回归固定效应与 Y 标准化均使用 `ExpertiseIndustryCode`。

### 控制变量

财务控制：`Size`、`Lev`、`ROA`、`CFO`、`Growth`、`FirmAge`（上市年限，不取对数）。

治理控制：`SOE`、`Dual`、`BoardSize`、`IndependentDirectorRatio`。

审计控制：`Big4`、`LnAuditFee`、`AuditorTenure`、`IsChangeFirm`。

`SOE = 1` 当同年 `HLD_Contrshr` 全部实际控制人记录中任一 `S0702b ∈ {1100, 2000, 2100, 2120}`；无实际控制人记 0；公司内性质不变且间隔不超过 2 年时前后填补；仍缺的残差记 0，不删样本。不使用实际控制人姓名是否非空。

营业收入增长率 `Growth` 始终在全市场收入数据上根据连续年度滞后收入构造，非正滞后收入或年份不连续时保持缺失。

连续控制变量在共同样本内按 1%/99% 缩尾；不缩尾因变量与 `IMS`。

## 回归规格

分别以 `IMS` 和 `IMS_D`（≥0.20）为核心解释变量，估计：

1. 一元回归；
2. 加财务控制；
3. 加财务与治理控制；
4. 加全部控制。

以上规格均含年度与行业固定效应，并按公司聚类标准误。全部模型使用相同的 15,381 条共同样本。

## 目录结构

```text
Regression/
├── regression README.md
├── requirements.txt
├── config/
│   └── regression_config.json
├── data/input/
│   ├── combined_greenwashing_auditor_expertise_panel.csv
│   └── full_market_financial_revenue_2020_2024.csv
├── scripts/
│   ├── build_combined_panel.py
│   ├── correct_combined_panel.py
│   ├── reproduce_regression.py
│   ├── prepare_empirical_tables.py
│   ├── generate_latex_tables.py
│   └── validate_reproduction.py
├── tests/
│   └── test_regression_smoke.py
└── outputs/
    ├── baseline/
    ├── paper_tables/
    └── validation/
```

## 运行

安装依赖：

```powershell
pip install -r requirements.txt
```

从已合并面板校正、回归、制表到验证：

```powershell
python scripts/correct_combined_panel.py
python scripts/reproduce_regression.py
python scripts/prepare_empirical_tables.py
python scripts/generate_latex_tables.py
python scripts/generate_paper_tables.py
python scripts/validate_reproduction.py
```

若需从原面板与扩展面板重新合并，再运行 `python scripts/build_combined_panel.py`（依赖项目根目录下的 `appendix/` 与 X 扩展输出），然后必须再次运行 `correct_combined_panel.py` 以统一 Y。

执行烟雾测试：

```powershell
python tests/test_regression_smoke.py
```

## 关键输出

- 回归系数：`outputs/baseline/基准回归完整系数.csv`
- 回归样本信息：`outputs/baseline/基准回归样本信息.csv`
- 描述统计：`outputs/paper_tables/descriptive_statistics_15381.csv`
- 共同样本：`outputs/paper_tables/common_sample_15381.csv`
- LaTeX 表格：`outputs/paper_tables/*.tex`
- 面板校正摘要：`outputs/validation/panel_correction_summary.json`
