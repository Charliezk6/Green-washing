# 因变量测度稳健性

本目录只回答导师第 8–10 条：现有因变量 `Greenwashing连续_基准 = GI_z − ESGE_z` 能不能被解释为行动声称—绩效缺口。不改 `Regression/` 的主规格、主面板和已有结果。

回归共同样本固定为 15,381 个公司—年。控制变量、年度固定效应 `C(时间)`、行业固定效应 `C(ExpertiseIndustryCode)`、按 `证券代码` 聚类，与 [Regression/config/regression_config.json](../Regression/config/regression_config.json) 的完整模型一致。核心解释变量只报 `IMS` 与 `IMS_D`（`IMS ≥ 0.20`）。

## 目录

```text
measurement Y robustness/
├── README.md
├── notes/
│   └── 口径说明.md
├── code/
│   ├── common.py                 # 路径、行业—年 z 分数、完整模型
│   ├── 01_adverse_share.py       # 实质性句子中的不利表述占比
│   ├── 02_gi_plus.py             # 净正面 GI+ 与 Greenwashing+
│   ├── 03_component_split.py     # GI_z 与 ESGE_z 拆开
│   └── 04_penalty_gap.py         # 用环保处罚替换华证得分
├── results/
│   ├── derived/                  # 公司—年面板与覆盖率
│   └── tables/                   # 系数表、占比表
└── tex/                          # 可放进附录的表
```

`notes/口径说明.md` 是变量定义。上面的 `code/`、`results/`、`tex/` 在执行时按该口径写入，不另起一套标准化规则。

## 运行顺序

在仓库根目录执行：

```text
python "measurement Y robustness/code/01_adverse_share.py"
python "measurement Y robustness/code/02_gi_plus.py"
python "measurement Y robustness/code/03_component_split.py"
python "measurement Y robustness/code/04_penalty_gap.py"
```

后三个脚本都读 `01` 不必先跑完才能做拆解或处罚替换。`02` 依赖 `01` 写出的句子级不利标记。

## 每个脚本做什么

| 脚本 | 回答的问题 | 主要输入 | 主要输出 |
|---|---|---|---|
| `01_adverse_share.py` | 被判为具体行动的句子里，有多少是在交代处罚、超标或违规 | 三份 Layer B jsonl | `results/tables/adverse_share.csv`；`results/derived/layer_b_substantive_flags.csv` |
| `02_gi_plus.py` | 把不利具体句移出分子后，专长系数是否仍为负 | 上一文件的标记；共同样本 | `results/tables/gi_plus_regression.csv` |
| `03_component_split.py` | 负向关联落在声称侧还是绩效侧 | 共同样本上的 `GI_z_行业年度`、`ESGE_z_行业年度` | `results/tables/component_split.csv` |
| `04_penalty_gap.py` | 绩效侧改成监管处罚后，结论是否还在 | `appendix/上市公司环保处罚总表.xlsx`；共同样本 | `results/tables/penalty_gap_regression.csv` |

## 输入位置

- 共同样本：`Regression/outputs/paper_tables/common_sample_15381.csv`
- Layer B：`Y_variable_reproduction/data/input/manufacturing/layer_b_results.jsonl`、`non_manufacturing/layer_b_results.jsonl`、`expansion/layer_b_results.jsonl`
- 处罚明细：`appendix/上市公司环保处罚总表.xlsx`
- 处罚汇总规则与 `validation/P1/mechanism_verifiability/code/merge_hard_env.py` 的 `load_penalty()` 相同：2021–2024，按证券代码—年加总次数与金额（万元），未匹配记 0

三份 jsonl 按 `key`（`证券代码_年份`）合并。同一 `key` 加 `sentence_id` 加 `word` 重复出现时保留一条，并在覆盖率表里报告冲突条数。

## 怎样算这项测度稳健

四项结果同时满足下面的条件，才把现有因变量写成可辩护的行动声称—绩效缺口。缺任何一项，正文只报告该项实际结果，不把它写成已经通过。

1. **不利具体句占比低。** `judgment = 1` 且 `status = success` 的关键词—句子对中，命中不利词表的比例低于 5%。高于 5% 时，以 `GI+` 回归为主证据，并在正文报告该比例。
2. **净正面缺口仍为负。** `Greenwashing+` 上 `IMS` 与 `IMS_D` 的完整模型系数为负，且至少在 10% 水平显著。系数方向翻转则不能用“高 GI 主要是正面行动声称”来解释主结果。
3. **声称侧有独立变化。** 以 `GI_z_行业年度` 为因变量、控制 `ESGE_z_行业年度` 后，`IMS` 或 `IMS_D` 仍为负。若负向完全来自 `ESGE_z` 上升，主结果只说明专家客户的华证环境分更高。
4. **外生绩效侧方向一致。** 二元处罚缺口上 `IMS` 与 `IMS_D` 为负。连续处罚缺口因组内无处罚变异会少样本，表注单独报告样本量；它不替代二元结果。

排放、碳强度、绿色专利、预计负债不进入本目录的回归。取舍写在 `notes/口径说明.md` 末节。
