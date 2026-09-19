# 漂绿 Y 变量复现包

本目录是当前研究使用的漂绿 Y 变量独立复现包。它从已经完成的 `DeepGreen Layer B` 分类结果出发，确定性地重建 2021—2024 年扩展样本的公司—年度绿色披露指标、GI、二元漂绿变量和连续漂绿变量；不需要再次调用 LLM 或 API。

## 1. 研究样本与边界

- 候选公司—年度样本：8,982 条；
- Layer B 有有效分类的公司—年度：7,325 条；
- GI 完整样本：7,325 条；
- 二元漂绿变量完整样本：7,258 条；
- 年份范围：2021—2024；
- 模型分类结果由 `qwen3.6-35b-a3b-fp8` 在上游 DeepGreen 工程中产生，并在本包中固化保存。

技术失败、无有效关键词—句子分类对或无法满足变量构造条件的样本均保留缺失，绝不填充为零。

## 2. 数据流

```text
上市公司年报
→ 环境文本抽取、分句与 S1
→ Layer A 绿色关键词识别
→ S2 关键词—完整句子对
→ Layer B：实质性 / 象征性判断
→ 本包：X / Y / GI / 行业年度基准 / Greenwashing
```

本包只覆盖最后的确定性变量构造。完整文本工程保存在 `LLM_classification/deepgreen_reproduction3`；本包已将变量构造所需的 Layer B 明细、样本元数据及行业参考输入复制到 `data/input/expansion/`，便于归档与独立重建。

## 3. 变量定义

对每个公司—年度，Layer B 的每条绿色“关键词—完整句子对”有一个判断：

- `judgment=1`：实质性披露；
- `judgment=0`：象征性披露。

公司—年度汇总为：

```text
X_实质性披露 = judgment=1 的关键词—句子对数量
Y_象征性披露 = judgment=0 的关键词—句子对数量
绿色披露总量 = X + Y
GI = X / (X + Y)
```

同一句的不同绿色关键词可分别贡献一条记录，故计数单位是“关键词—句子对”，不是唯一句子。

## 4. 行业基准与漂绿指标

行业基准在每个年度内计算，使用混合行业层级：

- 制造业：使用证监会二级行业代码；
- 非制造业：使用证监会一级行业代码；
- 分组字段：`行业基准分组`。

对每个“年度 × 行业基准分组”计算 `GI_mean`、`ESGE_mean`、`GI_sd` 与 `ESGE_sd`。

二元指标：

```text
Greenwashing = 1
当且仅当 GI > GI_mean 且 E得分 < ESGE_mean
否则为 0
```

连续指标：

```text
Greenwashing连续 = (GI - GI_mean) / GI_sd - (E得分 - ESGE_mean) / ESGE_sd
```

连续指标使用行业—年度组内**样本标准差**（`ddof=1`）。当组内 GI 或 E 得分标准差为零时，连续指标保留缺失，并将 `连续变量构造状态` 标记为 `insufficient_industry_year_variation`。

## 5. 文件结构

```text
Y_variable_reproduction/
├── Y variable README.md
├── config/reproduction_config.json
├── data/
│   └── input/
│       └── expansion/
│           ├── sample_metadata.csv
│           ├── layer_b_results.jsonl
│           ├── greenwashing_panel_expansion_reference.csv
│           └── industry_year_benchmarks_reference.csv
├── scripts/
│   ├── reproduce_y.py
│   └── validate_reproduction.py
├── tests/
│   └── test_reproduction_smoke.py
└── outputs/
    ├── combined/greenwashing_panel_expansion.csv
    ├── industry_benchmarks/industry_year_benchmarks.csv
    ├── reports/summary.json
    └── validation/summary.json
```



## 6. 运行与验证

Python 3.10+ 即可运行，核心脚本只使用标准库：

```powershell
python scripts/reproduce_y.py
python scripts/validate_reproduction.py
python tests/test_reproduction_smoke.py
```

校验检查：样本数、公司—年份唯一性、GI 值域、二元变量合法性、制造业二级/非制造业一级行业层级，以及预期的完整样本数与漂绿分布。

## 7. 当前版本结果


| 指标                | 数量    |
| ----------------- | ----- |
| 候选样本              | 8,982 |
| GI 完整样本           | 7,325 |
| Greenwashing 完整样本 | 7,258 |
| `Greenwashing=1`  | 2,110 |
| `Greenwashing=0`  | 5,148 |




## 8. 不可随意改变的口径

1. X、Y 统计单位为关键词—句子对。
2. 技术失败不能编码为零披露。
3. 制造业保持二级行业，非制造业使用一级行业。
4. 二元变量必须使用严格不等式 `>` 和 `<`。
5. 连续变量使用样本标准差；标准差为零时必须保持缺失。
6. 行业专长、回归控制变量和回归分析均不在本包内构造。

