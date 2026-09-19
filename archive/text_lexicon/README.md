# 文本词典工作归档（冻结）

归档日期：2026-09-15

本文件夹是计划一**第一阶段文本词典**的完整存档。工作区里对应的词典脚本、抽检和份额文件已删除；复现请只用这里。

结论见 `当前结果备忘.md`。

## 内容清单

**代码**

- `code/lexicon.py`：锁定词典与句级规则
- `code/build_audit_intersect.py`：Layer B 三文件并集、抽检、公司–年份额
- `code/merge_hard_env.py`：当时合并重点排污/处罚/投资（输入曾依赖 `audit_intersect_firm_year.csv`）
- `code/run_heterogeneity.py`：当时全部交互与分组回归
- `code/generate_tables.py`：当时 TeX 表

**结果**

- `当前结果备忘.md`：第一阶段文字备忘
- `results/validation/`：抽检句、准确率、锁定说明、与 GI 相关、句子覆盖
- `results/derived/audit_intersect_firm_year.csv`：公司–年可核对份额
- `results/derived/mechanism_firm_year_panel.csv`：15,381 样本 + 份额 + 硬信息
- `results/derived/regression_analysis_panel.csv`：回归用面板（含滞后 V）
- `results/derived/hard_env_coverage.csv`：硬信息覆盖
- `results/tables/`：系数 csv 与两张 TeX 表
