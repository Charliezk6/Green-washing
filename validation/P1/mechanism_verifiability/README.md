## 脚本说明

### `merge_hard_env.py`

把硬环境信息左连到基准共同样本`面板`。

- 重点排污单位、ISO14001：CSMAR `ENV_SupCertDisclose.csv` 
- 环保投资：CSMAR `ENV_InvestStats.csv` 
- 环保处罚：`上市公司环保处罚总表.xlsx`，按证券代码–年汇总，生成 `Penalty_D`

写出：

- `results/derived/mechanism_firm_year_panel.csv`
- `results/derived/hard_env_coverage.csv`

### `merge_balance_sheet.py`

在上一文件的 15,381 行上合并合并报表（`Typrep=A`）年报科目与资本支出。

- 固定资产净额 `PPE` = `A001212000`
- 在建工程 `CIP` = `A001213000`
- 预计负债 `Provision` = `A002207000`
- 资本支出 `Outcap`；负值当作缺失
- 主规格将资产负债表空白与 capex 缺失填 0，并保留 `*_observed`
- 比率相对 `TotalAssets`；`HighCapex` / `HighCIP` 按 `ExpertiseIndustryCode`–年中位数构造

写出：

- `results/derived/financial_crosscheck_panel.csv`
- `results/derived/financial_crosscheck_coverage.csv`



### `hetero_common.py`

三条回归脚本的共用函数：

- `load_analysis_panel()`：基准共同样本（要求恰好 15,381 行）左连 KeyPoll/处罚与财务交叉面板，并对比率调节做 1%/99% 缩尾；相邻年份才构造 `L_KeyPollMonUnit`
- `constrained_formula` / `full_interact_formula` / `subsample_formula`
- `run_three_layer`、`run_split`、公司聚类 OLS



### `run_keypoll_full_interact.py`

对应 pdf 表 2。

调节：`KeyPollMonUnit`、`L_KeyPollMonUnit`、剔除 `Penalty_D=1` 的年份、`Penalty_D`。
KeyPoll 另估完全交互 + 行业×年固定效应。

写出：`results/tables/keypoll_full_interact_coefficients.csv`

### `run_keypoll_advisor_diagnostics.py`

重点排污单位的补充估计。完全交互在这个脚本里单独写公式：控制变量、年份固定效应和行业固定效应都与调节变量相乘，不调用 `full_interact_formula`。

除 `S×V` 以外的交互项若联合显著，推断用该完全交互的 `β`、`γ` 和 `β+γ`；否则退回约束交互。`β+γ` 用聚类协方差上的线性组合检验。

写出：

- `results/tables/keypoll_advisor_slopes.csv`
- `results/tables/keypoll_advisor_lincom.csv`
- `results/tables/keypoll_full_other_terms.csv`
- `results/tables/keypoll_ims_correlation.csv`
- `tex/keypoll_advisor_tables.tex`

### `run_financial_hetero.py`

对应 pdf 表 3（财务交叉）。调节平行进入、不同时进同一方程：

`Capex_TA`、`HighCapex`、`CIP_TA`、`HighCIP`、`Provision_D`、`Provision_TA`、`PPE_TA`，
以及 capex/CIP/预计负债非缺失子样本和滞后 `L_Capex_TA` / `L_CIP_TA`。

写出：`results/tables/financial_hetero_coefficients.csv`

### `run_lit_hetero.py`

对应 pdf 表 4（文献导向异质性）。在分析面板上现场构造：

- `HeavyPoll`：环保部核查名录 + CAS 点名公用事业的 `IndustryCode`
- `ShortTenure`：`AuditorTenure ≤ 3`
- `CI`：客户审计费 / 同一事务所–年审计费合计；`HighCI` 为高于样本中位数

写出：

- `results/tables/lit_hetero_coefficients.csv`
- `results/tables/lit_hetero_coverage.csv`
- `results/derived/lit_moderators_panel.csv`

未估计：媒体关注、省级环境规制（无匹配数据）。

因变量 `Greenwashing连续_基准`。控制变量与基准回归相同。
年份 FE 为 `C(时间)`，行业 FE 为 `C(ExpertiseIndustryCode)`。
若调节已在控制集中（如 `SOE`），从控制向量中剔除后再进入主效应与交互。
财务交叉和文献调节里，组间斜率差看只让控制变量与 `V` 相乘的交互项；分组回归只报告各组 `IMS` / `IMS_D` 系数。
重点排污单位另按 `run_keypoll_advisor_diagnostics.py`：其他交互项联合显著时，推断改用含固定效应交互的完全交互，重点组效应是 `β+γ`。