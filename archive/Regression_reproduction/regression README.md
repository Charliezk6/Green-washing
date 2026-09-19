# 基准回归复现包

## 1 用途

本文件夹用于独立复现审计师事务所行业专长对企业连续漂绿变量影响的基准回归。它固定保存回归需要的最小输入、变量构造脚本、回归规格、参考结果和测试，因此复制到其他本地环境后即可重新生成结果。

研究逻辑为：

```text
Y\\\\\\\_variable\\\\\\\_reproduction → Greenwashing连续\\\\\\\_基准
X\\\\\\\_variable\\\\\\\_reproduction → IMS / IMS\\\\\\\_D
企业财务与治理数据 → 控制变量
→ 共同样本筛选
→ OLS 基准回归
```

当前复现的是 `regression/回归结果/基准回归\\\\\\\_8129共同样本`，不是“样本逐步回归筛选”版本。

## 2 数据与变量

主输入 `data/input/greenwashing\\\\\\\_auditor\\\\\\\_expertise\\\\\\\_panel.csv` 已包含 Y、X、企业财务、治理和审计字段。`data/input/full\\\\\\\_market\\\\\\\_financial\\\\\\\_revenue\\\\\\\_2020\\\\\\\_2024.csv` 是从全市场财务数据抽出的最小收入文件，用于计算营业收入增长率；保留 2020 年是为了计算 2021 年的滞后收入。

因变量：

```text
Greenwashing连续\\\\\\\_基准 = 行业年度 GI 标准分 - 行业年度 E 得分标准分
```

核心解释变量：

- `IMS`：事务所行业客户收入占行业年度总收入的比例；
- `IMS\\\\\\\_D`：`IMS >= 0.10` 时为 1，否则为 0。

控制变量：

```text
财务：Size, Lev, ROA, CFO, Growth, FirmAgeLog
治理：SOE, Dual, BoardSize, IndependentDirectorRatio
审计：Big4, LnAuditFee, AuditorTenure, IsChangeFirm
```

变量变换：

```text
Size = ln(TotalAssets)
Lev = TotalLiabilities / TotalAssets
ROA = NetProfit / TotalAssets
CFO = OperatingNetCashFlow / TotalAssets
Growth = Revenue\\\\\\\_t / Revenue\\\\\\\_(t-1) - 1
FirmAgeLog = ln(1 + FirmAge)
BoardSize = ln(DirectorNumber)
LnAuditFee = ln(TotalAuditFee)
```

正值条件不满足、年份不连续、分母缺失或结果为无穷时保留为缺失，不填零。`Growth` 在全市场收入数据上先计算，再连接回回归面板，不能只在最终回归样本内计算。

### 控制变量与识别设计的文献依据

控制变量的设置综合企业漂绿、审计师行业专长与审计质量研究，而不是将全部变量归因于单一文献：

- Marquis, Toffel, and Zhou (2016) 是漂绿研究及一般公司特征控制的核心顶刊依据。该文将漂绿刻画为选择性环境披露，并控制可能影响环境披露与实际环境表现的企业异质性和制度环境；本文据此控制 `Size`、`Lev`、`ROA`、`CFO`、`Growth`、`FirmAgeLog` 等财务特征以及公司治理特征，并纳入年度和行业固定效应。
- Reichelt and Wang (2010) 是审计师行业专长识别的核心顶刊依据。该文使用基于客户业务规模的行业市场份额衡量事务所行业专长，并区分行业专长与事务所一般品牌或规模效应；本文据此使用 `IMS` 和 `IMS_D` 衡量行业专长，并控制 `Big4`。
- DeFond and Zhang (2014) 是审计质量及其他审计关系特征的顶刊综述依据。该文系统梳理审计质量的投入、过程与结果衡量，讨论审计费用、审计师任期和审计师变更与审计投入、客户知识、独立性及审计质量之间的关系；本文据此控制 `LnAuditFee`、`AuditorTenure` 和 `IsChangeFirm`，以避免 `IMS` 吸收其他审计关系特征的影响。

参考文献及 DOI：

1. Marquis, C., Toffel, M. W., & Zhou, Y. (2016). Scrutiny, Norms, and Selective Disclosure: A Global Study of Greenwashing. *Organization Science, 27*(2), 483–504. [https://doi.org/10.1287/orsc.2015.1039](https://doi.org/10.1287/orsc.2015.1039)
2. Reichelt, K. J., & Wang, D. (2010). National and Office-Specific Measures of Auditor Industry Expertise and Effects on Audit Quality. *Journal of Accounting Research, 48*(3), 647–686. [https://doi.org/10.1111/j.1475-679X.2009.00363.x](https://doi.org/10.1111/j.1475-679X.2009.00363.x)
3. DeFond, M., & Zhang, J. (2014). A Review of Archival Auditing Research. *Journal of Accounting and Economics, 58*(2–3), 275–326. [https://doi.org/10.1016/j.jacceco.2014.09.002](https://doi.org/10.1016/j.jacceco.2014.09.002)

其中，`LnAuditFee` 也可能是行业专长影响漂绿的传导渠道，因此基准结果应同时关注加入审计费用前后的 `IMS` 系数；`AuditorTenure` 与 `IsChangeFirm` 可能相关，应在变量诊断中报告相关系数或 VIF，并通过分别剔除其中一个变量检验结果稳定性。

## 3 样本规则

共同样本要求同时拥有：因变量、`IMS`、`IMS\\\\\\\_D`、全部财务控制变量、全部治理控制变量、全部审计控制变量、证券代码、年份和 `ExpertiseIndustryCode`。因此所有模型使用同一批 8,129 个公司年度观测、2,669 家公司、4 个年度和 47 个行业。

这意味着减少控制变量不会增加样本数；这是“8129 共同样本”版本的核心口径。

## 4 回归规格

对 `IMS` 和 `IMS\\\\\\\_D` 分别估计：

1. 一元回归；
2. 加财务控制；
3. 加财务与治理控制；
4. 加全部控制；
5. 加全部控制、年度固定效应和行业固定效应；
6. 完整控制与双向固定效应，HC3 标准误；
7. 完整控制与双向固定效应，按 `ExpertiseIndustryCode` 聚类标准误。

模型 1—5 和模型 7 使用公司聚类或行业聚类标准误，固定效应通过 `C(时间)` 与 `C(ExpertiseIndustryCode)` 实现。当前主模型是模型 5：全部控制、年度和行业固定效应、公司聚类标准误。

## 5 运行

安装依赖：

```bash
pip install -r requirements.txt
```

生成回归结果：

```bash
python scripts/reproduce\\\\\\\_regression.py
```

单独校验：

```bash
python scripts/validate\\\\\\\_reproduction.py
```

运行测试：

```bash
python tests/test\\\\\\\_regression\\\\\\\_smoke.py
```

输出位于 `outputs/baseline/`：完整系数、样本信息、变量诊断和构造后的回归面板。校验结果位于 `outputs/validation/`。

## 6 复现边界

本包复现的是确定性的数据整理、样本筛选和 `statsmodels` 回归阶段，不重新构造 X/Y，也不重新执行 LLM。X 和 Y 的来源分别见项目中的 `X\\\\\\\_variable\\\\\\\_reproduction/README.md` 与 `Y\\\\\\\_variable\\\\\\\_reproduction/README.md`。若替换模型版本、行业口径、变量变换或样本规则，应作为新版本结果保存，不能称为当前基准回归的精确复现。

不要随意改变：

- 共同样本筛选规则；
- `Growth` 的全市场滞后计算方式；
- `IMS\\\\\\\_D` 的 10% 阈值；
- 行业固定效应字段；
- 公司聚类标准误；
- 2021—2024 年范围。

