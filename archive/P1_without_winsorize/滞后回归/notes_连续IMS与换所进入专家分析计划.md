# 连续 IMS 与换所后进入行业专家分析计划

## 一、目的与原则

本计划以连续行业专长度量 `IMS` 和可观察的事务所变更事件为补充识别路径，回应二元变量 `IMS_D` 在 PSM/IPSW 中经客户特征调整后不再显著的问题。

本计划**不以获得显著性为目标**，而是预先定义与论文理论相符的处理变量、样本限制、协变量和推断方式，并完整报告全部预设结果。连续 `IMS` 的结果用于保留行业专长强度信息；换所进入行业专家的结果用于把分析时点对准企业重新选择事务所的年份。

论文主假设仍为：


Greenwashing_{i,t}=\alpha+\beta Specialization_{i,t}+\gamma'X_{i,t}+YearFE_t+IndustryFE_j+\varepsilon_{i,t},
\qquad \beta<0.


其中，`Greenwashing连续_基准` 为行业—年度标准化后的实质性环境沟通占比减环境绩效得分，数值越高表示漂绿程度越高；`IMS` 为事务所对行业—年度客户营业收入的市场份额；`IMS_D=1(IMS\geq10\%)` 为现有二元阈值变量。

---

## 二、研究路径 A：连续 `IMS` 作为主补充变量

### 2.1 经济动机

`IMS_D` 将行业专长强度压缩为是否超过 10% 的二元状态：例如 `IMS=0.099` 和 `IMS=0.001` 都被定义为非专家，而 `IMS=0.10` 与 `IMS=0.50` 都被定义为专家。基准回归中，连续 `IMS` 的负向关系强于 `IMS_D`，因此有必要保留市场份额强度信息。

该分析不把连续处理变量错误地套入二元 PSM。连续处理的主分析采用带固定效应的回归和预设的非线性检验；如样本重叠、模型拟合和样本量允许，再将广义倾向得分（GPS）作为补充，而非为获取特定显著性而搜索剂量函数。

### 2.2 核心模型

使用论文的共同样本，并沿用年度、行业固定效应与公司聚类标准误：


GW_{i,t}=\alpha+\beta_1 IMS_{i,t}+\gamma'X_{i,t}+\lambda_t+\delta_j+\varepsilon_{i,t}.


预设并依次估计以下规格：


| 规格  | `IMS` 形式      | 协变量          | 用途              |
| --- | ------------- | ------------ | --------------- |
| C1  | 连续线性 `IMS`    | 论文当期完整控制变量   | 与论文基准主结果直接对应    |
| C2  | 连续线性 `IMS`    | 滞后财务、治理与审计变量 | 处理发生前协变量的严格规格   |
| C3  | `IMS` 的二次项    | 与 C1 相同      | 检查边际作用是否非线性     |
| C4  | `IMS` 四分位虚拟变量 | 与 C1 相同      | 检查高专长区间是否驱动关系   |
| C5  | `IMS` 分位组趋势项  | 与 C1 相同      | 用分位组中位数检验单调剂量关系 |


C3 中报告 `IMS` 与 `IMS^2`，并报告在第 25%、50%、75% 分位处的边际效应：


\frac{\partial GW}{\partial IMS}=\beta_1+2\beta_2 IMS.


C4 使用最低四分位作为基准组；应同时报告所有分位组系数和联合显著性检验，不能只保留有利分位组。

### 2.3 高低强度组的补充可比性检验

为补充说明连续 `IMS` 的经济量级，可进行**预先定义的极端组比较**：

1. 在全样本 `IMS` 分布中定义最低四分位 `LowIMS` 和最高四分位 `HighIMS`；
2. 删除中间两个四分位；
3. 定义 `HighIMS_D=1(IMS\geq P75)`，`HighIMS_D=0(IMS\leq P25)`；
4. 使用处理发生前的企业基本特征、公司治理和滞后漂绿估计高专长概率；
5. 采用年度 × 行业内 1:1 最近邻、有放回匹配，以及 `0.2\times SD[logit(\hat p)]` 的 caliper；
6. 报告 `0.1`、`0.2` 与 `0.25` 倍标准差的全套预设敏感性结果；
7. 对匹配权重回归使用公司聚类标准误，并报告匹配后所有协变量的 SMD。

该比较估计的是“市场份额很高的事务所客户”与“市场份额很低的事务所客户”的差异，不等同于 10% 专家门槛的因果效应，也不能替代主回归中的连续 `IMS`。

### 2.4 GPS 的可选补充

若实施 GPS，处理变量为连续 `IMS`，使用处理前协变量与年度、行业信息拟合条件密度：


r(IMS,X)=f_{IMS\mid X}(IMS\mid X).


在二阶段估计中拟合：

# 
E[GW\mid IMS,r(IMS,X)]

\alpha+\theta_1 IMS+\theta_2 IMS^2+\theta_3 r+\theta_4 r^2+\theta_5 IMS\times r.


报告剂量—反应曲线及 95% 置信区间，并限制解释范围于 `IMS` 的共同支持区域。若正态处理密度拟合差、共同支持不足或高阶模型不稳定，应不将 GPS 作为主结果，仅保留 C1—C5。

### 2.4.1 GPS 实施修订与文献依据

GPS 详细方案见同目录 `GPS方案补充.md`。连续 `IMS` 不适用二元 PSM；本研究使用 Hirano--Imbens 两阶段 GPS 作为补充，不将其表述为审计研究中的通行做法。其方法依据为 Imbens (2000, DOI: [10.1093/biomet/87.3.706](https://doi.org/10.1093/biomet/87.3.706))、Hirano and Imbens (2004) 和 Imai and van Dyk (2004, DOI: [10.1198/016214504000001187](https://doi.org/10.1198/016214504000001187))。截至本计划修订时，未核验到顶级会计期刊直接将该 GPS 设计用于审计师行业专长--漂绿问题。

第一阶段对 \(logit(IMS)\) 使用滞后漂绿、完整滞后财务、治理和审计特征、年度与行业固定效应拟合条件均值；将正态残差密度乘以 \(1/[IMS(1-IMS)]\) 转换为原始 `IMS` 尺度的 GPS。第二阶段预设 `IMS`、`IMS^2`、`log(GPS)`、`log(GPS)^2` 和交互项的二次响应面。主对象为平均剂量--反应函数，而不是任何单一响应面系数。

剂量反应仅在 `IMS` 的 P5--P95 支持区间解释，并报告 P5、P25、P50、P75、P95 及 P95--P5 差值。诊断必须包括 GPS 密度、IMS 五分位内协变量平衡、高低 GPS 子组 SMD，以及 GPS P1--P99 修剪的完整敏感性结果。主推断使用企业层面 cluster bootstrap 500 次，每次重估两阶段模型；未完成该 bootstrap 时，模型内置信区间只能标为开发性或描述性结果。

### 2.5 连续处理分析的诊断与报告

必须输出：

- `IMS` 的均值、标准差、分位数、最小值、最大值和按年度/行业的分布；
- `IMS_D` 与 `IMS` 的交叉分布，检验 10% 阈值附近是否存在密集观测；
- C1—C5 的系数、公司聚类标准误、p 值、样本数、企业数与调整后 R^2；
- C3 的边际效应及置信区间；
- C4 的联合 F/Wald 检验；
- 高低组匹配的样本流、caliper、控制组重复使用次数、SMD、匹配距离及全部预设回归结果；
- 如实施 GPS，报告处理密度模型、共同支持、剂量—反应图和敏感性分析。

---



## 三、研究路径 B：换所后进入行业专家状态



### 3.1 与现有换所分析的衔接

现有 `03_审计师换所样本分析` 已构造并核验换所事件：


VerifiedSwitch_{i,t}=1


要求当期 `IsChangeFirm=1`，且事务所代码满足：


InstitutionID_{i,t}\neq InstitutionID_{i,t-1}.


当前输出显示：


| 样本口径            | 观测数    | 企业数   | 进入专家事件数     |
| --------------- | ------ | ----- | ----------- |
| 输入面板            | 18,077 | 4,949 | —           |
| 核验后主换所样本        | 1,666  | 1,539 | 379（换入行业专家） |
| `UpgradeExpert` | 237    | 237   | 237         |


本路径不应重新以未核验的 `IsChangeFirm` 作为换所依据。

### 3.2 主处理变量：`UpgradeExpert`

主处理变量定义为：


UpgradeExpert_{i,t}=1(
VerifiedSwitch_{i,t}=1,
IMSD_{i,t-1}=0,
IMSD_{i,t}=1
).


其经济含义是：企业在已核验的换所年份，从非行业专家事务所转换至行业专家事务所。

为保持处理组与对照组处于相同的事前专家状态，主对照组定义为：


SwitchNonExpert_{i,t}=1(
VerifiedSwitch_{i,t}=1,
IMSD_{i,t-1}=0,
IMSD_{i,t}=0
).


主分析样本限制为：


VerifiedSwitch_{i,t}=1\quad \text{且}\quad IMSD_{i,t-1}=0.


在该样本内，处理变量就是 `UpgradeExpert`；剔除由专家换入非专家、专家换入专家等事前状态不一致的观测。这样避免把“进入行业专家”与“原本已是专家客户的持续关系”混在一起。

### 3.3 主结果变量与模型

主结果变量使用当期漂绿相对于换所前一年的变化：


\Delta GW_{i,t}=GW_{i,t}-GW_{i,t-1}.


主模型为：

# 
\Delta GW_{i,t}

\alpha+eta UpgradeExpert_{i,t}+
\gamma'X_{i,t-1}+
\lambda_t+delta_j+varepsilon_{i,t}.


其中，`X_{i,t-1}` 使用换所前可观测特征。预设两类协变量：


| 规格    | 协变量                                                                                                       |
| ----- | --------------------------------------------------------------------------------------------------------- |
| S1 精简 | `L_Greenwashing连续_基准`、`L_Size`、`L_Lev`、`L_ROA`、`L_SOE`、`L_Big4`、`L_AuditorTenure`                         |
| S2 扩展 | S1 加 `L_CFO`、`L_Growth`、`L_FirmAgeLog`、`L_Dual`、`L_BoardSize`、`L_IndependentDirectorRatio`、`L_LnAuditFee` |


年度和行业固定效应应保留；标准误按企业聚类。若样本中企业仅出现一次，聚类标准误与 HC3 稳健标准误一并报告。由于换所样本期短、处理事件有限，S1 作为主规格，S2 作为扩展规格，避免事件数相对于参数数目过低。

辅助结果使用当期水平：

# 
GW_{i,t}

\alpha+beta UpgradeExpert_{i,t}+rho GW_{i,t-1}+gamma'X_{i,t-1}+lambda_t+delta_j+varepsilon_{i,t}.


变化模型是主模型，因为它直接考察换入行业专家后相对于换所前的漂绿变化；水平模型只作为稳健性检验。

### 3.4 换所样本匹配与加权

由于 `UpgradeExpert` 不是随机发生，需在“换所且前一期非专家”的限定样本内进行处理前可比性调整。

#### 主方法：overlap weighting

1. 使用 Logit 估计：


Pr(UpgradeExpert_{i,t}=1\mid X_{i,t-1},YearFE,IndustryFE).


1. 使用重叠权重：


w_i=
\begin{cases}
1-\hat e_i,& UpgradeExpert=1,
\hat e_i,& UpgradeExpert=0.
\end{cases}


1. 采用加权变化模型，保留年度、行业固定效应与公司聚类标准误；
2. 检查加权后协变量的绝对 SMD 是否均小于 0.10；
3. 报告权重分布、ESS、共同支撑和任何极端权重处理。

选择 overlap weighting 的原因是当前处理组事件约 237 个，传统 1:1 PSM 会进一步丢弃换所观测，而重叠权重可将估计对象聚焦于真实存在专家与非专家选择可能性的换所企业。

#### 稳健性方法：年度 × 行业内 1:1 有放回 PSM

- 仅在同年度、同 `ExpertiseIndustryCode` 内匹配；
- Logit 倾向得分；
- caliper 为 `0.2×SD(logit propensity score)`；
- 有放回最近邻；
- 用控制观测的重复使用次数构造匹配权重；
- 报告 `0.1`、`0.2`、`0.25` 倍标准差结果与全套 SMD；
- 如年度×行业内处理组过少、无法估计或大面积失配，则明确报告可行性问题，不放宽为跨行业匹配来寻求显著性。



### 3.5 进入行业专家的连续强度扩展

在主 `UpgradeExpert` 样本中，进一步构造：


\Delta IMS_{i,t}=IMS_{i,t}-IMS_{i,t-1}.


并在所有核验换所样本、或前期非专家换所样本中估计：

# 
\Delta GW_{i,t}

\alpha+beta\Delta IMS_{i,t}+gamma'X_{i,t-1}+lambda_t+delta_j+varepsilon_{i,t}.


这不是主因果估计，而是检验“换所后行业专长提升幅度”是否与漂绿变化呈单调负向关系。需报告 `\Delta IMS` 分布、异常值处理规则、线性与二次项结果；不能仅报告最显著的函数形式。

### 3.6 事前趋势与安慰剂检验

若同一企业存在连续 `t-2`、`t-1`、`t` 观测，比较处理组和对照组在换所前的漂绿变化：


PreDeltaGW_{i,t}=GW_{i,t-1}-GW_{i,t-2}.


并估计或报告：

# 
PreDeltaGW_{i,t}

\alpha+theta UpgradeExpert_{i,t}+gamma'X_{i,t-1}+lambda_t+delta_j+varepsilon_{i,t}.


`\theta` 不显著仅表示未观察到系统性事前差异；由于样本期仅 2021—2024 年，不能将其视为严格平行趋势验证。

安慰剂结果可使用 `GW_{i,t-1}` 或 `PreDeltaGW`；不能使用处理发生后的变量作为倾向得分协变量。

---



## 四、实现结构与预期输出

建议在本目录下新增独立脚本，避免覆盖既有结果：

```text
validation/P1_14778/test/
├── 连续IMS与换所进入专家分析计划.md
├── run_continuous_ims_and_upgrade.py
└── outputs/
    ├── tables/
    ├── diagnostics/
    ├── figures/
    ├── matched_data/
    └── weighted_data/
```



### 4.1 数据字段

读取：

```text
validation/P1_14778/data/input/combined_regression_panel_14778.csv
```

核心字段：


| 角色       | 字段                                  |
| -------- | ----------------------------------- |
| 企业、年份、行业 | `证券代码`、`时间`、`ExpertiseIndustryCode` |
| 结果       | `Greenwashing连续_基准`                 |
| 连续专长     | `IMS`                               |
| 阈值专长     | `IMS_D`                             |
| 换所核验     | `IsChangeFirm`、`InstitutionID`      |
| 控制变量     | 论文中现有财务、治理及审计控制变量                   |


按企业—年份排序，仅在严格连续年度条件下生成滞后变量；不允许跨缺失年份机械取滞后值。

### 4.2 建议输出表


| 文件                               | 内容                                  |
| -------------------------------- | ----------------------------------- |
| `continuous_ims_regressions.csv` | C1—C5 连续专长回归、边际效应与联合检验              |
| `ims_distribution.csv`           | `IMS` 总体、年度和行业分布及阈值附近密度             |
| `high_low_ims_matching_flow.csv` | 高低四分位匹配样本流和参数                       |
| `high_low_ims_balance.csv`       | 匹配前后 SMD                            |
| `high_low_ims_outcomes.csv`      | 高低强度组匹配后回归                          |
| `upgrade_sample_flow.csv`        | 核验换所、前期非专家、进入专家与对照组的样本流             |
| `upgrade_pre_switch_balance.csv` | 升级组与非专家换所组的事前均值差、SMD                |
| `upgrade_weight_diagnostics.csv` | overlap/IPTW 权重分布及 ESS              |
| `upgrade_overlap_balance.csv`    | overlap weighting 后协变量平衡            |
| `upgrade_delta_gw_models.csv`    | `UpgradeExpert` 对 `DeltaGW` 的主/扩展模型 |
| `upgrade_level_gw_models.csv`    | `UpgradeExpert` 对当期 `GW` 的辅助模型      |
| `upgrade_psm_models.csv`         | 预设 0.1/0.2/0.25 caliper 的匹配结果       |
| `upgrade_pretrend_models.csv`    | 换所前漂绿变化的安慰剂模型                       |
| `upgrade_delta_ims_models.csv`   | `DeltaIMS` 与 `DeltaGW` 的连续强度扩展      |




### 4.3 图形

- `IMS` 的直方图、核密度图和 10% 阈值标记；
- `IMS` 四分位结果与 95% 置信区间图；
- GPS 剂量—反应图（仅在实施且诊断合格时）；
- `UpgradeExpert` 倾向得分的处理组/对照组重叠图；
- 换所升级样本的加权前后 Love plot；
- `DeltaGW` 的组别均值与置信区间图。

---



## 五、解释与论文写作规则



### 5.1 若连续 `IMS` 显著、而 `IMS_D` 选择校正后不显著

可作如下解释：

> 行业专长与漂绿的关系更多体现为连续市场份额强度差异，而非仅由是否跨越 10% 门槛决定。二元阈值变量压缩了市场份额的异质性，因而在严格的客户可比性调整后估计精度较低。

此结论不能写成“PSM证明了连续 IMS 的因果作用”，除非 GPS 的共同支持、平衡和剂量—反应诊断均支持该表述。

### 5.2 若换所进入专家的 `DeltaGW` 结果显著为负

可作如下解释：

> 在企业实际更换事务所且此前并非行业专家客户的样本中，换入行业专家与随后漂绿下降相关；结果聚焦于重新选择审计师的时点，因此比全样本横截面关联更贴近客户—审计师匹配机制。

同时需保留局限：换所原因可能与同期治理、经营冲击或披露调整共同发生；短面板下的事前趋势检验效力有限。

### 5.3 若换所结果仍不显著

应报告负向系数、置信区间和样本事件数，不能通过改换所定义、删除行业或更改 caliper 来事后筛选。可以得出的结论是：现有短面板和有限升级事件无法在可比换所企业中精确识别独立效应。

---



## 六、文献依据

- Rosenbaum, P. R., and D. B. Rubin. 1983. The central role of the propensity score in observational studies for causal effects. *Biometrika* 70 (1): 41–55. DOI: [10.1093/biomet/70.1.41](https://doi.org/10.1093/biomet/70.1.41).
- Hirano, K., G. W. Imbens, and G. Ridder. 2003. Efficient estimation of average treatment effects using the estimated propensity score. *Econometrica* 71 (4): 1161–1189. DOI: [10.1111/1468-0262.00442](https://doi.org/10.1111/1468-0262.00442).
- Imbens, G. W. 2000. The role of the propensity score in estimating dose-response functions. *Biometrika* 87 (3): 706–710. DOI: [10.1093/biomet/87.3.706](https://doi.org/10.1093/biomet/87.3.706).
- Austin, P. C. 2011. Optimal caliper widths for propensity-score matching when estimating differences in means and differences in proportions in observational studies. *Pharmaceutical Statistics* 10 (2): 150–161. DOI: [10.1002/pst.433](https://doi.org/10.1002/pst.433).
- Li, F., K. L. Morgan, and A. M. Zaslavsky. 2018. Balancing covariates via propensity score weighting. *Journal of the American Statistical Association* 113 (521): 390–400. DOI: [10.1080/01621459.2016.1260466](https://doi.org/10.1080/01621459.2016.1260466).
- Minutti-Meza, M. 2013. Does auditor industry specialization improve audit quality? *Journal of Accounting Research* 51 (4): 779–817. DOI: [10.1111/1475-679X.12017](https://doi.org/10.1111/1475-679X.12017).
- Shipman, J. E., Q. T. Swanquist, and R. L. Whited. 2017. Propensity score matching in accounting research. *The Accounting Review* 92 (1): 213–244. DOI: [10.2308/accr-51449](https://doi.org/10.2308/accr-51449).
- Lawrence, A., M. Minutti-Meza, and P. Zhang. 2011. Can Big 4 versus non-Big 4 differences in audit-quality proxies be attributed to client characteristics? *The Accounting Review* 86 (1): 259–286. DOI: [10.2308/accr.00000009](https://doi.org/10.2308/accr.00000009).

---



## 七、执行前检查清单

- [ ] 核验 `IMS` 为当前年份行业—年度事务所客户收入份额，且取值区间和异常值与论文一致。
- [ ] 核验 `IMS_D=1(IMS\geq0.10)`。
- [ ] 核验 `InstitutionID` 的当期与滞后代码均非缺失，且换所标志与代码变化一致。
- [ ] 核验 `UpgradeExpert` 的对照组限于 `L_IMS_D=0` 且换入非专家。
- [ ] 核验 `DeltaGW` 仅在严格连续年份计算。
- [ ] 所有匹配或加权变量均为处理前变量。
- [ ] 输出所有预设规格、所有样本损失和所有平衡诊断，不按 p 值筛选结果。
- [ ] 结果回归使用企业聚类标准误；若换所企业仅出现一次，同时报告 HC3。