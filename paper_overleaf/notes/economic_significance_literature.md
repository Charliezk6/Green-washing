# 经济显著性备忘录：写法、换算、文献句、导师回复

用途：回应导师“Economic significance is modest，应与文献量级对照”。**不改正文、不加新表。**

数据来源：`tables/descriptive_rows.tex`；`tables/baseline_regression_table_ims.tex` 列 (4)；`tables/baseline_regression_table_ims_d.tex` 列 (4)。

---

## 1. 顶刊写法总结

抽样：近年审计实证（RAST 2025）、专长经典文献（Krishnan 2003 *AH*；Balsam et al. 2003 *AJPT*；Reichelt and Wang 2010 *JAR*）、漂绿/脱钩顶刊（Marquis, Toffel, and Zhou 2016 *Organization Science*）。

### 1.1 连续变量：用 1 个标准差的 X，不是任意 +0.10

近年审计实证几乎固定写成：

> A one-standard-deviation increase in X is associated with a [ΔY] change in Y, which is a [Z] percent [decrease] relative to the unconditional mean (or probability) of Y.

算法常放脚注：\beta \times \mathrm{SD}(X)/\mathrm{mean}(Y)。

例（Khavis, Sheneman, and Szerwo 2025, *RAST*）：

> A one standard deviation increase in FemaleStaff is associated with a 0.8 percentage point decline in misstatements, which translates into an economically significant 8.9 percent reduction in the misstatement rate.
>
> The economic significance … is calculated as … (−0.145 × 0.057 / 0.093).

另一篇 RAST（审计办公室专长知识共享，2025）同样用 1 SD → 百分点，再换成占无条件概率/均值的百分比（他们得到占均值约 8%–26%，并称 economically significant）。

**对本文：** 主翻译应是 IMS +1 SD（0.0817），不是只报 +0.10。+0.10 可作为补充（市场份额提高 10 个百分点），但该增量已约 1.22 个 IMS 标准差，不是“小幅提高”。

### 1.2 虚拟变量：专家 vs 非专家的原生单位

Krishnan (2003, *Accounting Horizons*) 把量级写进摘要：

> Clients of nonspecialist auditors report absolute discretionary accruals that are, on average, 1.2 percent of total assets higher than the discretionary accruals reported by clients of specialist auditors.

该 1.2% 对应回归表中专家虚拟变量系数 −0.012（因变量是 |DA| / 资产）。**不**再除以 SD(Y)，除非 Y 本身难以解释。

**对本文：** IMS_D 应写成“达到 20% 门槛的客户，Greenwashing 平均低 0.143”，必要时再补“约 0.11 个因变量标准差”。这是和专长文献最可比的一句。

### 1.3 分母怎么选


| 情境                  | 顶刊常用分母              | 本文                                                                                         |
| ------------------- | ------------------- | ------------------------------------------------------------------------------------------ |
| Y 有清楚均值（应计、重述概率、费用） | 占 mean(Y) 或无条件概率的 % | **不适用**：Greenwashing 均值 −0.0021，除以均值会爆炸                                                    |
| Y 是标准化/脱钩分数         | 占 SD(Y)             | **适用**。Marquis et al. (2016)：\beta / \mathrm{SD}(Y)，“nearly one-half a standard deviation” |
| 判断大不大               | 同回归其他系数（Big4、规模）    | **适用**                                                                                     |
| 文献对照                | 正文通常一句，几乎不做对照表      | 对照放本备忘录                                                                                    |


Marquis, Toffel, and Zhou (2016, *Organization Science*) 最接近本文 Y 的构造（选择性披露幅度，也是沟通–绩效类缺口）：

> … a one-standard-deviation increase in environmental damage is associated with a 0.10 decline in selective disclosure magnitude, the equivalent of nearly one-half a standard deviation (calculated as β ÷ SD = −0.093 ÷ 0.23 = −0.40).



### 1.4 经典 JAR 专长文常常只报符号

Reichelt and Wang (2010, *JAR*)、Balsam et al. (2003, *AJPT*)、Dunn and Mayhew (2004, *RAST*) 的摘要/结果段以符号和相对大小为主，**没有**后来 RAST 那种完整 “1 SD / % of mean” 段落。导师要的是近年规范（量级必须写），不是 2010 年前可以省略。

### 1.5 何时说 economically significant

这句话几乎只出现在：

- 占 Y 均值约 **8%–20%+**（近年 RAST 审计质量文）；或
- 约 **0.4 SD of Y**（Marquis et al. 2016）。

本文连续 IMS 约 **0.03 SD of Y**，不应写成 economically significant。诚实表述：statistically significant but modest in magnitude。若谈“有意义”，应落在机制（CAS 1521 对年报其他信息的有限溢出），而不是把数字做大。

### 1.6 文献量级对照放哪里

顶刊几乎不在主文加“本文 vs 五篇文献”对照表。导师要的对照，本备忘录即可；若日后改正文，最多一句。

---



## 2. 本文换算



### 2.1 口径（先澄清导师用的两个数）


| 数字      | 实际是什么                        | 不是什么                     |
| ------- | ---------------------------- | ------------------------ |
| −0.4829 | 列 (4) IMS 系数                 | —                        |
| 0.2362  | 该系数的 **标准误**                 | **不是** Greenwashing 的标准差 |
| 1.3366  | Greenwashing 的样本标准差          | 正文已写 1.337               |
| 0.0483  | 0.10 × 0.4829，**因变量原单位** 的变化 | **不是** 0.048 个标准差        |


Greenwashing 均值 −0.0021（行业–年标准化后接近 0），故 **不用 % of mean(Y)**。

IMS：均值 0.0698，SD 0.0817，P25 0.0152，P75 0.0983。+0.10 是 0.10 / 0.0817 ≈ **1.22 个 IMS 标准差**。

公式：\text{standardized effect} = \beta \times \Delta X / \mathrm{SD}(Y)。\mathrm{SD}(Y)=1.3366。IQR(Y)=0.8979−(−0.8204)=1.7183。

### 2.2 换算表（列 (4)）


| 增量                           | \Delta X | \beta\times\Delta X（Y 单位） | / SD(Y)   | / IQR(Y) |
| ---------------------------- | -------- | ------------------------- | --------- | -------- |
| IMS +1 SD（主口径）               | 0.0817   | 0.0395                    | **0.030** | 0.023    |
| IMS +0.10（正文现用；>1 SD of IMS） | 0.10     | 0.0483                    | 0.036     | 0.028    |
| IMS P25→P75                  | 0.0831   | 0.0401                    | 0.030     | 0.023    |
| IMS_D（≥20% vs 其他）            | 1        | 0.1434                    | **0.107** | 0.083    |
| 同表 Big4                      | 1        | 0.3613                    | **0.270** | 0.210    |
| 同表 Size +1 SD（参考）            | 1.2148   | 0.1655                    | 0.124     | 0.096    |




### 2.3 读法

- 连续 IMS：1 个标准差的市场份额变动，对应 Greenwashing 下降约 0.04，即因变量离散度的 **3%**。相对四分位距也只有约 2%。量级小。
- 正文现用的 +0.10 看起来像“十分之一”，其实已超过 1 个 IMS 标准差，对应仍只有 0.036 SD of Y。
- 与专长文献更可比的是 **IMS_D：约 0.11 SD of Y**（专家客户低 0.143）。
- 同规格 Big4 约 0.27 SD of Y，Size 一个标准差约 0.12 SD of Y。专长小于品牌、也小于规模，符合“增量治理”而不是一阶治理工具。

GPS–IPW 列 (3) 系数 −0.2878：1 SD IMS → 约 0.024 单位 / **0.018 SD of Y**，可观测选择校正后更小，但仍为负。不作为回复主数字。

---



## 3. 文献句

原则：只抄原文经济意义句或摘要量级；缺则标明“原文只报符号”，**不事后编标准化 β**。Y 量纲不同，不能拿 −0.4829 直接比 −0.012。

### 3.1 Krishnan (2003), *Accounting Horizons* — 有明确量级，优先对照

题目：Does Big 6 Auditor Industry Expertise Constrain Earnings Management?

X：专家虚拟变量 / 连续市场份额（IMS）；Y：|DA|（占期初资产）。

**摘要（可直接引用）：**

> Clients of nonspecialist auditors report absolute discretionary accruals that are, on average, 1.2 percent of total assets higher than the discretionary accruals reported by clients of specialist auditors.

**回归表（Table 6，组合份额、虚拟变量）：** SPECLST = **−0.012**（即摘要的 1.2% of assets）。连续组合份额系数 −0.042。Table 7 按行业市场份额：虚拟变量 −0.002，连续 IMS −0.005。

**描述统计（Table 4 Panel A，组合份额）：** |DA| 均值专家 0.054、非专家 0.080。单变量差 0.026；控制后缩小到 0.012。

**写法启示：** 用 dummy 的原生单位（占资产百分比）写经济意义；连续 IMS 的系数更小，原文没有把连续系数再做成“1 SD”段落。

**与本文：** IMS_D = −0.143 ≈ 0.11 SD of Y，是“专家 vs 非专家”对照，和 Krishnan 的 dummy 句式同类。不能说“比 1.2% of assets 更大/更小”——Y 不是应计。能说的是：专长文献里，即使对财务报表质量，专家–非专家反差也是小到中等的一截，而不是翻转整个分布。

### 3.2 Balsam, Krishnan, and Yang (2003), *Auditing: A Journal of Practice & Theory*

题目：Auditor Industry Specialization and Earnings Quality。

**摘要（原文只报符号，未写 1 SD 或占均值 %）：**

> After controlling for variables established in prior work to be related to DAC and the ERC, we find clients of industry specialist auditors have lower DAC and higher ERC than clients of nonspecialist auditors.

未在公开摘要中给出可独立核验的经济意义换算。**不编造表内系数。**

### 3.3 Dunn and Mayhew (2004), *Review of Accounting Studies*

题目：Audit Firm Industry Specialization and Client Disclosure Quality。

**摘要（原文只报符号/分组，未写量级）：**

> Consistent with our hypotheses, we document a positive association between industry-specialist audit firms and analysts’ rankings of disclosure quality in unregulated industries, but no relation in regulated industries.

这是专长→披露质量的最近邻文献，但原文没有把 AIMR 排名系数译成 SD 或占均值。**不能做数字对照，只能做机制对照：** 专长与披露质量的关系本身就被认为有制度边界（受管制行业无关系）。本文是年报其他信息上的沟通–绩效缺口，量级偏小与“有边界的溢出”一致。

### 3.4 Reichelt and Wang (2010), *Journal of Accounting Research*

题目：National and Office-Specific Measures of Auditor Industry Expertise and Effects on Audit Quality。

**摘要（只报符号和相对大小，无 1 SD / % of mean）：**

> We find that auditors who are both national and city-specific industry specialists have clients with the lowest abnormal accruals, suggesting that joint national and city-specific industry specialists have the highest audit quality. … its clients are less likely to meet or beat analysts’ earnings forecasts by one penny per share and more likely to be issued a going-concern audit opinion.

JAR 经典专长文可以只写方向。近年实证才强制写量级。回复导师时不要假装 Reichelt 给了可直接比的 0.xx SD。

### 3.5 蔡春、鲜文铎 (2007), 《会计研究》— 中国 IMS 量纲参考

摘要：在中国上市公司审计市场，事务所行业专长与审计质量**负相关**；归因于独立性不高、行业经济依赖、专长发展程度低。**原文摘要未给出系数或 1 SD 换算。** 只说明：同一 IMS 构造在中国市场上，前人连符号都与美文相反，更不会默认“专长效应必须很大”。只比量纲、不比符号。

### 3.6 对照一句（给回复用，不是表）

- 连续 IMS +1 SD ≈ **0.03 SD of Y**：按 Marquis 的标尺（他们把 0.40 SD 写成 nearly one-half a SD），这是小效应；按近年 RAST 的 % of mean 标尺，本文 Y 均值≈0，无法用那套话说“economically significant”。
- IMS_D ≈ **0.11 SD of Y**：与 Krishnan dummy 的“小到中等、但写进摘要”同类——他们写的是 1.2% of assets，不是几个标准差的大移动。
- 同表 Big4 ≈ **0.27 SD of Y**：专长小于品牌，符合预期。

---



## 4. 导师回复草稿

可直接粘贴。不指责计算，只陈述口径；不写“正文已改”。

---

感谢老师指出经济显著性需要单独讨论，并要求与文献量级对照。我按描述统计重新换算如下。

第一，口径。列 (4) 的 0.2362 是 IMS 系数的标准误，不是因变量标准差。Greenwashing 的标准差是 1.337（描述统计为 1.3366）。正文里 “IMS 增加 0.10 → Greenwashing 下降 0.0483” 是因变量**原单位**的变化，不是 0.048 个标准差。0.10 的市场份额变动约等于 1.22 个 IMS 标准差，本身已是较大的份额变化。

第二，按顶刊近年写法，连续变量用 1 个标准差的 X 来翻译。IMS 增加一个标准差（0.082）对应 Greenwashing 下降约 0.039，相当于因变量标准差的约 **0.03**，或四分位距的约 0.02。量级确实 modest。同规格下，Big4 约为 0.27 个因变量标准差，Size 一个标准差约为 0.12；专长小于品牌和规模。

第三，与专长文献更可比的是专家虚拟变量。IMS_D（行业份额 ≥20%）的系数是 −0.143，约 **0.11** 个因变量标准差。Krishnan (2003, *Accounting Horizons*) 的摘要写法是：非专家客户的 |DA| 平均比专家客户高 **1.2% of total assets**（回归虚拟变量系数 −0.012）。Balsam et al. (2003)、Dunn and Mayhew (2004)、Reichelt and Wang (2010) 的摘要以符号为主，没有把系数译成 1 SD 或占均值百分比。因此，可核对的数字对照主要是 Krishnan 的专家–非专家反差：专长文献即使针对财务报表质量，报告的也是小到中等的一截，而不是翻转整个分布。

第四，是否 practically meaningful。我认为应承认经济量级 modest，不宜写成 economically significant（近年 RAST 用这个词时，常见是占结果均值的 8%–20%，或 Marquis et al. 2016 那种约 0.4 个标准差）。更准确的解释是：CAS 1521 / ISA 720 只要求阅读年报其他信息，不是可持续鉴证。在这一有限通道下，统计上可检出、但幅度不大的负向关联，更符合“财务审计对沟通–绩效缺口的有限溢出”，而不是漂绿的一阶治理工具。Dunn and Mayhew (2004) 也表明专长与披露质量的关系有制度边界（受管制行业无关系），与“溢出有限”一致。

若老师认为需要写进正文，我建议基准结果段按“1 SD IMS → IMS_D → 同表 Big4 → 一句 Krishnan”这个顺序补几句，并明确写 modest；主文不加文献对照表。

---



## 附：若日后改正文可用的英文骨架（本次不写入 Paper.tex）

> In column (4), the coefficient on IMS is -0.4829. A one-standard-deviation increase in IMS (0.082) is associated with a 0.039 decline in Greenwashing, or about 0.03 of the outcome’s standard deviation (1.337). A 10-percentage-point increase in industry market share—already larger than one standard deviation of IMS—corresponds to a 0.048 decline (0.036 of a standard deviation). Relative to other clients, firms audited by auditors meeting the 20 percent threshold (IMSD) have Greenwashing scores that are 0.143 lower, or about 0.11 of a standard deviation. By comparison, the same specification associates Big4 with a 0.36 decline (about 0.27 of a standard deviation). These magnitudes are modest. They are closer to the specialist-versus-nonspecialist contrasts reported for earnings quality (e.g., Krishnan 2003) than to a first-order sustainability-assurance effect, which is consistent with the limited reach of the other-information requirement in CAS 1521.

