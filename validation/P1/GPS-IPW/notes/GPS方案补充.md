# 连续 IMS 的广义倾向得分（GPS）方案补充

本文件补充 `连续IMS与换所进入专家分析计划.md` 中的 GPS 部分。GPS 仅作为连续 `IMS` 的补充识别设计，不能替代论文的连续 `IMS` 基准回归、二元 `IMS_D` 的 PSM/IPSW，亦不以得到显著系数为目标。

## 1. 文献依据与适用边界

### 1.1 核心方法文献

- **Imbens (2000)** 首先将倾向得分思想扩展至连续处理，提出在弱可忽略性和共同支持假设下，通过条件处理密度识别剂量—反应函数：Imbens, G. W. 2000. *The role of the propensity score in estimating dose-response functions*. **Biometrika** 87 (3): 706–710. DOI: [10.1093/biomet/87.3.706](https://doi.org/10.1093/biomet/87.3.706).
- **Hirano and Imbens (2004)** 给出常用的两阶段实施框架：先估计连续处理的条件密度，再以处理剂量、GPS 及其交互项拟合结果模型。Hirano, K., and G. W. Imbens. 2004. *The propensity score with continuous treatments*. In A. Gelman and X.-L. Meng (eds.), **Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives**, 73–84. Wiley. 该章节无 DOI，不应虚构 DOI。
- **Imai and van Dyk (2004)** 推广一般处理机制下的平衡评分与诊断逻辑：Imai, K., and D. A. van Dyk. 2004. *Causal inference with general treatment regimes: Generalizing the propensity score*. **Journal of the American Statistical Association** 99 (467): 854–866. DOI: [10.1198/016214504000001187](https://doi.org/10.1198/016214504000001187).
- **Bia, Mattei, and Bia (2014)** 提供连续处理 GPS 的实证实现、共同支持与 bootstrap 推断讨论：Bia, M., A. Mattei, and M. Bia. 2014. *A Stata package for the estimation of the dose-response function through adjustment for the generalized propensity score*. **The Stata Journal** 14 (3): 580–601. DOI: [10.1177/1536867X1401400302](https://doi.org/10.1177/1536867X1401400302).

### 1.2 经济管理应用文献与可借鉴之处

检索结果显示，连续处理 GPS 在经济学和管理相关的实证应用中**存在且有成熟的剂量—反应使用范式**，但在会计审计顶级期刊中并非行业专长研究的常规设计。具体而言：

- **Kluve, Schneider, Uhlendorff, and Zhao (2012)** 使用 GPS 估计培训持续时间这一连续处理的就业剂量—反应关系。该文不是会计论文，但直接展示了“连续处理 + 选择进入 + 剂量—反应曲线”的劳动经济学应用：*Evaluating continuous training programmes by using the generalized propensity score*. **Journal of the Royal Statistical Society: Series A** 175 (2): 1–31. DOI: [10.1111/j.1467-985X.2011.01000.x](https://doi.org/10.1111/j.1467-985X.2011.01000.x)。本研究可借鉴其核心思想：不把持续时间或强度任意二元化，而是在共同支持区间内估计连续剂量与结果之间的关系。
- **Behrman, Cheng, and Todd (2004)** 研究学前项目暴露时长不同的情形，使用非参数方法估计连续暴露的结果差异。其不是 Hirano--Imbens GPS 的直接应用，但说明经济学顶级领域承认“暴露长度/强度是连续处理、不能被粗略二元化”的研究问题：*Evaluating preschool programs when length of exposure to the program varies: A nonparametric approach*. **Review of Economics and Statistics** 86 (1): 108–132. DOI: [10.1162/003465304323023714](https://doi.org/10.1162/003465304323023714)。本研究借鉴的是连续剂量问题的经济解释，而不是声称该文使用 GPS。
- **Ciaian, Kancs, and Michalek** 的欧盟补贴—土地租金研究将连续补贴强度作为处理并采用 GPS 估计资本化率的剂量差异；可作为农业与公共经济应用的补充例证，但若未获得正式期刊版本和 DOI 的交叉核验，不能将其列为论文正式核心引文。

因此，论文中最稳妥的定位为：`IMS` 是事务所行业市场份额强度，属于连续处理；GPS 是经济学和统计学中处理连续剂量选择问题的成熟方法。不能写成“顶级会计审计文献普遍采用 GPS”，也不能将 Behrman et al. (2004) 误引为 GPS 应用。

### 1.3 会计审计文献定位

截至本计划修订时，未核验到将 Hirano–Imbens 连续处理 GPS 直接用于“审计师行业专长—企业漂绿”或同类审计师选择问题的顶级会计期刊论文。因此，论文不能写成“GPS 是该类审计研究的普遍做法”。

可采用的准确表述是：连续 `IMS` 不适合套用二元 PSM；GPS 是连续处理效应分析的经典计量方法。审计领域关于客户自选择和行业专长的研究可用于说明识别问题本身，例如 Minutti-Meza (2013, DOI: [10.1111/1475-679X.12017](https://doi.org/10.1111/1475-679X.12017))；PSM 的会计研究报告规范可参考 Shipman, Swanquist, and Whited (2017, DOI: [10.2308/accr-51449](https://doi.org/10.2308/accr-51449))。

## 2. 识别对象与假设

处理变量为当期连续行业市场份额：

\[
T_i=IMS_{i,t}.
\]

潜在结果为在任意市场份额剂量 \(t\) 下的漂绿：

\[
Y_i(t)=Greenwashing_{i,t}(t).
\]

GPS 识别需要以下假设：

1. **弱可忽略性**：对所有处于共同支持区间的剂量 \(t\)，\(Y_i(t)\perp T_i\mid X_{i,t-1}\)。
2. **重叠/共同支持**：在给定滞后协变量、年度和行业后，观察到的 `IMS` 剂量在条件密度中具有足够正概率。
3. **SUTVA**：一家企业的漂绿潜在结果不受其他企业 `IMS` 剂量直接影响。鉴于行业市场份额具有市场层面的机械关联，该假设在本研究中尤其强，结果应解释为条件关联，而非无保留因果效应。

协变量只使用处理发生前变量：滞后漂绿、滞后财务、治理和审计特征；年度、行业信息进入处理模型和结果模型。

## 3. 修订后的估计程序

### 3.1 第一阶段：处理密度模型

由于 `IMS` 位于 \((0,1)\)，先进行 logit 变换：

\[
Z_i=\log\left(\frac{IMS_i}{1-IMS_i}\right).
\]

估计：

\[
Z_i=\eta(X_{i,t-1},YearFE,IndustryFE)+u_i,
\qquad u_i\sim N(0,\sigma^2).
\]

再计算观察剂量的条件密度：

\[
R_i=r(T_i,X_i)=f_{T\mid X}(T_i\mid X_i).
\]

实施时应同时报告：

- 处理模型样本量、企业数、残差标准差和调整后 \(R^2\)；
- `IMS` 的 P5、P95；
- GPS 条件密度的分位数；
- 各剂量分位组内协变量平衡诊断。

### 3.2 共同支持和极端 GPS 处理

主剂量—反应曲线只在 `IMS` 的 P5—P95 范围估计。该范围在本样本中为约 0.0016 至 0.2098。

此外，以观察值 GPS 密度的 P1—P99 作为诊断性修剪范围：只要极端 GPS 观测不改变主要剂量—反应方向，即保留全样本二阶段结果、同时报告修剪稳健性；若修剪明显改变结论，则不能将 GPS 结果作为稳定证据。

不得依据哪一种修剪能显著而选择性报告。

### 3.3 第二阶段：结果模型

使用 Hirano–Imbens 二次响应面：

\[
E[Y\mid T,R]=
\alpha_0+
\alpha_1T+
\alpha_2T^2+
\alpha_3\log R+
\alpha_4(\log R)^2+
\alpha_5T\log R+
YearFE+IndustryFE.
\]

系数本身不直接等于剂量效应。主报告对象是平均剂量—反应函数：

\[
\widehat\mu(t)=\frac{1}{N}\sum_{i=1}^N
\widehat E[Y_i\mid T_i=t,\widehat r(t,X_i)].
\]

并报告预设剂量点（P5、P25、P50、P75、P95）的预测漂绿，以及 P95 相对 P5 的差值。

### 3.4 推断：企业层面 cluster bootstrap

普通二阶段预测标准误忽略第一阶段处理密度估计误差，也不能正确反映同一企业多年度观测的相关性。因此，主置信区间应以企业为抽样单位进行 cluster bootstrap：

1. 有放回抽取企业；
2. 保留每个抽中企业的全部年份；
3. 重估第一阶段处理模型、GPS 和第二阶段响应面；
4. 重新计算每个预设剂量点的 \(\widehat\mu(t)\)；
5. 使用 bootstrap percentile 95% 区间。

建议目标重复次数为 500；开发和调试阶段可先使用 50 次，但不能将 50 次区间作为论文主表。若 500 次因计算资源受限不能完成，论文仅报告模型内区间，并明确其未纳入第一阶段不确定性。

### 3.5 平衡诊断

GPS 不是只检查一个二元处理组 SMD。实施以下诊断：

1. 将 `IMS` 按五分位划分为剂量组；
2. 在每个剂量组内，按照该组 GPS 的中位数划分高/低 GPS 子组；
3. 对每个滞后协变量计算高/低 GPS 子组 SMD；
4. 输出每个剂量组内的最大绝对 SMD、通过比例和极端组诊断；
5. 对照原始协变量—IMS 相关性，判断 GPS 是否削弱系统性选择模式。

由于连续处理 GPS 平衡诊断尚无唯一统一阈值，\(|SMD|<0.10\) 作为参考标准而非机械通过条件。

## 4. 结果表述规则

若 bootstrap 和诊断均支持单调下降曲线，可以写：

> 将 `IMS` 视为连续处理变量的 GPS 分析表明，在可观测滞后客户特征、年度和行业条件相同的前提下，较高行业市场份额对应更低的预测漂绿；该结果与 H1 一致。

不得写“GPS 证明行业专长因果降低漂绿”，除非另有外生识别处理未观测客户—审计师匹配。

若极端 GPS 修剪、平衡诊断或 bootstrap 区间不支持主曲线，应将 GPS 放入附录，正文仅报告连续 `IMS` 的滞后控制回归结果。
