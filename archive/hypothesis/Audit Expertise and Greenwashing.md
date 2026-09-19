# 会计师事务所行业专长对企业漂绿行为的影响机制与假设开发

> **研究情境：** 中国 A 股资本市场  
> **核心关系：** 会计师事务所行业专长 → 企业漂绿行为  
> **理论主线：** 识别机制＋约束机制；声誉资本仅作为辅助解释  
> **样本建议：** 2021—2024 年中国 A 股上市公司

---

## 目录

- [一、文献检索策略](#一文献检索策略)
- [二、文献证据矩阵](#二文献证据矩阵)
- [三、理论框架选择与比较](#三理论框架选择与比较)
- [四、理论逻辑链](#四理论逻辑链)
- [五、假设开发（长版）](#五假设开发长版)
- [六、假设开发（精简版）](#六假设开发精简版)
- [七、正式假设与竞争性假设](#七正式假设与竞争性假设)
- [八、竞争性解释与边界条件](#八竞争性解释与边界条件)
- [九、内生性处理与实证建议](#九内生性处理与实证建议)
- [十、参考文献](#十参考文献)
- [十一、补充资料链接](#十一补充资料链接)

---

## 一、文献检索策略

### 1.1 研究目标

本研究旨在构建会计师事务所行业专长（*Audit Firm Industry Specialization*）影响上市公司漂绿行为（*Corporate Greenwashing*）的理论逻辑链。鉴于传统审计研究中二者直接交叉的文献较少，本文整合财务审计、非财务信息披露、ESG 鉴证、印象管理及大语言模型（LLM）文本分析等相邻领域的研究成果。

### 1.2 数据库与检索式

检索范围包括 Web of Science（Core Collection）、ScienceDirect、Wiley Online Library、Emerald Insight、JSTOR、SSRN、中国知网（CNKI）及 CSSCI 数据库。

**英文检索主题：**

1. **审计师行业专长与审计质量**
  > ("auditor industry specialization" OR "audit firm industry expertise" OR "auditor industry knowledge" OR "industry specialist auditor") AND ("audit quality" OR "earnings management" OR "discretionary accruals" OR "financial reporting quality" OR "restatement" OR "risk assessment" OR "audit effort")
2. **漂绿、环境披露与脱钩机制**
  > ("greenwashing" OR "corporate greenwashing" OR "symbolic disclosure" OR "substantive disclosure" OR "environmental disclosure performance gap" OR "ESG decoupling" OR "impression management") AND ("annual report" OR "textual analysis" OR "LLM" OR "NLP")
3. **审计监督与非财务信息**
  > ("auditor expertise" OR "audit monitoring" OR "sustainability assurance" OR "ESG reporting" OR "nonfinancial disclosure") AND ("credibility" OR "greenwashing detection" OR "spillover effect" OR "other information")

**中文关键词：** 会计师事务所行业专长、审计师行业专长、审计质量、企业漂绿、环境信息披露、象征性披露、实质性披露、ESG 评级、年报文本分析。

### 1.3 文献筛选标准

#### 纳入标准

- 发表在权威会计、审计、金融或管理学期刊上的同行评审文献；
- 对行业专长度量具有规范设定的经典研究，包括市场份额法、客户组合份额法以及城市与全国层面的专长；
- 研究环境披露与实际环境绩效脱钩、漂绿定义或 NLP/LLM 漂绿测度的文献；
- 研究财务报表审计师对年报“其他信息”及非财务数据产生治理外溢效应的文献。

#### 排除标准

- 缺乏严谨理论推导或实证数据的商业报道、新闻评论和非学术简报；
- 混淆“财务报表审计”与“独立第三方 ESG 鉴证”，且未清晰界定二者边界的研究。

### 1.4 文献池结构

经 DOI 注册信息与题名、作者、期刊、年份及卷期页码逐项交叉核验后，本文保留 **23 篇可验证的核心文献**。对于 DOI 无法解析、DOI 指向其他论文或书目信息无法相互印证的条目，不再作为本文证据。该处理优先保证文献真实性，而不追求文献数量。


| 研究主题            | 代表性期刊                                                             | 核心内容                      | 数量  |
| --------------- | ----------------------------------------------------------------- | ------------------------- | --- |
| A. 行业专长与审计质量    | *TAR*, *AJPT*, *JAR*, *MAJ*, *IJA*                                | 行业经验、审计质量、错报识别、内控披露与治理互动  | 10  |
| B. 漂绿与制度脱钩      | *Organization Science*, *CMR*, *JEMS*, *CSR&EM*                   | 合法性、印象管理、审计威胁及披露型漂绿       | 4   |
| C. ESG 披露与鉴证治理  | *AJPT*, *Accounting & Finance*, *BEER*, *SAMPJ*, *Sustainability* | CSR/可持续鉴证、披露可信度、重述与资本市场反应 | 6   |
| D. 披露、文本分析与资本市场 | *JAR*, *FMII*                                                     | 信息不对称、资本成本及年报 ESG 文本测度    | 2   |
| E. LLM 漂绿测度     | *Computational Economics*                                         | DeepGreen、语义识别与环境处罚预测     | 1   |


### 1.5 研究动机与理论贡献

如果本研究的逻辑链条成立，那么它至少说明三件事。

第一，**审计师行业专长并不只是“审计财报更准”这么简单**，它还可能通过对年报“其他信息”、环境叙事和披露一致性的识别与约束，外溢地影响企业的非财务披露行为。换言之，审计师的专业知识不是只服务于利润表和资产负债表，而是可以延伸到更广义的信息治理。

第二，**漂绿并不仅是企业公关问题，而是一个可被审计监督改变的治理问题**。如果专家事务所确实能够压缩披露—绩效偏离，那么漂绿就不再只是企业内部动机、组织合法性或市场偏好的结果，而是一个与外部鉴证约束强相关的现象。这意味着，企业绿色声明的真实性并非完全取决于企业自身的道德选择，也受外部审计质量和审计专长影响。

第三，**研究能够帮助理解“高声誉中介”究竟是在约束漂绿，还是在为漂绿提供合法性背书**。这正是本研究的关键学术价值：它不是简单地假定行业专长一定有益，而是把“识别—约束”与“背书—包装”两种方向放在同一框架下比较，并通过实证检验判断净效应。换言之，如果结果为负向关系，则说明高质量审计在非财务信息治理中具有真实约束作用；如果结果弱化或不显著，则说明行业专长在漂绿场景下的治理边界是有限的。

从更一般的意义上说，本研究试图回答的是：**在 ESG 治理成为资本市场重要议题的背景下，传统财务审计的专业能力到底能在多大程度上延伸到非财务信息领域。** 这一问题不仅关系到审计师行业专长的经济后果，也关系到未来如何理解“财务审计—ESG 鉴证—文本披露治理”之间的边界。

---

## 二、文献证据矩阵


| 文献                                       | 样本或背景                    | 核心变量                 | 主要发现                          | 支撑机制                 | 属性  |
| ---------------------------------------- | ------------------------ | -------------------- | ----------------------------- | -------------------- | --- |
| Balsam et al. (2003)                     | 1991—1999 年美国 Big 6/5 客户 | 行业专长、DAC、ERC         | 专家审计师客户的可操纵应计更低、ERC 更高        | 行业知识提升信息鉴别与错报约束能力    | 直接  |
| Bills et al. (2015)                      | 美国上市公司及审计费用              | 行业齐次性、专长、成本效率        | 同质行业中，跨客户学习提高成本效率与风险评估精度      | 基准比较与跨客户验证           | 间接  |
| Owhoso et al. (2002)                     | 行业专长审计团队的实验研究            | 行业经验、错误识别            | 行业专业化会影响审计人员在连续复核中的错误识别表现     | 支持行业知识与异常识别渠道        | 直接  |
| Sun & Liu (2013)                         | 美国上市公司                   | 行业专长、董事会、盈余管理        | 专家审计师与内部治理互动并限制管理层裁量权         | 外部监督与内部沟通的治理外溢       | 间接  |
| Reichelt & Wang (2010)                   | 美国上市公司                   | 全国及事务所办公室层面行业专长、审计质量 | 全国与办公室层面的行业专长对审计质量具有不同且互补的解释力 | 支持行业专长的多层次测度         | 直接  |
| Rose-Green et al. (2011)                 | 美国上市公司                   | 行业专长、内控缺陷披露          | 行业专长与企业披露内部控制缺陷之间存在显著关联       | 支持专家审计师影响披露质量        | 直接  |
| Schmidt & Wilkins (2012)                 | 财务报表重述企业                 | 审计师质量、审计委员会专长、重述及时性  | 审计师质量和审计委员会专长影响重述披露的及时性       | 支持高质量审计促进异常信息及时披露    | 间接  |
| Ngelo et al. (2022)                      | 2010—2020 年印尼上市公司        | 前审计师 CEO、CSR、GRI     | 审计背景高管增加符合 GRI 的实质性披露         | 审计思维约束选择性非财务披露       | 间接  |
| Cohen & Simnett (2014)                   | CSR 鉴证研究综述               | CSR 报告、鉴证服务、审计研究     | 界定 CSR 报告鉴证的需求、供给和研究议题        | 支持审计专业能力向非财务鉴证延伸     | 间接  |
| Martínez-Ferrero & García-Sánchez (2017) | 国际企业样本                   | 可持续鉴证、资本成本、CSR 可信度   | 可持续鉴证通过提高 CSR 信息可信度影响资本成本     | 支持鉴证降低非财务信息不对称       | 直接  |
| Bentley-Goode et al. (2025)              | 可持续报告及其重述                | 鉴证提供者、报告质量、重述        | 鉴证提供者选择与可持续报告质量和重述相关          | 支持鉴证质量约束非财务错报        | 直接  |
| Lyon & Maxwell (2011)                    | 环境披露理论模型                 | 环境披露、审计威胁            | 审计威胁会改变企业环境披露策略及漂绿激励          | 为监督约束漂绿提供直接理论基础      | 直接  |
| Delmas & Burbano (2011)                  | 漂绿驱动因素理论框架               | 外部、组织与个体驱动因素         | 系统梳理企业漂绿的多层次动因                | 支持合法性压力与转型成本冲突的解释    | 直接  |
| Zhou et al. (2024)                       | 企业环境信息披露                 | 披露型漂绿、企业价值           | 环境披露型漂绿存在短期溢价与价值折损效应          | 支持以披露—绩效偏离刻画漂绿及其经济后果 | 直接  |
| Baier et al. (2020)                      | 企业年度报告                   | ESG 披露、文本分析          | 使用文本方法识别和比较年报中的 ESG 报告内容      | 支持年报 ESG 文本变量构造      | 方法  |
| Xu et al. (2026)                         | 2021—2023 年中国 A 股        | DeepGreen、LLM、环境处罚   | LLM 识别的漂绿信号可用于经验检验并关联环境处罚     | 支持 LLM 漂绿测度          | 直接  |


---

## 三、按逻辑链分层的真实文献清单

> 本清单按“识别机制—约束机制—漂绿定义—边界条件”分层组织，目的不是罗列数量，而是让每一层都能被真实文献支撑。下列文献均已通过 DOI 元数据核对，适合在后续论文正文中按需压缩或展开。

### 3.1 第一层：行业专长如何提升识别能力

- **Balsam et al. (2003)**：行业专长与更高审计质量、较低可操纵应计相关，说明行业知识可以提高识别错报与约束操纵的能力。
- **Owhoso et al. (2002)**：行业专门化团队在审计计划和复核中的错误识别表现更好，直接支持“行业知识—异常识别”链条。
- **Reichelt & Wang (2010)**：全国层面与事务所办公室层面的行业专长具有不同且互补的解释力，说明行业专长不是抽象标签，而是可分层的知识结构。
- **Rose-Green et al. (2011)**：行业专长与内部控制缺陷披露有关，表明行业专长不仅影响财务数字，也影响披露质量。
- **Schmidt & Wilkins (2012)**：审计师质量与审计委员会专长影响重述披露及时性，支持“异常信息被更快识别和披露”的逻辑。

### 3.2 第二层：行业专长如何转化为约束动作

- **Sun & Liu (2013)**：行业专长与董事会治理互动，说明审计师的专业判断会通过治理沟通影响管理层裁量空间。
- **Francis (2011)**：审计质量框架强调审计投入、程序与发现能力之间的系统联系，可为“识别后如何约束”提供结构性解释。
- **Lambert et al. (2007)**：信息披露影响资本成本，说明披露质量变化不是中性的，审计约束有现实经济后果。
- **Cohen & Simnett (2015)**：CSR 和鉴证服务研究议程表明，鉴证服务可以把可信度要求从财务信息延伸到非财务信息。
- **Martínez-Ferrero & García-Sánchez (2017)**：可持续鉴证提高 CSR 信息可信度并影响资本成本，直接支持“约束非财务披露”的有效性。
- **Martínez-Ferrero et al. (2021)**：资本市场会评估 CSR 报告的可信度与准确性，且鉴证质量和重述信息会影响这种评估。
- **Bentley-Goode et al. (2025)**：鉴证提供者选择与可持续报告重述质量相关，说明鉴证机制确实会约束非财务报告质量。
- **Simnett & Huggins (2015)**：integrated reporting 与 assurance 的研究议程，支持审计专业能力向可持续报告扩展。

### 3.3 第三层：漂绿是什么、如何测度

- **Delmas & Burbano (2011)**：系统梳理漂绿驱动因素，奠定“合法性压力—转型成本—漂绿动机”的基础。
- **Lyon & Maxwell (2011)**：在审计威胁下企业环境披露会调整，说明漂绿具有可被监督约束的策略属性。
- **Zhou et al. (2024)**：环境披露型漂绿与企业价值有关，说明漂绿不是纯粹话语问题，而有经济后果。
- **Baier et al. (2020)**：可用文本分析识别年报 ESG 报告内容，为后续构造漂绿指标提供方法基础。
- **Xu et al. (2026)**：LLM 驱动的 DeepGreen 可用于检测象征性与实质性披露，并与环境处罚相联系。

### 3.4 第四层：边界条件与辅助解释

- **Ngelo et al. (2022)**：审计背景高管与实质性 CSR 披露相关，说明“审计思维”可以外溢到非财务披露决策。
- **Harymawan et al.（若采用此路径，建议仅保留已核验的实际作者版）**：可作为前审计背景治理外溢的候选方向，但最终正文应以 DOI 核验后的作者信息为准。
- **Ashforth & Gibbs (1990)**：合法性并非单向驱动，组织可能同时利用合法性修辞与实质性行动，适合放在竞争性解释部分。

### 3.5 使用建议

- 若写论文正文，优先使用第一层和第二层文献构建主逻辑；
- 第三层文献用于定义被解释变量与测度方法；
- 第四层文献只作为边界条件、异质性或竞争性解释，不宜抢占主线。

## 四、理论框架选择与比较

### 3.1 主理论

#### （1）审计风险模型与专业怀疑理论

审计风险由重大错报风险与检查风险共同决定。行业专家审计师之所以更重要，不在于其抽象地“更愿意管”，而在于其更可能通过行业知识形成更强的风险识别与专业怀疑。行业专长使审计师能够将企业的环境叙事与预计负债、固定资产减值、绿色资本化支出、持续经营假设及年报“其他信息”之间的关系重新组织为可审查的证据链。因此，该理论主要解释审计师为何**能够识别**夸大、失真或选择性呈现的环境叙事，并在发现异常后启动后续约束程序。

#### （2）一致性审查与约束执行理论

一旦识别到异常，审计师并不会只停留在“看见问题”这一层，而会进入一致性审查、补证要求和管理层质询的执行阶段。对于年报中的绿色叙事，真正形成治理效果的不是声誉上的抽象压力，而是审计师基于 CAS 1521 对“其他信息”进行阅读、比较、追问、验证和必要修正建议的过程。因此，该理论主要解释审计师如何把识别能力转化为对披露行为的实质性约束。

### 3.2 辅助理论


| 理论      | 对本研究的解释作用                                    |
| ------- | -------------------------------------------- |
| 信息不对称理论 | 漂绿利用实际绩效与利益相关者认知之间的信息差；专家审计师可缩小文本表述与客观绩效的差距。 |
| 制度脱钩理论  | 企业面对合法性压力时可能“多说少做”，为以披露—绩效偏离度衡量漂绿提供依据。       |


### 3.3 不宜作为主理论的框架

- **合法性理论：** 主要解释企业为何漂绿，而非审计师如何治理；高声誉审计师还可能成为企业的“合法性盾牌”。
- **信号理论：** 强调高质量企业主动选择专家审计师，容易引出严重的自选择问题，无法单独证明治理效应。

---

## 四、理论逻辑链

```text
事务所行业专长（IMS / IPS）
          │
          ├── 识别机制
          │   ├─ 行业特定知识
          │   ├─ 异常文本与绿色叙事识别
          │   └─ 财务数据与环境叙事的一致性判断
          │
          └── 约束机制
              ├─ 质询与补证
              ├─ 年报“其他信息”勾稽审查
              ├─ 修改/删减夸大表述
              └─ 降低漂绿的披露空间
                          │
                          ▼
              披露—绩效偏离缩小
                          │
                          ▼
                    企业漂绿程度下降
```

### 4.1 专家与非专家审计师对比


| 逻辑环节   | 行业专家审计师                            | 非行业专家审计师             |
| ------ | ---------------------------------- | -------------------- |
| 行业知识   | 拥有多客户经验、行业惯例知识和技术基准                | 缺乏行业知识，难以把握绿色表述的行业语境 |
| 识别能力   | 更容易发现环境叙事、财务证据与行业惯例之间的不一致          | 更可能把绿色话语视为一般性管理层修辞   |
| 一致性判断  | 更容易将年报“其他信息”与财务数据、减值、资本支出和风险事项进行勾稽 | 更容易把非财务叙事与财务审计割裂开来   |
| 约束执行   | 会触发质询、补证、删改表述和一致性修正建议              | 多停留在形式审查层面，约束力度较弱    |
| 漂绿压缩结果 | 披露—绩效偏离更可能被压缩                      | 漂绿空间更容易保留            |


---

## 五、假设开发（长版）

### 5.1 企业语言与行动的脱钩

随着中国“双碳”战略推进及资本市场对 ESG 的重视，上市公司面临更高的环境合规压力与利益相关者期待。环境披露有助于企业获取合法性、降低融资成本并塑造市场形象，但真实绿色转型需要长期资本投入、技术创新和经营成本，短期内可能压制财务表现。合法性压力与高额转型成本之间的矛盾由此催生漂绿。

漂绿集中体现为“语言”（*Talk*）与“行动”（*Walk*）的制度性脱钩：企业在年报、MD&A 或 ESG 报告中大量使用“碳中和”“绿色转型”等象征性表述，却缺乏环保投入、减排数据或可核验项目。LLM 可识别象征性与实质性披露，并将文本披露程度与同行业、同年度的客观环境绩效进行标准化比较。当披露显著超过实际绩效时，即形成连续的漂绿偏离。

### 5.2 行业专长的形成与测度

会计师事务所行业专长是事务所在特定行业长期提供服务后形成的专业知识、技术工具、经验与人才储备，主要来自两条路径：

1. **规模经济与资源集聚：** 较高的行业市场份额（IMS）支持事务所集中投资专用技术、软件和人才；
2. **学习曲线与知识溢出：** 较高的客户组合份额（IPS）使团队通过重复审计快速掌握行业经营模式与风险特征。

实证研究通常采用连续市场份额、最高市场份额虚拟变量或全国—事务所办公室双层指标识别行业专长。Reichelt 与 Wang（2010）发现，全国层面与办公室层面的行业专长度量具有不同且互补的审计质量解释力。因此，**10% 市场份额阈值只能作为一种可检验的替代口径，不应被表述为普遍成立的固定标准**。专家事务所依靠重复业务、专门人才和跨客户知识转移形成“行业透视能力”。

### 5.3 识别机制：行业知识与异常信息识别

环境绩效与财务状况并非相互隔离。真实污染治理通常伴随环保设备增加、运行费用上升或环保资本支出确认；夸大环境绩效则可能掩盖罚款、资产减值和环境诉讼负债。行业专家审计师不仅与更高的盈余质量和错误识别能力相关（Balsam et al., 2003；Owhoso et al., 2002），还会影响客户披露内部控制缺陷的行为（Rose-Green et al., 2011）。这些证据虽不直接检验漂绿，却共同支持“行业知识提升异常信息识别与披露约束能力”的外推逻辑。更进一步地，行业专长的关键不只是知道“行业里通常会发生什么”，而是能够识别何种披露在行业语境下显得异常、何种绿色表述缺乏可验证支撑、何种环境叙事与财务表现存在结构性冲突。专家审计师可通过三类识别路径发现漂绿：

1. **跨客户基准比较：** 将积极的绿色叙事与同行环保投入、能源成本和产出基准比较；
2. **年报“其他信息”勾稽：** 依据《中国注册会计师审计准则第 1521 号》，识别非财务信息与财务报表或审计中已知事实的重大不一致；
3. **环境风险财务映射：** 将碳配额、停产整治和环境处罚等风险映射至资产减值、预计负债及持续经营判断。

### 5.4 约束机制：一致性审查与披露修正

识别如果不能转化为后续审查，就不会形成治理效果。行业专家事务所的真正影响体现在：一旦识别到绿色叙事与实际表现之间的偏离，它便会将这一偏离转化为可操作的审计后果，包括要求管理层提供补充证据、核对年报“其他信息”的表述、建议删除或修正夸大措辞、重新评估风险事项与披露边界。换言之，约束机制并不依赖审计师是否具有“更强道德偏好”，而依赖其是否具备把异常信息转化为审查动作的能力。对于漂绿研究而言，这一机制更关键，因为漂绿本身就是一种文本层面的策略性表达；只要审计师能够提高管理层进行文本包装的成本，企业就更可能减少象征性修辞或提高实质性披露。

### 5.5 辅助解释：声誉资本与行业传染风险

行业专家事务所的品牌价值依赖其在特定行业建立的高质量声誉。某一客户因漂绿遭受处罚或舆情危机时，同行业其他客户可能质疑事务所的专业能力和独立性，进而导致客户流失。与此同时，中国资本市场的监管与民事追责趋严，也会提高审计失败的预期成本。因此，声誉压力可以作为对识别—约束机制的辅助强化：它解释审计师为何在识别异常后更倾向于采取严格措施，但不应替代“识别”和“约束”本身作为主机制。需要谨慎的是，现有可验证文献更多直接证明行业专长与审计质量、披露质量之间的关系，而“漂绿事件造成事务所行业声誉传染”仍属于待检验的理论推断，不应当作已有直接经验证据。

### 5.6 治理外溢与管理层行为重塑

面对专家审计师的质询、补证要求、一致性审查以及潜在报告延迟，管理层实施漂绿的被发现概率和预期成本上升。其理性应对包括：

- **减少象征性修辞：** 删除缺乏事实依据的绿色口号和模糊承诺；
- **增加实质性行动：** 提高环保投入和客观环境绩效，使其与对外形象相匹配。

两种路径均会缩小绿色披露与实际环境绩效之间的正向偏离，从而降低企业漂绿程度。

> **H1：** 在其他条件不变的情况下，会计师事务所行业专长与客户企业的漂绿程度显著负相关；即事务所行业专长越高，客户企业的漂绿程度越低。
>
> **H1a（识别机制）：** 事务所行业专长通过提高审计师识别特定行业环境风险、异常文本信息及言行脱钩模式的能力，进而抑制企业漂绿。
>
> **H1b（约束机制）：** 事务所行业专长通过增强对年报“其他信息”的一致性审查、补证要求与披露修正压力，进而抑制企业漂绿。

---

## 六、假设开发（精简版）

企业漂绿是管理层在合法性压力下实施的印象管理，表现为年报中的象征性绿色叙事与实际环境绩效脱钩。披露型漂绿会影响企业价值（Zhou et al., 2024），而年报文本分析可用于构造 ESG 披露指标（Baier et al., 2020）。虽然财务审计主要对财务信息发表意见，但 CSR 与可持续鉴证研究表明，审计和鉴证机制能够提高非财务信息的可信度并影响资本市场评价（Cohen & Simnett, 2015；Martínez-Ferrero & García-Sánchez, 2017）。因此，行业专家事务所可能通过知识能力与声誉激励，对年报环境披露产生治理外溢效应。

在识别渠道上，行业专家事务所依靠较高的市场份额和资源投入，形成行业知识库与跨客户基准数据库，能够将环境叙事与资本支出、经营成本、资产减值及预计负债交叉验证，并依据 CAS 1521 识别“其他信息”与财务事实的不一致。在约束渠道上，专家事务所不会止步于“看出问题”，而会将异常识别转化为质询、补证与修正建议，从而压缩企业使用模糊绿色话语进行漂绿的空间。声誉资本可作为辅助解释：它可能强化审计师在识别后的约束强度，但不作为本文的主机制。

专家审计师的监督首先提高了漂绿被识别的概率，继而通过质询、补证、修正和删减等约束动作提高漂绿的执行成本，促使管理层减少象征性修辞或改善实际环境绩效，最终缩小披露—绩效缺口。

> **H1：在其他条件不变的情况下，会计师事务所的审计行业专长与客户企业的漂绿程度显著负相关。**

---

## 七、正式假设与竞争性假设

### 7.1 正式假设

- **H1（主假设）：** 事务所行业专长越高，客户企业漂绿程度越低。
- **H1a（识别机制）：** 行业专长通过增强审计师对环境风险、异常文本和言行脱钩模式的识别能力，降低企业漂绿。
- **H1b（约束机制）：** 行业专长通过提高质询、补证、修正和一致性审查强度，进一步压缩企业漂绿空间。

### 7.2 非方向性备选假设

> **H1_alt：** 会计师事务所行业专长与客户企业漂绿程度显著相关；其方向取决于治理外溢效应与合法性背书效应的净结果。

### 7.3 竞争性假设

> **H1_compete：** 会计师事务所行业专长与客户企业漂绿程度显著正相关。

竞争逻辑是：专家事务所可能因行业收入集中而产生经济依赖，导致独立性受损；同时，企业也可能借助高声誉审计师形成“合法性盾牌”，实施更加隐蔽的精致化漂绿。

---

## 八、竞争性解释与边界条件

### 8.1 竞争性解释

1. **经济依赖与独立性受损：** 关键客户可能“绑架”高度依赖特定行业收入的事务所；
2. **合法性背书与精致化漂绿：** 高声誉审计师的信任红利可能降低外部监督者警惕；
3. **财务审计法定边界：** 纯叙述性 ESG 信息可能不属于财务报表法定审计范围；
4. **重污染行业共生内生性：** 高污染行业既更可能漂绿，也更需要专家审计师，可能造成伪相关。

### 8.2 边界条件


| 调节因素         | 预期作用                               |
| ------------ | ---------------------------------- |
| 重污染行业        | 负向关系更强；环境风险更容易转化为减值、负债与停产风险。       |
| 独立第三方 ESG 鉴证 | 可能产生替代或互补效应；缺乏独立鉴证时，财务审计的外溢作用可能更强。 |
| 国企与民企        | 政治合法性压力、管理层声誉考量和治理机制不同，作用强度可能存在差异。 |
| 地区法治与监管强度    | 执法越严格，违规成本越高，专家事务所的声誉保护动机越强。       |


---

## 九、内生性处理与实证建议

研究中国 A 股 2021—2024 年样本时，应重点处理反向因果、自选择偏误及事务所客户组合的内生形成。

### 9.1 基础模型与固定效应

- 核心解释变量滞后一期，缓解当期双向因果；
- 控制公司固定效应（Firm FE）和行业×年度固定效应（Industry-Year FE）；
- 标准误建议在公司或事务所层面聚类，并根据识别层级进行稳健性比较。

### 9.2 审计师变更 DID

- 处理组：由非行业专家事务所变更为行业专家事务所的企业；
- 对照组：变更后仍为非专家事务所，或始终聘请非专家事务所的企业；
- 使用交错 DID，并检验平行趋势和动态效应，观察聘请专家审计师后漂绿程度是否下降。

### 9.3 PSM 与熵平衡

以企业规模、杠杆率、ROA、独立董事比例、第一大股东持股、机构持股及行业污染属性等估计选择专家事务所的 Logit 模型。实施 1:1 或 1:2 近邻匹配，并使用熵平衡检验高维协变量平衡性。

### 9.4 工具变量

可考虑同省或同城市其他上市公司聘请该事务所的平均比例，或历史区域事务所行业聚集度。工具变量应同时满足相关性与排除性约束，并进行弱工具变量检验。

### 9.5 机制检验

- **识别机制：** 审计费用、审计报告时滞、环境相关关键审计事项（KAMs）及异常文本识别；
- **约束机制：** 质询次数、补证要求、披露修正与删改行为；
- **替代解释排除：** 控制 ESG 鉴证、事务所规模、客户重要性及审计师变更动机。

---

## 十、参考文献

1. Ashforth, B. E., & Gibbs, B. W. (1990). The double-edge of organizational legitimation. *Organization Science, 1*(2), 177–194. [https://doi.org/10.1287/orsc.1.2.177](https://doi.org/10.1287/orsc.1.2.177)
2. Balsam, S., Krishnan, J., & Yang, J. S. (2003). Auditor industry specialization and earnings quality. *Auditing: A Journal of Practice & Theory, 22*(2), 71–97. [https://doi.org/10.2308/aud.2003.22.2.71](https://doi.org/10.2308/aud.2003.22.2.71)
3. Bills, K. L., Jeter, D. C., & Stein, S. E. (2015). Auditor industry specialization and evidence of cost efficiencies in homogenous industries. *The Accounting Review, 90*(5), 1721–1754. [https://doi.org/10.2308/accr-51003](https://doi.org/10.2308/accr-51003)
4. Delmas, M. A., & Burbano, V. C. (2011). The drivers of greenwashing. *California Management Review, 54*(1), 64–87. [https://doi.org/10.1525/cmr.2011.54.1.64](https://doi.org/10.1525/cmr.2011.54.1.64)
5. Francis, J. R. (2011). A framework for understanding and researching audit quality. *Auditing: A Journal of Practice & Theory, 30*(2), 125–152. [https://doi.org/10.2308/ajpt-50006](https://doi.org/10.2308/ajpt-50006)
6. Ngelo, A. A., Permatasari, Y., Rasid, S. Z. A., Harymawan, I., & Ekasari, W. F. (2022). Ex-auditor CEOs and corporate social responsibility (CSR) disclosure: Evidence from a voluntary period of sustainability report in Indonesia. *Sustainability, 14*(18), 11418. [https://doi.org/10.3390/su141811418](https://doi.org/10.3390/su141811418)
7. Hogan, C. E., & Jeter, D. C. (1999). Industry specialization by auditors. *Auditing: A Journal of Practice & Theory, 18*(1), 1–17. [https://doi.org/10.2308/aud.1999.18.1.1](https://doi.org/10.2308/aud.1999.18.1.1)
8. Lambert, R., Leuz, C., & Verrecchia, R. E. (2007). Accounting information, disclosure, and the cost of capital. *Journal of Accounting Research, 45*(2), 385–420. [https://doi.org/10.1111/j.1475-679X.2007.00238.x](https://doi.org/10.1111/j.1475-679X.2007.00238.x)
9. Lyon, T. P., & Maxwell, J. W. (2011). Greenwash: Corporate environmental disclosure under threat of audit. *Journal of Economics & Management Strategy, 20*(1), 3–41. [https://doi.org/10.1111/j.1530-9134.2010.00282.x](https://doi.org/10.1111/j.1530-9134.2010.00282.x)
10. Owhoso, D. J., Messier, W. F., Jr., & Lynch, J. G., Jr. (2002). Error detection by industry-specialized teams during sequential audit review. *Journal of Accounting Research, 40*(3), 883–900. [https://doi.org/10.1111/1475-679X.00075](https://doi.org/10.1111/1475-679X.00075)
11. Solomon, I., Shields, M. D., & Whittington, O. R. (1999). What do industry-specialist auditors know? *Journal of Accounting Research, 37*(1), 191–208. [https://doi.org/10.2307/2491403](https://doi.org/10.2307/2491403)
12. Sun, J., & Liu, G. (2013). Auditor industry specialization, board governance, and earnings management. *Managerial Auditing Journal, 28*(1), 45–64. [https://doi.org/10.1108/02686901311282498](https://doi.org/10.1108/02686901311282498)
13. Xu, C., Liu, J., Li, Z., & Lin, C. (2026). DeepGreen: Effective LLM-driven greenwashing monitoring system designed for empirical testing—Evidence from China. *Computational Economics*. Advance online publication. [https://doi.org/10.1007/s10614-026-11328-5](https://doi.org/10.1007/s10614-026-11328-5)
14. Reichelt, K. J., & Wang, D. (2010). National and office-specific measures of auditor industry expertise and effects on audit quality. *Journal of Accounting Research, 48*(3), 647–686. [https://doi.org/10.1111/j.1475-679X.2009.00363.x](https://doi.org/10.1111/j.1475-679X.2009.00363.x)
15. Rose-Green, E., Huang, H.-W., & Lee, C.-C. (2011). The association between auditor industry specialization and firms' disclosure of internal control weaknesses. *International Journal of Auditing, 15*(2), 204–216. [https://doi.org/10.1111/j.1099-1123.2011.00430.x](https://doi.org/10.1111/j.1099-1123.2011.00430.x)
16. Schmidt, J., & Wilkins, M. S. (2013). Bringing darkness to light: The influence of auditor quality and audit committee expertise on the timeliness of financial statement restatement disclosures. *Auditing: A Journal of Practice & Theory, 32*(1), 221–244. [https://doi.org/10.2308/ajpt-50307](https://doi.org/10.2308/ajpt-50307)
17. Cohen, J. R., & Simnett, R. (2015). CSR and assurance services: A research agenda. *Auditing: A Journal of Practice & Theory, 34*(1), 59–74. [https://doi.org/10.2308/ajpt-50876](https://doi.org/10.2308/ajpt-50876)
18. Martínez-Ferrero, J., & García-Sánchez, I.-M. (2017). Sustainability assurance and cost of capital: Does assurance impact on credibility of corporate social responsibility information? *Business Ethics: A European Review, 26*(3), 223–239. [https://doi.org/10.1111/beer.12152](https://doi.org/10.1111/beer.12152)
19. Martínez-Ferrero, J., Ruiz-Barbadillo, E., & Guidi, M. (2021). How capital markets assess the credibility and accuracy of CSR reporting: Exploring the effects of assurance quality and CSR restatement issuance. *Business Ethics, the Environment & Responsibility, 30*(4), 551–569. [https://doi.org/10.1111/beer.12355](https://doi.org/10.1111/beer.12355)
20. Simnett, R., & Huggins, A. L. (2015). Integrated reporting and assurance: Where can research add value? *Sustainability Accounting, Management and Policy Journal, 6*(1), 29–53. [https://doi.org/10.1108/SAMPJ-09-2014-0053](https://doi.org/10.1108/SAMPJ-09-2014-0053)
21. Bentley-Goode, K. A., Simnett, R., Thompson, A., & Trotman, A. J. (2025). Choice of assurance provider and impact on quality of sustainability reporting: Evidence from sustainability reporting restatements. *Accounting & Finance, 65*(2), 2135–2172. [https://doi.org/10.1111/acfi.13367](https://doi.org/10.1111/acfi.13367)
22. Zhou, Y., Chen, L., Zhang, Y., & Li, W. (2024). “Environmental disclosure greenwashing” and corporate value: The premium effect and premium devalue of environmental information. *Corporate Social Responsibility and Environmental Management, 31*(3), 2424–2438. [https://doi.org/10.1002/csr.2698](https://doi.org/10.1002/csr.2698)
23. Baier, P., Berninger, M., & Kiesel, F. (2020). Environmental, social and governance reporting in annual reports: A textual analysis. *Financial Markets, Institutions & Instruments, 29*(3), 93–118. [https://doi.org/10.1111/fmii.12132](https://doi.org/10.1111/fmii.12132)

---

## 十一、补充资料链接

- [DeepGreen（arXiv）](https://arxiv.org/html/2504.07733v2)
- [Environmental Responsibility, Social Responsibility, and Governance from the Perspective of Auditors（PMC）](https://pmc.ncbi.nlm.nih.gov/articles/PMC9564669/)
- [Ex-Auditor CEOs and CSR Disclosure（MDPI）](https://www.mdpi.com/2071-1050/14/18/11418)
- [Auditor Industry Specialization and Earnings Quality（ResearchGate）](https://www.researchgate.net/publication/228225800_Auditor_Industry_Specialization_and_Earnings_Quality)
- [National and Office-Specific Measures of Auditor Industry Expertise（ResearchGate）](https://www.researchgate.net/publication/46541445_National_and_Office-Specific_Measures_of_Auditor_Industry_Expertise_and_Effects_on_Audit_Quality)
- [Impact of Sustainability Reporting Level on Audit Report Lag](https://ajajournal.org/index.php/aja/article/view/631/906)
- [企业漂绿的“同构”行为（PDF）](https://cssd1992.org/files/TWMcSUIB8xSmWLf84U4x_%E4%BC%81%E4%B8%9A%E6%BC%82%E7%BB%BF%E7%9A%84%E2%80%9C%E5%90%8C%E6%9E%84%E2%80%9D%E8%A1%8C%E4%B8%BA.pdf)

---

