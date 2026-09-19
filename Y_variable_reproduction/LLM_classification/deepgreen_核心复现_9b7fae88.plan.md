---
name: DeepGreen 核心复现
overview: 基于 `appendix` 中 2021–2024 年年报 TXT 与华证 ESG 年度 CSV，复现制造业样本的章节抽取、Layer A、RAG 增强 Layer B、GI 及 Greenwashing 二元变量。采用单一模型 `qwen3.6-35b-a3b-fp8`，不重复论文的多模型比较与消融实验，但保留可追溯日志、断点续跑和必要的数据质检。
todos:
  - id: build-panel
    content: 整理 2021–2024 制造业公司—年度样本，处理年报版本、证券代码和华证 E 得分匹配
    status: pending
  - id: extract-environment
    content: 实现并质检年报环境信息章节抽取、分句、分词和 S1 构造
    status: pending
  - id: run-layer-a
    content: 接入指定 Qwen 接口完成 Layer A 去重词判断并生成绿色词典与 S2
    status: pending
  - id: run-layer-b-rag
    content: 实现可审计 RAG 检索与 Layer B 实质/象征披露判断，支持缓存和断点续跑
    status: pending
  - id: construct-variables
    content: 按公司—年度汇总 X、Y、GI，并结合行业年度华证 E 得分构造 Greenwashing
    status: pending
  - id: validate-export
    content: 完成抽样复核、完整性检查并导出明细、证据、最终面板与异常报告
    status: pending
isProject: false
---

# DeepGreen 核心复现计划

## 1. 明确口径并建立制造业面板

- 数据源：[`appendix/`](appendix/) 下四个年度 TXT 文件夹，以及 [`appendix/2.华证指数esg 评级评分（年度） 2009-2024(排序).csv`](appendix/2.华证指数esg%20评级评分（年度）%202009-2024(排序).csv)。
- 从文件名解析 `证券代码—年份—简称—发布日期`，证券代码统一补齐为 6 位；同一公司年度存在原版、更正/修订版时保留发布日期最晚且优先标有更正/修订的版本，并输出去重记录。
- 以华证 CSV 的 `证监会行业新` 筛选制造业，限定 2021–2024；限定沪深 A 股代码范围，按论文口径剔除当年简称含 `ST`、`*ST/＊ST`、`PT` 的样本。以公司—年度为主键连接年报和 `E得分`，输出匹配率、缺失和重复清单。
- 行业均值默认按“同一年度 × 证监会制造业细分行业”计算，避免跨年度评分尺度变化；同时保留行业字段，便于复核论文公式中的行业平均值口径。

## 2. 抽取环境信息文本

- **独立处理文件夹：**`deepgreen_reproduction/02_environment_extraction/`。本步骤的代码、运行日志和全部中间结果只写入该文件夹，不改写 `appendix/` 中的原始 TXT。
- **建议保存路径：**抽取后的环境原文保存至 `deepgreen_reproduction/02_environment_extraction/environment_text/`，分句结果保存至 `deepgreen_reproduction/02_environment_extraction/sentences/`，S1 保存至 `deepgreen_reproduction/02_environment_extraction/s1/`，质检与失败清单保存至 `deepgreen_reproduction/02_environment_extraction/reports/`。
- 从年报 TXT 正文而非目录定位“第五节 环境和社会责任/环境与社会责任”等章节，再截取其中“一、环境信息/重大环保问题/环境保护相关情况”等环境部分，终止于“二、社会责任”或下一节。
- 为标题差异、目录重复、OCR 空格和换行设计分级规则；记录 `exact/fallback/failed` 抽取状态，并保存起止位置和原文证据。
- 按中文句末标点、换行及表格行进行句子切分；使用 Jieba 与 HIT 停用词表形成去重词序列 `S1`。数字、页码、纯标点和明显乱码剔除，但保留文件名、化学物质、排放指标和环保术语。
- 对无环境披露或抽取失败分别编码：前者按论文规则候选 `GI=0`，后者保持缺失并进入人工复核，避免把技术失败误判为零披露。

## 3. Layer A：构建绿色关键词词典

- **独立处理文件夹：**`deepgreen_reproduction/03_layer_a/`。本步骤只读取步骤 2 的 S1，并将 API 输出、缓存及 S2 全部保存在该文件夹。
- **建议保存路径：**全局唯一词表保存至 `deepgreen_reproduction/03_layer_a/input/`，逐词判断及原始响应保存至 `deepgreen_reproduction/03_layer_a/results/`，请求缓存保存至 `deepgreen_reproduction/03_layer_a/cache/`，绿色关键词词典保存至 `deepgreen_reproduction/03_layer_a/dictionary/`，供 Layer B 使用的 S2 保存至 `deepgreen_reproduction/03_layer_a/s2/`，日志和失败队列保存至 `deepgreen_reproduction/03_layer_a/logs/`。
- 通过 OpenAI 兼容接口 `BASE_URL=https://token.nau.edu.cn/v1` 调用 `qwen3.6-35b-a3b-fp8`；API Key 仅从环境变量读取，不写入代码或结果。
- 对全样本 `S1` 的全局唯一词汇调用论文 Appendix B 的 Layer A 中文提示词，强制返回 `{"judgment": 0/1, "confidence": 0-1}`；低温度、固定参数，解析失败自动重试。
- 将判定为绿色的词汇组成全局词典，并保存每个词的判断、置信度、原始响应、模型名、提示词版本和时间戳；缓存唯一词结果，避免跨公司重复付费。
- 用词典回查环境文本，提取所有包含目标词的完整句子，形成 Layer B 输入 `S2 = 关键词—句子—公司—年度`；同一句多个关键词分别保留，以对应论文的关键词级判断。

## 4. Layer B：实质性披露判断

- **独立处理文件夹：**`deepgreen_reproduction/04_layer_b_rag/`。本步骤只读取步骤 3 的 S2，RAG 检索记录和模型判断不得与 Layer A 结果混放。
- **建议保存路径：**S2 输入副本/索引保存至 `deepgreen_reproduction/04_layer_b_rag/input/`，检索查询和证据保存至 `deepgreen_reproduction/04_layer_b_rag/rag_evidence/`，Layer B 判断明细及原始响应保存至 `deepgreen_reproduction/04_layer_b_rag/results/`，请求与检索缓存保存至 `deepgreen_reproduction/04_layer_b_rag/cache/`，失败队列和运行日志保存至 `deepgreen_reproduction/04_layer_b_rag/logs/`。
- 按论文四类规则判断每个 `关键词—完整句子` 是否体现企业或子公司的具体环保行动；`1=实质性披露 X`，`0=象征性披露 Y`。
- 保存检索查询、来源、证据文本、最终判断、置信度及原始响应；支持并发限速、指数退避、请求缓存、失败队列和断点续跑。先对少量公司试跑，检查 JSON 合规、证据质量和成本，再执行全量。

## 5. 汇总 GI 与构造 Greenwashing

- **独立处理文件夹：**`deepgreen_reproduction/05_variables/`。本步骤只读取前述步骤产物和华证 ESG CSV，不回写任何上游中间结果。
- **建议保存路径：**公司—年度 X/Y/GI 汇总保存至 `deepgreen_reproduction/05_variables/gi/`，行业—年度均值保存至 `deepgreen_reproduction/05_variables/industry_benchmarks/`，Greenwashing 最终变量及合并面板保存至 `deepgreen_reproduction/05_variables/final_panel/`，描述统计、异常、缺失和校验报告保存至 `deepgreen_reproduction/05_variables/reports/`。
- 公司—年度层面汇总：`X=实质性关键词—句子对数量`，`Y=象征性关键词—句子对数量`，`GI=X/(X+Y)`；当确认为无相关披露或仅象征披露时 `GI=0`，抽取/API 未完成时保持缺失。
- 同时输出 `X`、`Y`、总披露量、唯一关键词数、唯一句子数和平均置信度，避免只留下不可审计的最终指标。
- 在每个“年度 × 证监会细分行业”内计算有效样本的 `GI_mean` 与 `ESGE_mean`，按论文公式构造：`Greenwashing=1` 当且仅当 `GI > GI_mean` 且 `E得分 < ESGE_mean`，否则为 0；行业样本过少时标记并提供合并到制造业大类的备选值，不直接混用。
- 导出公司—年度最终面板、Layer A 词典、Layer B 明细、RAG 证据、异常/缺失报告和年度描述统计；核验 `GI∈[0,1]`、二元变量合法、主键唯一及 2021–2024 年覆盖情况。

### 5.1 步骤 2–5 目录总览

```text
deepgreen_reproduction/
├── 02_environment_extraction/
│   ├── environment_text/
│   ├── sentences/
│   ├── s1/
│   └── reports/
├── 03_layer_a/
│   ├── input/
│   ├── results/
│   ├── cache/
│   ├── dictionary/
│   ├── s2/
│   └── logs/
├── 04_layer_b_rag/
│   ├── input/
│   ├── rag_evidence/
│   ├── results/
│   ├── cache/
│   └── logs/
└── 05_variables/
    ├── gi/
    ├── industry_benchmarks/
    ├── final_panel/
    └── reports/
```

- 以上均为后续执行时创建的目标路径；当前阶段只在计划中确定目录规范，不提前创建空文件夹。
- 各步骤之间仅通过明确的产出文件传递数据，原则上不跨步骤覆盖文件，以便单独重跑 Layer A、Layer B 或变量构造。

## 6. 复核与交付标准

- 随机抽查章节抽取结果，并从 Layer A 正负例、Layer B 实质/象征例及边界置信度样本中各抽取一批供人工复核；这不是重复论文实验，而是保证本次变量可用的最低质量控制。
- 提供可配置入口，包括年份、样本数量、并发数、模型参数、是否启用 RAG 和重跑失败项；全流程可从原始 TXT/CSV 重建最终变量。
- 最终重点对照论文第三节的 `S1 → Layer A → S2 → RAG Layer B → X/Y → GI` 以及第五节公式 (8)，并在结果中明确说明本复现使用单一 Qwen 模型、扩展到 2024 年且样本限定制造业，因此不以数值复刻论文 2021–2023 全 A 股结果为目标。

## 7. 样本范围确认

- 本次复现先从 `appendix` 各年度文件夹的年报 TXT 中筛选制造业上市公司，不处理非制造业公司。
- 制造业识别以华证 ESG CSV 中的 `证监会行业新` 为准，并以 `证券代码—年份` 匹配年报；不能成功匹配行业的 TXT 不直接纳入制造业样本，而是进入待复核清单。
- 时间范围为 2021–2024 年。该范围比论文的 2021–2023 年多一年，因此最终样本数、绿色词典、GI 分布及 Greenwashing 比例不要求与论文完全一致。

## 8. Layer A 与 Layer B 提示词

> 说明：论文 Appendix B 声明实际处理中文年报时输入均为中文，但论文为方便读者理解，只刊载了英文译文。以下内容是对 Appendix B 刊载文本的忠实中文回译，不额外增加或删减判断标准。实际调用时，仅把待判断词语以及完整句子填入相应位置；JSON 输出约束保持不变。

### 8.1 Layer A 提示词

**角色锚定（Role Anchoring）**

你是一名外部 ESG 分析师，最关注公司财务报告 ESG 部分中的环境方面。

**人物设定（Persona Setting）**

你非常熟悉公司 ESG 的各个维度，包括但不限于：绿色、环境保护、环境、清洁能源、节能、降碳、低碳、减排、污水处理、空气污染、公司环境标准、国家环境法规等。

**任务描述（Task Description）**

你是一名诚实的分析师，不会编造材料。你需要判断给定词语是否为与公司绿色披露相关的关键词。

待判断词语：`{{word}}`

**回答模板（Answer Template）**

请严格遵循以下 JSON Schema 格式，仅返回纯 JSON 字符串，不要返回任何不必要的解释：

```json
{"judgment": <int>, "confidence": <float>}
```

其中，`judgment = 1` 表示肯定，`judgment = 0` 表示否定；`confidence` 的取值范围为 0 到 1。

### 8.2 Layer B 提示词

**角色锚定（Role Anchoring）**

你是一名外部 ESG 分析师，最关注公司财务报告 ESG 部分中的环境方面。

**人物设定（Persona Setting）**

你非常熟悉公司 ESG 的各个维度，包括但不限于：绿色、环境保护、环境、清洁能源、节能、降碳、低碳、减排、污水处理、空气污染、公司环境标准、国家环境法规等。

**任务描述（Task Description）**

你是一名诚实的分析师，不会编造材料。你将依次获得：（1）一个词语；（2）一个包含该词语的完整句子（该词语使用 `##` 标记）。你收到的词语主要分为以下四类：

（a）带有书名号的词语，通常是某一文件。你需要根据给定语境判断，该文件是否由公司自身发布、响应或制定，用于解决公司自身及其子公司的环境问题。例如，《XX公司环境项目评估》《XX公司污染防治方案》等文件，表明这是公司自愿披露其环境信息。但是，《环境保护法》《XX市污染治理要求》等文件由国家、政府及其他非公司主体发布，即使公司响应了该文件，也不属于公司自愿披露自身相关环境信息。

（b）化学物质或化学术语，例如“锰”“氧化物”“缩合反应”等。你需要根据给定语境判断，公司是否通过技术手段处理了该化学物质，或者是否在公司的生产经营过程中通过这种化学手段缓解了某一类环境问题。

（c）在语义上与环境、环境保护和绿色直接相关的词语，例如“环境保护部”“绿色生产”“碳排放”等。你需要根据给定语境判断，公司是否通过发布具体文件或采用具体技术方法对该词语所指事项作出了贡献，或者公司是否通过该词语所指事项缓解了其生产经营过程中的某一类环境问题。例如，“公司响应碳排放政策”只是一句没有实际行动的口号，而“公司响应碳排放政策并净化污染气体”则包含实际行动。

（d）其他通常与公司和工厂相关联的词语，例如“大烟囱”。你需要根据给定语境判断，公司是否实际且具体地处理了该污染性词语所指对象，或者是否实际且具体地促进了该有益于环境的词语所指事项。

待判断词语：`{{word}}`

包含该词语的完整句子：`{{sentence_with_marked_word}}`

**回答模板（Answer Template）**

请严格遵循以下 JSON Schema 格式，仅返回纯 JSON 字符串，不要返回任何不必要的解释：

```json
{"judgment": <int>, "confidence": <float>}
```

其中，`judgment = 1` 表示肯定，`judgment = 0` 表示否定；`confidence` 的取值范围为 0 到 1。

### 8.3 RAG 使用边界

- RAG 是 Layer B 的外部证据补充机制，不改变上述 Appendix B 判断标准与 JSON 输出格式。
- 为保证提示词复现的可比性，论文提示词正文保持原样；检索证据应作为单独的上下文块注入，而不改写四类任务规则。
- 每次判断仍以 `{{word}}` 和 `{{sentence_with_marked_word}}` 为核心输入，并保存检索证据及来源以供复核。