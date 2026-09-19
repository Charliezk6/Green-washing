# 漂绿因变量 LLM 分类验证方案

## 1. 文档目的

本文档回答三个问题：

1. 如果仿照 DeepGreen 论文，应如何实施 Layer A 和 Layer B 的人工检验；
2. 除该论文的随机抽样人工检验外，还有哪些可用于高水平期刊投稿的检验；
3. 检验方式应依据什么确定，而不是任意选择样本量和指标。

本方案对应当前流程：

```text
年报环境章节
→ Layer A：从去重词序列识别绿色候选词
→ 回查包含候选词的完整句子
→ Layer B：判断关键词—句子对属于实质性披露（1）或象征性披露（0）
→ 汇总 X、Y、GI
→ 结合华证 E 得分构造 Greenwashing
```

需要区分三类证据：

- **分类准确性**：LLM 标签与独立人工金标准是否一致；
- **分类可靠性**：换模型、提示词或重复运行后是否稳定；
- **构念效度**：分类结果汇总成的漂绿变量是否确实反映漂绿。

三者不能互相替代。多模型一致不能单独证明真实准确，回归显著也不能单独证明单条文本标签正确。

---

## 2. DeepGreen 论文实际采用了什么检验

DeepGreen 在第 4.2 节说明：

- Layer A：每轮随机抽取 100 个词，重复 10 轮；
- Layer B：随机抽取 500 个“关键词—上下文”对，重复相同的人工验证程序；
- 人工标准：由若干具有金融背景的学生进行人工标注；
- 指标：Accuracy、F1-score 和 Matthews Correlation Coefficient（MCC）；
- Layer A 比较四个模型：LLaMA3-8B-Instruct、Kimi-K2、DeepSeek-V3.1 和 Qwen3-plus；
- 依据人工标签上的表现及模型置信度分布选择 Layer A 冠军模型；
- Layer B 使用冠军模型比较原始、RAG 和扩展上下文三种配置；
- 最终选择人工验证表现最好的配置生成 GI。

论文没有充分交代以下细节：

- 人工标注者具体人数；
- 是否独立盲标；
- 是否向人工显示模型预测；
- 分歧如何裁决；
- 标注者间一致性；
- 100 个词的每轮样本是否互不重叠；
- Layer B 的 500 条是每轮 500 条，还是总计 500 条；
- 抽样是否按类别、行业、年份分层；
- 人工编码手册及训练过程；
- 是否将用于修改提示词的样本与最终测试样本分开。

因此，可以仿照其核心思路，但不应机械复制其信息不完整的部分。更规范的实现应预先注册抽样、盲标、裁决和统计规则。

---



## 3. 仿照论文实施人工检验的推荐版本



## 3.1 基本原则

人工标签必须尽可能独立于被检验的 LLM：

1. 标注者看不到模型标签、confidence 和模型理由；
2. 标注者分别作答，不能先讨论后形成共同标签；
3. 先冻结编码手册，再标正式测试集；
4. 用于修改 prompt 的开发集不能再充当最终测试集；
5. 分歧由第三名裁决者处理，不能让原模型裁决；
6. 保留每名标注者的原始标签，而不只保存裁决结果。

推荐至少安排：

- 标注者 A：金融、会计或管理背景；
- 标注者 B：环境、ESG、可持续发展或相关背景；
- 裁决者 C：研究负责人或具有相关研究经验者。

如果只能找到相同专业背景的学生，也可以实施，但应如实报告其资格、培训和编码手册，不能称为“行业专家”。

## 3.2 编码手册开发

正式标注前建立书面 coding manual，至少包括：

- 研究构念定义；
- 正类与负类的操作化定义；
- 纳入和排除规则；
- 企业、子公司、政府和第三方主体的处理；
- “已实施、正在实施、计划、拟、力争”等时态处理；
- 政策响应、许可、处罚、排放数据和栏目标题的处理；
- 一句话中同时存在象征性和实质性内容时如何处理；
- 信息不足或文本损坏时如何编码；
- 每类至少 10—20 个正反例和边界例。

实施两轮预测试：

1. 第一轮共同培训 50 条，讨论规则但不纳入正式评价；
2. 第二轮独立标注另 50—100 条，计算一致性；
3. 只有当一致性达到预设要求后，冻结手册并进入正式测试。

建议预测试标准：

- percent agreement ≥ 80%；
- Cohen’s \kappa 或 Krippendorff’s \alpha ≥ 0.70；
- 若低于标准，修订定义并重新预测试。

0.70 不是绝对真理，而是内容分析中常用的最低可接受参考；核心变量和高风险判断最好达到 0.80 左右。

---



## 4. Layer A 人工检验怎么做



## 4.1 若严格仿照 DeepGreen

评价单位为去重后的单个词，人工看到的字段只包括：


| 字段             | 是否展示  |
| -------------- | ----- |
| 匿名样本 ID        | 是     |
| 目标词            | 是     |
| LLM 标签         | 否     |
| LLM confidence | 否     |
| 来源公司和年份        | 否或匿名化 |
| 完整句子           | 否     |


问题表述可为：

> 该词在公司环境信息披露中是否可以作为检索潜在绿色披露的候选关键词？判断标准应偏向避免遗漏，而不是要求该词在所有语境中均表示绿色。

标签建议不只设置 0/1，而先由人工使用：

- 1：应纳入候选词；
- 0：不应纳入；
- U：孤立词无法判断。

主指标计算时可以：

- 将 U 交由第三人裁决为 0/1；
- 同时单独报告 U 的比例；
- 不应静默删除 U，因为它正好反映 Layer A 的上下文歧义。

抽样：

- 固定随机种子；
- 从 Layer A 全部被判断词中按模型预测 0/1 分层；
- 每轮 100 个词，共 10 轮；
- 推荐十轮互不重叠，得到 1,000 个唯一词；
- 如果重复抽样，应报告唯一词数量和重复率。

统计：

- 每轮及十轮合并的 Accuracy、Precision、Recall、F1、MCC；
- 人工标注者之间的 agreement、Cohen’s \kappa；
- 指标的 95% bootstrap 置信区间；
- U 比例和典型边界词。



## 4.2 为什么严格仿照还不够

Layer A 的功能是候选检索，而非最终语义分类。孤立词是否“绿色”存在上下文依赖：同一个“循环”“治理”“处理”在不同句子中可能完全不同。因此，单词准确率只能检验模型是否复现人工对词表的直觉，不能证明它没有漏掉相关句子。

Layer A 对假阴性和假阳性的成本也不对称：

Cost(FN) \gg Cost(FP)

假阳性会进入 Layer B 后被过滤，假阴性会永久消失。因此 Layer A 应以召回率和遗漏审计为核心，而不是只报告 Accuracy。

## 4.3 推荐增加：上下文化 Layer A 检验

在上述单词检验之外，再抽取“词—完整句子”对。人工回答：

> 在该完整句子中，目标词是否触发了一条值得送入 Layer B 进一步判断的环境披露候选？

这样可区分：

- 词本身通常绿色；
- 词在当前语境绿色；
- 词虽然含义模糊，但作为高召回候选是合理的。

建议至少抽 500 对，并按以下维度分层：

- Layer A 预测 0/1；
- 高频词/低频词；
- 制造业/非制造业；
- 年份；
- 一词多义程度；
- 目标词所在句子的长度。



## 4.4 推荐增加：未召回文本的反向遗漏审计

这是 Layer A 最重要的补充检验。

1. 从原始环境章节的所有句子中排除已被 Layer A 召回的句子；
2. 对剩余句子进行行业—年度分层随机抽样；
3. 人工判断句子是否含有潜在环境披露；
4. 对确认相关但未召回的句子，记录遗漏原因和应新增候选词；
5. 根据抽样权重估计未召回总体中的绿色句子数量。

估计遗漏数：

\widehat N_{miss}=\sum_h N_h\frac{m_h}{n_h}

其中 h 为抽样层，N_h 为该层未召回句子总量，n_h 为人工审计数，m_h 为发现的遗漏数。

如果能够估计已召回部分的真实绿色数量，可进一步估计：

\widehat{Recall}*A=
\frac{\widehat N*{retrieved,true}}
{\widehat N_{retrieved,true}+\widehat N_{miss}}

该检验直接回答“Layer A 有没有漏掉绿色披露”，比孤立词准确率更符合检索层的功能。

---



## 5. Layer B 人工检验怎么做



## 5.1 人工任务

评价单位与当前数据完全一致：一个“目标词—完整句子”对。人工只看到：

- 匿名样本 ID；
- 目标词；
- 标记目标词的完整句子；
- 必要时提供前后各一句，但所有样本必须采用同一规则。

人工问题：

> 该句是否披露了企业或其子公司围绕目标词已经实施或正在实施的具体环保行动？

建议标签：

- 1：实质性披露；
- 0：象征性披露或不构成实际环保行动；
- U：信息不足、文本损坏或无法可靠判断。

核心判定标准：

### 编码为 1

- 建设、安装或运行环保设施；
- 完成节能、减排或清洁生产改造；
- 实际处理、回收、净化污染物；
- 已实施企业自身具体环保方案；
- 明确主体、对象和行动，能对应到企业经营活动。



### 编码为 0

- 栏目标题；
- 绿色愿景、口号或泛泛承诺；
- “重视环保”“坚持绿色发展”；
- 仅遵守、引用或响应法规；
- 仅列示许可证、污染物或排放数据；
- 仅陈述未来计划且没有已实施行动；
- 主体不是企业或其子公司；
- 行动或对象不明确。

边界情况必须在 coding manual 中提前固定。例如：

- “推进改造”是否要求有已启动证据；
- “持续加强治理”通常为 0，除非紧随具体措施；
- 同一句既有口号又有行动，只要存在明确行动可编码为 1；
- 处罚后的整改若已实施可为 1，只有整改要求则为 0。



## 5.2 抽样方式



### 最小复刻版

- 随机抽取 500 个关键词—句子对；
- 两名人工独立盲标；
- 第三人裁决分歧；
- 以裁决后的标签作为 gold standard；
- 计算 Accuracy、F1、MCC。



### 推荐投稿版

简单随机 500 条可能因类别不平衡而几乎抽不到少数类或困难样本。推荐分层抽取 800—1,000 条：


| 分层变量       | 建议                       |
| ---------- | ------------------------ |
| 模型预测标签     | 0/1 各占约一半                |
| confidence | 高、中、低均覆盖，低置信度适度过抽        |
| 行业         | 制造业与非制造业，重点行业均覆盖         |
| 年份         | 2021—2024 均覆盖            |
| 词频         | 高频词和低频词均覆盖               |
| 重复结构       | 一句单词与一句多关键词均覆盖           |
| 边界表达       | 计划、推进、遵守、许可、处罚、数据列示等适度过抽 |


因为这是不等概率分层抽样，应同时报告：

1. 分层样本中的分类别表现；
2. 按总体抽样概率加权后的总体表现；
3. 不要把人为 50/50 的 pooled accuracy 直接称为总体准确率。



## 5.3 人工金标准形成

保存以下字段：

```text
sample_id
word
marked_sentence
model_label
model_confidence
coder_a_label
coder_a_reason
coder_b_label
coder_b_reason
adjudicated_label
adjudicator_reason
sampling_stratum
sampling_probability
```

顺序必须是：先完成人工盲标和裁决，再与模型标签合并。向标注者提供的文件中不应出现 `model_label` 和 `model_confidence`。

## 5.4 必须报告的指标

以人工裁决标签为真值：

- 混淆矩阵 TP、TN、FP、FN；
- Accuracy；
- 正类 Precision、Recall、F1；
- 负类的分类别指标或 macro-F1；
- MCC；
- Balanced Accuracy；
- 每个指标的 95% 置信区间；
- 两名人工之间的 agreement 和 Cohen’s \kappa；
- 如果保留 U，报告 coverage：模型可判定样本占比；
- 按行业、年份、句长和 confidence 的子组表现。

其中：

Accuracy=\frac{TP+TN}{TP+TN+FP+FN}

Precision=\frac{TP}{TP+FP},\qquad
Recall=\frac{TP}{TP+FN}

F1=\frac{2\cdot Precision\cdot Recall}{Precision+Recall}

MCC=\frac{TP\cdot TN-FP\cdot FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}

MCC 适合类别不平衡，F1 强调正类，但两者都不能替代完整混淆矩阵。

## 5.5 样本量依据

样本量不应只因为原论文抽 500 条就固定为 500。若目标是估计某一比例 p，95% 置信区间半宽为 d，简单随机抽样近似为：

n\approx \frac{1.96^2p(1-p)}{d^2}

例如预期准确率为 90%，希望误差不超过 ±3 个百分点：

n\approx \frac{1.96^2\times0.9\times0.1}{0.03^2}\approx384

考虑类别指标、子组分析、分层设计和无效样本，建议至少 500—800 条；若需稳定比较制造业与非制造业或多个模型，建议 800—1,200 条。

---



## 6. 除仿照论文外还可以做哪些检验



## 6.1 多模型外部复核

使用不同模型家族独立判断相同测试集，例如 Qwen、DeepSeek 与另一个闭源或开源模型。所有模型：

- 看不到主模型标签；
- 使用独立 prompt；
- temperature 和版本固定；
- 保留完整输出。

报告 pairwise Cohen’s \kappa、Fleiss’ \kappa 或 Krippendorff’s \alpha。它检验的是跨模型可靠性，不是 ground-truth accuracy，不能替代人工金标准。

## 6.2 Test–retest 稳定性

同一模型、同一输入重复运行 3—5 次，检验：

- 单条标签翻转率；
- 重复运行的一致性；
- 公司年度 GI 的相关系数；
- 最终 Greenwashing 标签翻转率；
- 主回归系数是否稳定。

temperature=0 仍可能因服务端更新或并行推理而变化，因此需要保存模型版本和运行日期。

## 6.3 Prompt 稳健性

改变不应影响构念含义的提示词表述：

- 改变规则顺序；
- 交换标签符号 0/1 或 A/B；
- zero-shot 与 few-shot；
- 是否要求理由；
- 是否提供前后句；
- 中文定义的同义改写。

若结果只在某一个 prompt 下成立，说明存在 prompt-induced measurement error。

## 6.4 Contrast-set 与对抗性测试

人工构造最小差异样本：

```text
公司建设并运行废水处理设施。              → 1
公司计划建设废水处理设施。                → 0
地方政府要求公司建设废水处理设施。        → 0
公司高度重视废水处理工作。                → 0
公司完成废水处理设施升级并投入运行。      → 1
```

检验模型是否真正依据主体、行动和实施状态，而不是只看到“废水处理”就判断为 1。该方法依据行为测试和内容效度思想，适合发现关键词捷径，但不能代替自然样本准确率。

## 6.5 Layer A 词表敏感性

构造严格、当前、宽松和极宽松四版候选集合，比较：

- 召回句子数量；
- Layer B 正类和负类数量；
- 公司年度 GI；
- Greenwashing 标签翻转率；
- 主回归结果。

如果词表扩展后公司年度指标和结论稳定，可说明 Layer A 边界词对实证结论影响有限。

## 6.6 聚合口径稳健性

当前 X、Y 统计关键词—句子对，同一句包含多个词会重复计数。至少比较：

- 关键词—句子对口径；
- 去重句子口径；
- 去重语义事件口径；
- 多模型概率加权口径。

如果结论只在重复计数口径下成立，应调查词汇密度和行业术语造成的偏差。

## 6.7 误差传播或标签扰动检验

根据人工测试集估计 Layer B 的假阳性率和假阴性率，在合理区间内对全样本标签进行 Monte Carlo 扰动：

1. 按估计误差率随机翻转标签；
2. 每次重新计算 X、Y、GI 和 Greenwashing；
3. 重复 500—1,000 次；
4. 报告主回归系数、显著性和方向的分布。

该方法回答：即使分类器存在已观测水平的误差，经济学结论是否仍成立。

## 6.8 外部构念效度



### 收敛效度

与理论相关指标比较：环境处罚、环境负面新闻、ESG 争议、排放表现、既有漂绿指标或评级分歧。

### 区分效度

证明指标不是年报长度、环境章节长度、公司规模、普通正面语调或绿色词频的替代物。

### 已知组效度

检验理论上高风险组是否具有更高漂绿值，例如高污染行业、受到环境处罚但积极宣传的企业。

### 预测效度

使用当期漂绿预测未来处罚、负面新闻、评级下调或监管问询。应避免把同一期结果简单解释为因果效度。

---



## 7. 检验方式究竟依据什么确定

研究者通常不是照搬某一论文，而是依据以下逻辑选择检验。

## 7.1 依据任务在测量链条中的功能

- Layer A 是 information retrieval/candidate generation：重点是 Recall、遗漏率和词表敏感性；
- Layer B 是 binary content classification：重点是 Precision、Recall、F1、MCC 和人工一致性；
- 公司年度 GI 是聚合测量：重点是聚合口径、重复计数和误差传播；
- Greenwashing 是理论构念：重点是收敛、区分、已知组和预测效度。

不能用同一套 Accuracy 检验整个链条。

## 7.2 依据假阳性和假阴性的相对成本

Layer A 漏掉一句会永久损失信息，因此优先 Recall；Layer B 若把口号判为行动，会高估 GI，因此 Precision 同样重要。若研究结论对某一类错误更敏感，应将该类指标列为主要指标。

## 7.3 依据类别分布

类别严重不平衡时，Accuracy 可能很高但没有意义，因此需要 MCC、balanced accuracy、macro-F1、PR 曲线或分类别指标。抽样类别比例与总体不同时，要使用抽样权重还原总体表现。

## 7.4 依据编码是否存在主观性

“是否属于具体环保行动”不是完全客观事实，因此先检验人际一致性。若人工之间都无法一致，不能简单把模型与某一名人工的差异全部视作模型错误，需要修订构念和编码手册。

## 7.5 依据论文的核心贡献

- 如果 LLM 只是辅助变量构造，适度人工验证加稳健性可能足够；
- 如果论文声称提出新的 LLM 测量方法，验证必须更严格，包括冻结测试集、多模型、子组误差、误差传播和完整材料；
- 如果目标期刊重视因果识别，还需要证明测量误差不会驱动因果结果，但 IV、PSM 和 placebo 不能替代文本标签验证。



## 7.6 依据可复现与审计要求

应保留：

- 数据抽样脚本和随机种子；
- 未含模型标签的人工标注表；
- coding manual 版本；
- 每位标注者的原始结果；
- 裁决日志；
- 模型名称、版本、日期和参数；
- prompt 原文；
- 原始 API 响应；
- 所有指标计算脚本。

这是好的 LLM 实证论文可信度的重要组成部分。

---



## 8. 对当前项目最推荐的组合



## 8.1 最低复刻方案

1. Layer A：100 个词 × 10 个互不重叠随机样本；
2. Layer B：500 个关键词—句子对；
3. 两名有相关背景的学生独立盲标；
4. 第三人裁决；
5. 报告 Accuracy、F1、MCC；
6. 增加 Cohen’s \kappa 和 95% 置信区间；
7. 冻结测试集，不再根据测试结果修改 prompt。

这比 DeepGreen 披露的方法更完整，但 Layer A 的检验仍受孤立词语歧义限制。

## 8.2 推荐投稿方案

1. Layer A 保留 1,000 个孤立词测试，以便与 DeepGreen 对比；
2. 增加 500 个词—句子上下文化测试；
3. 增加至少 500 个未召回句子的反向遗漏审计；
4. Layer B 分层抽取 800—1,000 条；
5. 两人盲标加第三人裁决，形成冻结人工 gold set；
6. 主模型与至少两个不同模型家族比较；
7. 进行 test–retest、prompt robustness 和 contrast-set 测试；
8. 比较关键词—句子与去重句子两种 GI；
9. 做 Monte Carlo 标签误差传播；
10. 做外部构念效度和未来处罚预测。



## 8.3 如果坚持完全不使用人工

可以采用多模型盲审、重复运行、prompt 扰动、对抗性测试、反向遗漏审计和外部效度，但结论只能表述为：

> 获得了较强的跨模型可靠性、稳定性和构念效度证据。

不能表述为：

> 经验证分类准确率达到某一数值。

当前 `audit_layer_b_pilot.py` 使用同一 Qwen 模型重新审计 1,000 条，适合称为 **同模型自动复核或 test–retest agreement**，不应称为人工检验或独立准确率验证。其 500/500 平衡抽样得到的总体 Accuracy 也不能直接外推到原始类别分布。

---



## 9. 建议预先注册的通过标准

以下是可执行的建议值，不是所有期刊统一规定：


| 项目                          | 建议标准             |
| --------------------------- | ---------------- |
| 人工预测试 Cohen’s \kappa        | ≥ 0.70，最好 ≥ 0.80 |
| Layer A 人工 Recall           | ≥ 0.90           |
| Layer A 未召回文本估计遗漏率          | 尽可能低，并报告置信区间     |
| Layer B macro-F1            | ≥ 0.80           |
| Layer B MCC                 | ≥ 0.70           |
| Layer B 正类 Precision/Recall | 均建议 ≥ 0.80       |
| 模型重复运行标签一致率                 | ≥ 0.90           |
| 多模型多数票争议率                   | 建议 < 10%—15%     |
| 去重口径与原口径 GI 相关系数            | 建议 ≥ 0.90        |
| 替代口径 Greenwashing 标签翻转率     | 建议 < 10%         |
| 误差传播模拟中核心结论同方向比例            | 建议 ≥ 95%         |


不要为了达到阈值反复查看正式测试集并调整 prompt，否则会造成测试集泄漏。若未达标，应报告失败模式、修订开发集、重新冻结一个未使用的新测试集。

---



## 10. 论文中可直接使用的方法说明框架

> 为评估 LLM 文本分类的有效性，本研究在正式分类前建立了书面编码手册，并由两名不知晓模型预测结果的标注者独立编码分层随机样本。标注分歧由第三名裁决者处理，裁决后的标签构成人工测试基准。Layer A 被定义为召回优先的候选检索层，因此除复刻既有研究的孤立词随机验证外，本研究进一步从未被召回的原始句子中实施反向遗漏审计。Layer B 以关键词—完整句子对为评价单位，报告混淆矩阵、Accuracy、Precision、Recall、F1、MCC、balanced accuracy 及其 95% 置信区间，同时报告人工标注者之间的 Cohen’s kappa。正式测试集在提示词和编码规则冻结后仅使用一次。此外，本研究通过跨模型比较、重复运行、提示词扰动、对抗性样本、聚合口径替换及标签误差传播检验分类可靠性和实证结论对测量误差的敏感性。

---



## 11. 方法依据与参考文献



### 直接参照论文

- Xu, C., Liu, J., Li, Z., & Lin, C. (2026). *DeepGreen: Effective LLM-Driven Greenwashing Monitoring System Designed for Empirical Testing—Evidence from China*. Computational Economics. [https://doi.org/10.1007/s10614-026-11328-5](https://doi.org/10.1007/s10614-026-11328-5)

该文提供“双层 LLM + 随机人工样本 + ACC/F1/MCC + 模型比较/消融实验”的直接参照。

### 内容分析与人工一致性

- Krippendorff, K. *Content Analysis: An Introduction to Its Methodology*. 该体系强调编码单位、编码手册、独立编码、可复现性和 Krippendorff’s alpha。
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46. 提供 Cohen’s kappa。



### 分类评价

- Matthews, B. W. (1975). Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochimica et Biophysica Acta*, 405(2), 442–451. MCC 的经典来源。
- 类别不平衡任务通常同时报告混淆矩阵、分类别 Precision/Recall/F1、balanced accuracy 和 MCC，而不是只使用 Accuracy。



### LLM 标注的人工基准思想

- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *Proceedings of the National Academy of Sciences*, 120(30), e2305016120. 该研究通过与独立人工标注基准比较评价 LLM，而不是让同一个 LLM 自证准确。



### 本项目构念与文本测量依据

- Bochkay, K., Brown, S. V., Leone, A. J., & Tucker, J. W. (2023). Textual analysis in accounting: What’s next? *Contemporary Accounting Research*, 40(2), 765–805.
- Seele, P., & Gatti, L. (2017). Greenwashing revisited. *Business Strategy and the Environment*, 26(2), 239–252.
- Yu, E. P. Y., Van Luu, B., & Chen, C. H. (2020). Greenwashing in environmental, social and governance disclosures. *Research in International Business and Finance*, 52, 101192.

这些文献共同支持：分类检验应由任务功能、误差成本、类别分布、人工编码可靠性和构念效度决定，而不是只因前一篇论文抽取了某个数量就照搬。

---



## 12. 最终建议

最稳妥的做法不是在“完全照搬 DeepGreen”和“完全不做人工”之间二选一，而是：

- 用 DeepGreen 的随机人工验证作为可比基线；
- 修正其未充分报告的盲标、裁决、人际一致性和置信区间；
- 把 Layer A 从普通分类问题改为召回问题，增加未召回文本审计；
- 对 Layer B 建立一次小规模但真正独立、冻结的人工 gold set；
- 再用多模型、prompt、重测、对抗样本、聚合口径和误差传播补充可靠性；
- 最后用环境处罚等外部结果检验漂绿构念效度。

如果以较好期刊为目标，建议将人工验证视为建立测量基准，而不是大规模替代 LLM。500—1,000 条高质量双人盲标的边际价值，通常高于再增加一次同模型 AI 复核。

---



## 13. 高影响力 LLM 文本识别研究及其检验方式

本节优先选择发表在 PNAS、PLOS ONE、Computational Linguistics 等高影响力综合或领域期刊，且直接涉及 LLM 文本标注、分类或计算社会科学测量的研究。需要注意：“影响力大”在此主要依据期刊地位、研究的奠基性和后续方法讨论中的常见程度，而不是声称某一固定时间点的精确被引次数。

### 13.1 Gilardi、Alizadeh 与 Kubli（2023）：LLM 文本标注的经典基准研究

**研究任务**

使用 ChatGPT 对推文和新闻文章执行相关性、立场、主题和框架分类，共涉及 4 个数据集、6,183 篇文本，并与受训研究助理和 MTurk 众包标注者比较。

**如何检验 LLM 结果**

1. 由 3 名政治学学生作为受训标注者，根据书面 codebook 独立编码；
2. 仅将两名受训标注者意见一致的文本作为人工 gold standard；
3. ChatGPT、受训标注者和 MTurk 使用尽可能相同的任务说明；
4. 计算 ChatGPT 相对于人工 gold standard 的 Accuracy；
5. 每条文本分别获得两次 ChatGPT 判断，比较重复运行的 intercoder agreement；
6. 比较 temperature = 1 与 temperature = 0.2，以检验随机性设置对稳定性的影响；
7. 使用 2023 年新推文作为补充样本，降低测试文本已进入模型训练数据所造成的记忆污染疑虑；
8. 同时比较准确性、标注一致性、成本和速度。

对本项目的启示

- LLM 的准确率需要相对于独立人工标签计算；
- 人工 gold set 可以只保留人工一致样本，但最好另外报告被排除的人工争议率；
- 同一模型重复运行的 agreement 是可靠性检验，不能替代人工准确率；
- 可以比较不同 temperature，并记录模型版本和运行日期。

**引用与 DOI**

- Gilardi, F., Alizadeh, M., & Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks. *Proceedings of the National Academy of Sciences*, 120(30), e2305016120. [https://doi.org/10.1073/pnas.2305016120](https://doi.org/10.1073/pnas.2305016120)



### 13.2 Törnberg（2023）：政治文本零样本分类及专家比较

**研究任务**

使用 GPT-4 对政治推文进行零样本标注，并将其与专家和众包标注结果比较。任务涉及政治内容中较依赖语境和构念定义的分类，因此与“实质性—象征性绿色披露”较为接近。

**如何检验 LLM 结果**

1. 使用已经过人工标注的政治推文测试集；
2. 将专家共识或既有人工标签作为参照；
3. 比较 GPT-4、专家标注者和众包标注者的分类表现；
4. 报告分类准确性和一致性，而不是只展示若干成功案例；
5. 在 zero-shot 条件下使用清晰的类别定义，检验 LLM 能否直接执行 coding scheme；
6. 分析不同类别上的错误，而不是只依赖总体 Accuracy。

**对本项目的启示**

- Layer B 可让人工和 LLM 使用同一份冻结的 coding manual；
- 应分别报告实质性披露和象征性披露上的表现；
- 对语境依赖任务，专家或受训标注者比同模型自审更适合作为外部标准；
- 清晰的构念定义和边界案例是检验的一部分，而不只是 prompt 工程。

**引用与 DOI**

- Törnberg, P. (2023). ChatGPT-4 outperforms experts and crowd workers in annotating political Twitter messages with zero-shot learning. *PLOS ONE*, 18(9), e0288012. [https://doi.org/10.1371/journal.pone.0288012](https://doi.org/10.1371/journal.pone.0288012)



### 13.3 Rathje 等（2024）：跨语言、跨构念的大规模文本识别验证

**研究任务**

在 15 个数据集、约 47,925 条人工标注文本和 12 种语言上，检验 GPT-3.5、GPT-4 和 GPT-4 Turbo 对情感、离散情绪、冒犯性和道德基础等构念的识别能力。

**如何检验 LLM 结果**

1. 以既有人工标注数据集作为 gold standard；
2. 对分类任务报告 Accuracy、Precision、Recall 和 macro-F1；
3. 对 Likert 连续评分计算 GPT 与人工评分的 Spearman 相关系数；
4. 展示 confusion matrix，检查模型是否系统性混淆特定类别；
5. 与词典法以及各数据集已有的最佳 fine-tuned 模型比较；
6. 比较 GPT-3.5、GPT-4 和 GPT-4 Turbo，检验跨模型版本稳健性；
7. 比较 zero-shot 与 few-shot，发现示例并非总能改善结果；
8. 对相隔一天的两次运行计算 weighted Cohen’s kappa，检验 test–retest reliability；
9. 比较英文 prompt 和目标语言 prompt 的一致性；
10. 单独分析小语种和复杂、稀有类别，揭示高 Accuracy 可能主要来自大量 true negatives。

**对本项目的启示**

- Layer B 应使用 macro-F1 和分类别 Precision/Recall，不能只报告 Accuracy；
- 应展示混淆矩阵，尤其检查模型是否倾向将模糊陈述判成实质性行动；
- 可以跨 Qwen、DeepSeek 等模型版本比较，但版本一致不等于人工准确；
- 对“具体环保行动”这种相对复杂构念，必须检查少数类表现；
- 当前同模型 AI 抽样复核可以仿照其 test–retest 部分，改报 Cohen’s kappa 和标签翻转率，但仍应另建人工 gold set。

**引用与 DOI**

- Rathje, S., Mirea, D.-M., Sucholutsky, I., Marjieh, R., Robertson, C. E., & Van Bavel, J. J. (2024). GPT is an effective tool for multilingual psychological text analysis. *Proceedings of the National Academy of Sciences*, 121(34), e2308950121. [https://doi.org/10.1073/pnas.2308950121](https://doi.org/10.1073/pnas.2308950121)



### 13.4 Ziems 等（2024）：LLM 用于计算社会科学的系统方法框架

**研究性质**

该文不是只评价单一二分类器，而是系统讨论 LLM 在计算社会科学中的标注、测量、生成和模拟用途，是设计完整验证框架的重要方法文献。

**建议的检验思想**

1. 将 LLM 与人工、传统 NLP 或已有基准模型比较；
2. 区分 construct validity、predictive performance 和 reliability；
3. 检查 prompt、模型和采样参数的敏感性；
4. 记录模型版本、prompt、调用参数和时间，保证可复现性；
5. 检查社会群体、语言和文本域之间的异质性及偏差；
6. 警惕训练数据污染、黑箱性和模型更新导致的结果漂移；
7. 不把高平均性能自动解释为每个子组或构念均有效。

**对本项目的启示**

- 你的验证应覆盖“人工准确性—跨模型可靠性—最终漂绿构念效度”三层；
- Layer A、Layer B 和公司年度 Greenwashing 不是同一种任务，必须分别验证；
- 应按行业、年份、句子类型和 confidence 报告异质性表现；
- 要保存当前 Qwen 模型的完整名称、调用日期、prompt 和原始响应。

**引用与 DOI**

- Ziems, C., Held, W., Shaikh, O., Chen, J., Zhang, Z., & Yang, D. (2024). Can large language models transform computational social science? *Computational Linguistics*, 50(1), 237–291. [https://doi.org/10.1162/coli_a_00502](https://doi.org/10.1162/coli_a_00502)



### 13.5 Song 等（2020）：为什么人工 gold standard 本身也必须检验

**研究性质**

该文早于生成式 LLM 的广泛使用，但直接研究“把不完美人工标签作为自动内容分析 gold standard”会造成什么后果，是设计 LLM 验证时非常关键的方法依据。

**核心检验思想**

1. 人工标签不是天然无误，必须报告标注者间一致性；
2. 低质量人工标签会低估或错误估计自动分类器性能；
3. 应使用经过培训的多名标注者、明确 codebook 和分歧处理程序；
4. 不应将单名未经训练的标注者直接当作真值；
5. 自动分类结果与人工分歧时，应区分模型错误、人工错误和构念本身含混。

**对本项目的启示**

- Layer A 的孤立词歧义会直接降低人工 gold set 的质量，因此必须允许 U/无法判断并增加上下文化检验；
- Layer B 应先验证两名人工是否能稳定区分实质性和象征性披露；
- 如果人工 Cohen’s kappa 较低，应先修改 coding manual，而不是立即评价 LLM；
- 应保留双人原始标签和裁决标签，不能只保留最终共识。

**引用与 DOI**

- Song, H., Tolochko, P., Eberl, J.-M., Eisele, O., Greussing, E., Heidenreich, T., Lind, F., Galyga, S., & Boomgaarden, H. G. (2020). In validations we trust? The impact of imperfect human annotations as a gold standard on the quality of validation of automated content analysis. *Political Communication*, 37(4), 550–572. [https://doi.org/10.1080/10584609.2020.1723752](https://doi.org/10.1080/10584609.2020.1723752)



### 13.6 Bail（2024）：生成式 AI 社会科学应用的效度边界

**研究性质**

该文是 PNAS 的方法评论，讨论生成式 AI 能否改善社会科学研究。它不是具体分类 benchmark，但对研究设计、可重复性、偏差和效度的要求具有较高方法影响力。

**建议的检验思想**

1. 不因 LLM 输出看似合理就视为有效测量；
2. 应与人类数据和外部现实结果比较；
3. 关注训练数据不透明、模型版本变化和系统偏差；
4. 把 LLM 当作研究工具而不是不可质疑的社会主体替代物；
5. 通过透明报告和多种验证证据限定结论范围。

**对本项目的启示**

- 环境处罚、负面新闻和 ESG 争议等外部变量是必要的 criterion validity 证据；
- 外部效度不能取代单条标签准确率，但可证明汇总后的漂绿变量有现实含义；
- 论文中应明确区分“LLM 分类可靠”与“漂绿构念有效”。

**引用与 DOI**

- Bail, C. A. (2024). Can generative AI improve social science? *Proceedings of the National Academy of Sciences*, 121(21), e2314021121. [https://doi.org/10.1073/pnas.2314021121](https://doi.org/10.1073/pnas.2314021121)



### 13.7 这些高影响力研究形成的共同验证范式

综合上述文献，具有说服力的 LLM 文本识别检验通常包含：


| 验证环节               | 代表文献                                        | 本项目对应做法                            |
| ------------------ | ------------------------------------------- | ---------------------------------- |
| 独立人工 gold standard | Gilardi et al.; Törnberg; Rathje et al.     | Layer B 双人盲标加第三人裁决                 |
| 人工标签自身可靠性          | Gilardi et al.; Song et al.                 | 报告人工 agreement、Cohen’s kappa 和争议率  |
| 完整分类性能             | Rathje et al.; Törnberg                     | 混淆矩阵、Precision、Recall、macro-F1、MCC |
| test–retest        | Gilardi et al.; Rathje et al.               | 同模型重复运行并报告 kappa 和翻转率              |
| 模型/版本比较            | Rathje et al.; Ziems et al.                 | Qwen、DeepSeek 和第三模型独立盲审            |
| prompt/参数敏感性       | Gilardi et al.; Rathje et al.; Ziems et al. | temperature、zero/few-shot、措辞扰动     |
| 子组与复杂类别分析          | Rathje et al.; Ziems et al.                 | 行业、年份、低置信度和边界表达分层                  |
| 外部构念效度             | Ziems et al.; Bail                          | 环境处罚、负面新闻和 ESG 争议验证                |
| 人工基准缺陷控制           | Song et al.                                 | 训练标注者、冻结手册、保留 U 和裁决日志              |


因此，本项目最有文献依据的方案不是仅复制 DeepGreen 的 `100 × 10` 和 `500` 抽样数字，而是采用：

1. Layer A：孤立词复刻验证 + 上下文化验证 + 未召回句子遗漏审计；
2. Layer B：冻结的人工 gold set + 完整分类指标；
3. 模型层：重复运行 + 跨模型 + prompt 稳健性；
4. 聚合层：去重口径 + 标签误差传播；
5. 构念层：环境处罚等外部效度。

其中，Gilardi 等和 Rathje 等最适合支持“LLM 文本标签如何与人工比较”；Song 等最适合支持“人工 gold standard 为什么也要双人编码和一致性检验”；Ziems 等和 Bail 最适合支持“为什么还需要跨模型、透明性、子组和外部效度”。