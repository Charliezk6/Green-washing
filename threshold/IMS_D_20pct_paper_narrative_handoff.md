# 正文 IMS_D=20% 论证与文献挂靠 — 新对话交接备忘

> 供新对话框快速接续。  
> **本轮任务范围**：只改 `Writting_paper/Paper.tex`（及必要 bibliography）中的**定义句文献挂靠** + **按论证计划补写选阈叙述**。  
> **不要**再改阈值代码、重跑回归、或动换所/GPS 数字（那些已完成）。

---

## 一句话现状

全项目实证口径已统一为 `IMS_D = 1{IMS ≥ 0.20}`（含面板、基线回归、P1 多数模块）。正文**数字**大多已换成 20%；**定义句仍错误地把蔡春 (2007) 挂成 20% 依据**，且**尚未写入**「为何选 20%」的构念/国际文献/分布三句式论证。

---

## 新对话要做的两件事

### A. 修正定义句文献挂靠（事实错误）

当前正文（约 L288）：

```text
Following the 20 percent specialization threshold used in Chinese audit-market research
\cite{cai2007specialization}, we define IMS_D as one when IMS is at least 20 percent ...
```

问题：蔡春、贤成毅 (2007) 实际用的是 **10%**，不能当作 20% 的文献依据。

改法方向（与论证计划一致）：

- 去掉「中国文献常用 20% / 跟随蔡春」的表述；
- 改为：国际专长文献常用约 **20%** 作为强专长指示（Neal & Riley 2004；Dunn & Mayhew 2004 等）；
- **不要**在正文展开「蔡春 10% 作稳健性对照」（用户明确不做该叙事）；
- 连续 `IMS` 的市场份额法仍可保留 `\citep{cai2007specialization,krishnan2003expertise}`（L277），那是连续度量法，不是二元切点。



### B. 按论证计划补写正文叙述

权威提纲：`[c:\Users\ZZfvr\.cursor\plans\ims_20%论证思路_e469ad9d.plan.md](file:///c:/Users/ZZfvr/.cursor/plans/ims_20%论证思路_e469ad9d.plan.md)`

**总叙事**：连续 `IMS` = 专长强度；二元 `IMS_D` = **强行业专长**指示。因机制要求深层行业知识，采用国际文献常用的约 **20%** 市场份额切点。

**三条依据（按序写入 Measures，由强到弱）**：

1. **构念**：机制段（Hammersley / 不一致识别 / CAS 1521 质询）对应 *strong specialist*，不是「略高于最低经验」。
2. **文献**：Neal & Riley (2004, *AJPT*) — 切点应与构念一致；Big 6 时期常用约 20%（如 1/6\times1.2）。Dunn & Mayhew (2004, *RAST*，正文已有 `dunn2004audit`)。语气：约 20% 是国际常用强专长指示；**禁止**写「Neal–Riley 公式证明中国最优切点唯一=20%」。
3. **分布（一句）**：本样本 IMS 右偏、均值≈0.07；切点过宽会把中等专长并入专长组；20% 将处理组收窄为尾部（共同样本约 **5.8%**）。必须 *ex ante*，**禁止**写因回归/EB 显著才选 20%。

**各段落落点**：


| 位置               | 现状                                     | 建议                                                 |
| ---------------- | -------------------------------------- | -------------------------------------------------- |
| Measures L288    | 错误挂蔡春 20%                              | **主战场**：重写 2–4 句（定义 + Neal/Riley + Dunn + 机制半句）    |
| Introduction L62 | 只写连续市场份额                               | 可补半句：+ 份额≥20% 的强专长指示                               |
| Desc L325–326    | 5.8% / 20% 已对；「meaningful variation」略满 | 可弱化为强专长尾部可识别变异                                     |
| Results L353–354 | −0.1449* / 20% / ≈0.145 已对             | 可把 non-specialized 改为 non–strong-specialist 口径（可选） |
| bibliography     | 无 Neal & Riley                         | **需新增** `neal2004auditor`（或等价 key）                 |


---



## 已完成、本轮不要重做


| 项                                                        | 状态                       |
| -------------------------------------------------------- | ------------------------ |
| X `binary_threshold: 0.2`；面板 `IMS_D`/`IPS_D`             | 已改并重跑                    |
| 基线回归 / paper_tables / 描述相关数字                             | 已 20%                    |
| Paper 硬编码：5.8%、−0.033、0.404、−0.1449、附录 −0.1454 等         | 已更新                      |
| 换所 § + `tab:auditor-selection` / `tab:pre-switch-change` | 已按 20% 专家定义更新            |
| EB / PSM                                                 | 流水线内本就 0.20              |
| GPS–IPW                                                  | 连续 IMS，与切点无关             |
| `generate_paper_tables.py`                               | 已对齐用户调过的 IMS / IMS_D 表版式 |


---



## 关键数字（写叙述时可用）

- 共同样本 N = **14,778**；`IMS` 均值 ≈ **0.070**
- `IMS_D`（≥0.20）均值 ≈ **5.8%**（N₁=854）
- 基线全控制：`IMS` ≈ **−0.5431**；`IMS_D` ≈ **−0.1449**
- 旧 10% 切点下专长组约 24.9%（仅作内部对照理解；**正文不写稳健性对照**）

---



## 正文禁止写入

- 「因为 0.20 时回归/EB 才显著所以采用」
- 「10% 放在稳健性/附录」
- 「Neal–Riley 公式给出中国唯一最优切点=20%」
- 「20% 消除内生性」

---



## 建议先读文件

1. 本备忘
2. 论证计划：`C:\Users\ZZfvr\.cursor\plans\ims_20%论证思路_e469ad9d.plan.md`
3. `[Writting_paper/Paper.tex](../Writting_paper/Paper.tex)` — 尤其 L127–140（机制）、L275–288（Measures）、L325–354（描述/结果）
4. 已有 bib：`dunn2004audit`（L738 附近）；待加 Neal & Riley (2004) *Auditing: A Journal of Practice & Theory*

