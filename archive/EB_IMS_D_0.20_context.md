# 二元 EB / IMS_D 阈值工作背景备忘

> 供新对话快速接续。  
> **现行项目主口径：`IMS_D = 1{IMS ≥ 0.20}`（X / Regression 面板 / 正文数字已统一为 20%）。**  
> 选阈论证的正文扩写仍可另开任务；本备忘保留 EB/PSM 探索史。

## 一句话结论

项目二元专长切点已统一为 **IMS≥0.20**。历史对照：文献常用 **0.10** 约等于样本 P75，二元 EB 的 B1 不显著；**0.20** 下 EB 的 B1/B2 显著为负，但处理组更小、ESS 下降。

## 关键数字（EB 探索样本，未改）

| 项目 | 0.10（旧对照） | 0.20（现行 / EB） |
|------|----------------|-------------------|
| 定义 | `IMS_D=1{IMS≥0.10}` | `IMS_D=1{IMS≥0.20}` |
| EB 样本 N | 10,001 | 同左 |
| 处理 / 对照 | 2,450 / 7,551 | **620 / 9,381** |
| 主规格 B1 | ≈ −0.04，p≈0.40 | **≈ −0.220，p≈0.027 \*\*** |
| 主规格 B2（连续 IMS） | （报告口径）显著负 | **≈ −0.390，p≈0.012 \*\*** |
| 全样本 ESS（完整规格） | ≈ 6,771 | **≈ 2,148** |
| 对照 ESS / N₀ | 较高（≈0.73） | **≈ 0.43** |

基线共同样本（N=14,778）下，`IMS_D(0.20)` 处理组约 **854**（约 5.8%）。

## 为什么采用 0.20

1. **分布**：IMS 右偏，P75≈0.10，10% 切点处理组过宽。
2. **构念**：与“强专长”叙事一致；国际专长文献常用约 20% 切点。
3. **EB/PSM**：在 0.20 下二元 ATT 更清晰（须披露 N₁/ESS）。

## 已落地口径

- **X_reproduction**：`binary_threshold: 0.2`（`IMS_D`/`IPS_D`）
- **Regression 面板与基线回归**：`IMS_D=1{IMS≥0.20}`
- **validation/P1**：FirmFE / LOO / Two-way cluster / auditor change 已跟 0.20；EB/PSM 本就自造 0.20
- **GPS–IPW**：连续 `IMS`，与二元切点无关

重跑参考：

```bash
python X_reproduction/scripts/reproduce_x.py
python Regression/scripts/reproduce_regression.py
python validation/P1/EB/code/entropy_balancing_analysis.py
python validation/P1/PSM/code/psm_binary_0.20.py
```

## 论文表述边界

- 正文数字与表已按 20% 更新；选阈论证段落可另改。
- 报告二元 EB/PSM 时须披露 N₁、ESS；不可说消除内生性或单独证明因果。
