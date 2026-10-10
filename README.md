# 审计师行业专长与企业漂绿

本仓库存放论文正文、变量构造、基准回归和各项检验。研究使用中国 A 股 2021–2024 年公司—年数据，共同样本为 15,381 条，考察会计师事务所行业专长是否与更低的企业漂绿相关。漂绿用年报环境表述的实质性比例（DeepGreen / GI）与华证环境得分在行业—年内的偏离来度量。专长度量是事务所在行业—年的客户营业收入份额 `IMS`，以及 `IMS ≥ 20%` 的二元指标 `IMS_D`。

仓库地址：[https://github.com/Charliezk6/Green-washing](https://github.com/Charliezk6/Green-washing)

## 如何查看和下载

在网页上打开某个文件后，可用页面上的下载按钮，或打开 Raw 后另存。需要整库时，在仓库首页选择 **Code → Download ZIP**，或在本地执行：

```text
git clone https://github.com/Charliezk6/Green-washing.git
```

## 建议先看这些文件

- 正文 PDF：[`paper_overleaf/Paper.pdf`](paper_overleaf/Paper.pdf)；LaTeX 源文件：[`paper_overleaf/Paper.tex`](paper_overleaf/Paper.tex)
- 基准回归、描述统计与相关系数：[`Regression/outputs/paper_tables/`](Regression/outputs/paper_tables/)；口径说明：[`Regression/regression README.md`](Regression/regression%20README.md)
- 漂绿变量（Y）：[`Y_variable_reproduction/Y variable README.md`](Y_variable_reproduction/Y%20variable%20README.md)
- 行业专长变量（X）：[`X_reproduction/X variable README.md`](X_reproduction/X%20variable%20README.md)
- DeepGreen 人工检验说明：[`DeepGreen_validation_revision/README_执行说明.md`](DeepGreen_validation_revision/README_执行说明.md)；检验结果：[`DeepGreen_validation_revision/检验结果/`](DeepGreen_validation_revision/检验结果/)

## 现行目录

- [`paper_overleaf/`](paper_overleaf/)：当前正文、图和表
- [`Regression/`](Regression/)：合并面板、基准回归和论文用表
- [`Y_variable_reproduction/`](Y_variable_reproduction/)：漂绿指标的确定性重建
- [`X_reproduction/`](X_reproduction/)：`IMS`、`IPS` 及 20% 切点
- [`DeepGreen_validation_revision/`](DeepGreen_validation_revision/)：Layer A 与 Layer B 的人工复核
- [`threshold/`](threshold/)：20% 切点的分布与选择材料
- [`validation/IMS_construction/`](validation/IMS_construction/)：`IMS` 在全市场口径上的构造核对
- [`validation/panel_version_audit/`](validation/panel_version_audit/)：面板版本对照
- [`validation/P1/`](validation/P1/)：换所、聚类、熵平衡、公司固定效应、GPS-IPW、滞后回归、留一法、机制可验证性、倾向得分匹配
- [`documents/`](documents/)：参考文献 PDF

## 阅读时可跳过

- `archive/`：早期草稿和已替换的检验，不是当前结果。
- `NAU paper/`：课程材料，与本论文正文无关。

## 网页上看不到的内容

以下材料只保留在本地，不会出现在 GitHub 上：

- `appendix/`：CSMAR 等授权原始数据，已写入 [`.gitignore`](.gitignore)
- `Y_variable_reproduction/LLM_classification/` 以及所有 `*.jsonl`：大模型分类的中间文件
