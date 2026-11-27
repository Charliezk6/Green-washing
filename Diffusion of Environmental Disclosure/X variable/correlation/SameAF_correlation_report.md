# `correlation` 说明

本目录检验同年度、同行业无序公司对上的 `SameAF` 与九条竞争性扩散通道是否重合。样本来自 `../data/pairs/company_pair_x_variables.parquet`，共 2,051,610 对，其中同所 153,692 对、不同所 1,897,918 对。生成脚本是 `../scripts/build_channel_correlation.py`。

## 目录文件

- `company_pair_channels.parquet`：2,051,610 个公司对的标识列、`SameAF` 和九个通道 0/1 变量。
- `corr_1_pairwise_matrix.csv`：`SameAF` 与九个通道两两之间的 Phi 系数矩阵。两个 0/1 变量的 Phi 等于 Pearson 相关系数。
- `corr_2_channel_overlap_crosstab.csv`：同所组与不同所组在每条通道上取 1 的对数和比例。
- `corr_3_clustered_t_and_vif.csv`：每条通道对 `SameAF` 回归的系数、按 `证券代码_i` 与 `证券代码_j` 双向聚类的标准误、t 值、p 值，以及九条通道与 `SameAF` 同时进入时的 VIF。回归与 VIF 都先去掉 `时间 × ExpertiseIndustryCode` 均值。

## 通道口径

九个变量都是公司对上的 0/1。

| 变量 | 含义 | 数据表与规则 |
| --- | --- | --- |
| `SameGroup` | 同一企业集团 | `HLD_Contrshr.csv` 年末记录。实际控制人 `S0701b` 相同且非空，去掉含“国资委”或“国有资产监督管理委员会”的名称 |
| `SameBlockholder` | 共同大股东 | `HLD_Shareholders.csv` 年末记录。至少共享一名 `S0304a >= 5` 的股东，去掉香港中央结算有限公司和 `HKSCC` |
| `SameProvince` | 注册地同省 | `STK_LISTEDCOINFOANL.csv`，`PROVINCECODE` 相同 |
| `SameCity` | 注册地同城 | `STK_LISTEDCOINFOANL.csv`，`CITYCODE` 相同 |
| `SameOfficeZip4` | 办公地同邮区 | `STK_LISTEDCOINFOANL.csv`，办公地址邮编 `Zipcode` 按逗号拆开后，任一对前四位相同 |
| `BoardInterlock` | 共享董事或高管 | `CG_Director.csv`，共享至少一名职务代码以 `P351` 或 `P353` 开头的 `PersonID`，不含监事 `P352` |
| `SameIndepDir` | 共享独立董事 | `CG_Director.csv`，同一 `PersonID` 至少在一方任独立董事 `P35106`，并在另一方任董事或高管 |
| `SupplyChainLink` | 上市前五大客户或供应商 | `SC_TopFiveSaleInfo.csv` 与 `SC_TopFivePurchaseInfo.csv`。`IsListed = Y` 且披露方与 `BusinessSymbol` 恰好为配对双方，方向不限 |
| `BothInquired` | 双方近年均被年报问询 | `CMS_InquiriesSE.csv` 中 `InquiryType = 1`。两家在当年或上一年都收到过年报问询函 |

## 检验结果

| 通道 | Phi | 同所组比例 | 不同所组比例 | 聚类系数 | t值 | VIF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `SameProvince` 注册地同省 | 0.0643 | 19.77% | 11.74% | 0.0771 | 14.60 | 1.59 |
| `SameCity` 注册地同城 | 0.0331 | 7.45% | 4.73% | 0.0269 | 9.75 | 2.08 |
| `SameGroup` 同一企业集团 | 0.0297 | 0.28% | 0.03% | 0.0025 | 4.49 | 1.06 |
| `SameOfficeZip4` 办公地邮编前四位相同 | 0.0220 | 3.86% | 2.53% | 0.0135 | 8.00 | 1.51 |
| `BoardInterlock` 共享董事或高管 | 0.0154 | 0.52% | 0.23% | 0.0030 | 7.32 | 3.96 |
| `SameBlockholder` 共同大股东 | 0.0126 | 0.20% | 0.07% | 0.0014 | 4.40 | 1.11 |
| `SameIndepDir` 共享独立董事 | 0.0078 | 0.30% | 0.17% | 0.0013 | 4.72 | 3.84 |
| `SupplyChainLink` 上市前五大客户或供应商 | 0.0009 | 0.01% | 0.01% | 0.0000 | 0.81 | 1.00 |
| `BothInquired` 双方近年均收到年报问询 | -0.0083 | 0.96% | 1.31% | -0.0035 | -5.28 | 1.00 |

九条通道一起放进固定行业×年度的回归时，`SameAF` 的 VIF 为 1.01。`BoardInterlock` 与 `SameIndepDir` 的 Phi 为 0.85，两者高度重合。`SupplyChainLink` 只覆盖前五名里披露为上市公司且给出代码的对手方，全样本共 211 对，其中同所 21 对。

## 复现

在项目根目录运行：

```powershell
python "Diffusion of Environmental Disclosure/X variable/scripts/build_channel_correlation.py"
```
