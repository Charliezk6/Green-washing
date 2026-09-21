针对 IMS 在同一审计事务所—行业—年份组合内具有共同来源的问题，我重新进行了多种 clustering 设定下的稳健性检验。所有设定使用相同的样本、回归方程、控制变量和固定效应，因此 clustering 不会改变回归系数，只会影响标准误、t 值和 p 值。

本文共包含 15,381 个 firm-year observations、4,604 家企业、96 家审计事务所和 47 个行业。具体使用了以下 clustering 方式：

1. Firm clustering：允许同一家企业不同时期的误差项相关；
2. Audit-firm clustering：允许同一家审计事务所服务的不同客户之间存在共同冲击；
3. Audit-firm–industry clustering：允许同一家审计事务所在同一行业内的客户共享相关冲击。这一设定与 IMS 的构造方式最接近；
4. Industry clustering：允许同一行业内企业受到共同的行业冲击；
5. Two-way clustering：同时考虑 Firm + Audit firm 以及 Audit firm + Industry 两个维度的相关性。

其中，audit-firm–industry clustering 尤其重要，因为 IMS 是基于 audit firm × industry × year 构造的。同一家审计事务所在同一行业中的多个客户可能共享相同的行业专长测量来源。正式分析中，我将 clustering 设置在 audit-firm–industry 层面，而不是 audit-firm–industry–year 层面。这样可以允许同一审计事务所—行业组合内的相关性跨年份持续存在，避免将可能存在的 serial correlation 切断。样本中共有 1,205 个 audit-firm–industry clusters。

连续 IMS 的估计结果如下：

- Firm clustering：系数为 −0.4829，标准误为 0.2362，在 5% 水平显著；
- Audit-firm clustering：系数为 −0.4829，标准误为 0.2428，在 5% 水平显著；
- Audit-firm–industry clustering：系数为 −0.4829，标准误为 0.2277，在 5% 水平显著；
- Firm + Audit firm two-way clustering：系数为 −0.4829，标准误为 0.2470，在 10% 水平显著；
- Audit firm + Industry two-way clustering：系数为 −0.4829，标准误为 0.2770，在 10% 水平显著；
- Industry clustering：系数为 −0.4829，标准误为 0.2639，在 10% 水平显著。

因此，IMS 的点估计在所有 clustering 设定下完全不变，始终为 −0.4829，说明其负向关系并不是由某一种标准误设定所产生的。不过，在更保守的 two-way clustering 和 industry clustering 下，统计显著性由 5% 水平下降到约 10% 水平。因此，我会将结果表述为：IMS 与 Greenwashing 之间的负向关系在不同 clustering 设定下方向和大小都非常稳定，但在更保守的推断方法下，统计证据更接近 10% 显著性水平，而不能表述为在所有设定下都达到 5% 显著。

此外，我特别针对 small-cluster problem 进行了进一步处理。Industry clustering 只有 47 个 clusters，Audit-firm clustering 有 96 个 clusters，而且 audit-firm clusters 的规模较不均衡。因此，我使用了 restricted wild cluster bootstrap 作为 small-cluster-robust inference 方法。

Wild cluster bootstrap 的主要作用是：不完全依赖传统 cluster-robust standard error 的大样本近似，而是在 cluster 层面重新赋予随机权重，从而在 cluster 数量较少或 cluster size 不均衡时构造更可靠的有限样本参考分布。该方法仍然保留了 cluster 内部观测值可能相关这一核心假设，因此特别适合检验 47 个行业 clusters 下的推断是否稳健。

具体设置如下：

- Restricted wild cluster bootstrap；
- 原假设为 H_0:\beta_{IMS}=0；
- 使用 Webb six-point weights；
- 999 次 bootstrap replications；
- 双侧检验；
- 使用 plus-one adjusted p-value；
- 同时报告 unrestricted bootstrap percentile 95% confidence interval。

Wild cluster bootstrap 的结果为：

- Industry clustering（47 个 clusters）：IMS 系数为 −0.4829，传统 sandwich p-value 为 0.0672，wild cluster bootstrap p-value 为 0.0730，95% confidence interval 为 [-0.9567,-0.0078]；
- Audit-firm clustering（96 个 clusters）：IMS 系数为 −0.4829，传统 sandwich p-value 为 0.0467，但 wild cluster bootstrap p-value 为 0.0910，95% confidence interval 为 [-0.9354,-0.0132]。

因此，在只有 47 个行业 clusters 的 small-cluster-robust inference 下，IMS 的置信区间仍然不包含 0，负向关系仍得到约 10% 水平的统计支持。对于 audit-firm clustering，wild cluster bootstrap 将传统方法下接近 5% 的 p-value 调整为 0.0910，说明在更保守的有限样本推断下，证据更接近 10% 水平，而不是 5% 水平。

作为补充，我也检验了二元处理变量 IMS_D。其系数始终为 −0.1434，但在 audit-firm clustering、two-way clustering 和 industry clustering 下显著性较弱；wild cluster bootstrap p-value 分别为 0.1330（industry clustering）和 0.1730（audit-firm clustering），其 95% 置信区间均包含 0。因此，连续 IMS 的证据明显强于二元 IMS_D，我会将连续 IMS 作为主要处理变量，将 IMS_D 作为补充稳健性检验。

总体而言，重新考虑 treatment structure 后，IMS 的负向系数在 audit-firm、audit-firm–industry、two-way clustering 以及 47-industry clustering 下均保持稳定。small-cluster wild bootstrap 进一步表明，该负向关系仍具有统计支持，但更加准确和谨慎的表述应是：其统计显著性主要处于约 10% 水平，而不是在所有推断设定下都达到 5% 水平。