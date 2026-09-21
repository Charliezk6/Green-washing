针对 IMS 在同一审计事务所—行业—年份组合内具有共同来源的问题，我具体使用了以下 clustering 方式：

1. Firm clustering：允许同一家企业不同时期的误差项相关；
2. Audit-firm clustering：允许同一家审计事务所服务的不同客户之间存在共同冲击；
3. Audit-firm–industry clustering：允许同一家审计事务所在同一行业内的客户共享相关冲击。这一设定与 IMS 的构造方式最接近；
4. Industry clustering：允许同一行业内企业受到共同的行业冲击；
5. Two-way clustering：同时考虑 Firm + Audit firm 以及 Audit firm + Industry 两个维度的相关性。

连续 IMS 的估计结果如下：

- Firm clustering：系数为 −0.4829，标准误为 0.2362，在 5% 水平显著；
- Audit-firm clustering：系数为 −0.4829，标准误为 0.2428，在 5% 水平显著；
- Audit-firm–industry clustering：系数为 −0.4829，标准误为 0.2277，在 5% 水平显著；
- Firm + Audit firm two-way clustering：系数为 −0.4829，标准误为 0.2470，在 10% 水平显著；
- Audit firm + Industry two-way clustering：系数为 −0.4829，标准误为 0.2770，在 10% 水平显著；
- Industry clustering：系数为 −0.4829，标准误为 0.2639，在 10% 水平显著。

IMS 的点估计在所有 clustering 设定下完全不变，始终为 −0.4829。不过，在 two-way clustering 和 industry clustering 下，统计显著性由 5% 水平下降到约 10% 水平。我认为结果大致可以说成 IMS 与 Greenwashing 之间的负向关系在不同 clustering 设定下方向和大小都非常稳定，但在更保守的推断方法下，统计证据更接近 10% 显著性水平。

此外，我特别针对 small-cluster problem 进行了进一步处理。Industry clustering 只有 47 个 clusters，Audit-firm clustering 有 96 个 clusters，而且 audit-firm clusters 的规模较不均衡。因此，我使用了 restricted wild cluster bootstrap 作为 small-cluster-robust inference 方法。

> ```
> Wild cluster bootstrap 的主要作用是：不完全依赖传统 cluster-robust standard error 的大样本近似，而是在 cluster 层面重新赋予随机权重，从而在 cluster 数量较少或 cluster size 不均衡时构造更可靠的有限样本参考分布。该方法仍然保留了 cluster 内部观测值可能相关这一核心假设，因此适合检验 47 个行业 clusters 下的推断是否稳健。
> ```

> 具体设置如下：
>
> - Restricted wild cluster bootstrap；
> - 原假设为 H_0:\beta_{IMS}=0；
> - 使用 Webb six-point weights；
> - 999 次 bootstrap replications；
> - 双侧检验；
> - 使用 plus-one adjusted p-value；
> - 同时报告 unrestricted bootstrap percentile 95% confidence interval。

Wild cluster bootstrap 的结果为：

- Industry clustering（47 个 clusters）：IMS 系数为 −0.4829，传统 sandwich p-value 为 0.0672，wild cluster bootstrap p-value 为 0.0730，95% confidence interval 为 [-0.9567,-0.0078]；
- Audit-firm clustering（96 个 clusters）：IMS 系数为 −0.4829，传统 sandwich p-value 为 0.0467，但 wild cluster bootstrap p-value 为 0.0910，95% confidence interval 为 [-0.9354,-0.0132]。

因此，small-cluster 下的证据更接近 10% 水平。

作为补充，我也检验了二元处理变量 IMS_D。其系数始终为 −0.1434，但在 audit-firm clustering、two-way clustering 和 industry clustering 下显著性较弱；wild cluster bootstrap p-value 分别为 0.1330（industry clustering）和 0.1730（audit-firm clustering），其 95% 置信区间均包含 0。连续 IMS 的证据明显强于二元 IMS_D，我打算在正文将连续 IMS 作为主要处理变量。