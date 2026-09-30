# TLR8 增强案例：模型看到的不是一个 motif，而是一段固定化学背景中的序列语境

## 先说结论

对 sequence `GTCCGTACCTCCACCCACCG`，模型给出的不是“发现一个危险 motif”的结论，而是：**在 Alharbi 2020 的固定 5–10–5 2′-O-methyl／DNA gap／全 PS assay 背景中，这条偏 C-rich、T 含量较低且没有 TT 的序列更接近中低度 TLR8 potentiation，而不是强增强序列。**实测相对 reporter response 为 1.70，15 个完全未训练过该 row、HELM 或 canonical sequence 的 outer-held-out 成员平均预测 1.271，低估 0.429。

## 这批数据从哪里来，能回答什么

训练标签来自 Alharbi et al. 2020 的 192 条 2′OMe gapmer 与 sequence mutant：100 nM ASO、HEK293-TLR8 NF-κB reporter、R848 条件归一为 1.0；数值越高表示 potentiation 越强。192 条分子的 modification pattern 全部是 `OOOOODDDDDDDDDDOOOOO`，所以这个 endpoint 能学习同一 chemistry architecture 内的 sequence 与位置语境，**不能**从自身数据识别“把 OMe 换成 LNA/MOE 会怎样”。论文原文将主要现象限定为 “in a sequence-dependent manner”；这也是这里最可靠的解释层级。

模型使用的 1,918 个 descriptor 也不是从一篇论文整表复制而来。Raw positional token 的表示来源是 HELM 与 OligoGym；composition/k-mer 的基线设计来自 OligoGym；wing/gap 位置与 sequence×chemistry interaction 则由项目代码自行构造，Papargyri、Bhamra 与 Alharbi 的实验设计只提供概念动机。换言之，论文说明“为什么值得描述这个分子事实”，项目实现决定“怎样把它变成具体数值列”；Supplement 的逐 feature provenance database 为每列保存推导代码、来源、locator 和这条边界。

## Endpoint 呈现的趋势

在完整 192-row source endpoint 中，T fraction 与实测 response 的描述性 Spearman 为 0.468，TT count 为 0.420；G fraction 为 -0.466，GG count 为 -0.429。这些是训练来源中的总体关联，不是 held-out 性能或因果效应。查询序列 T fraction 只有 0.15、TT=0，却有 C fraction 0.55；它有 2 个 TCC，但 TCC count 自身与 response 的相关只有 -0.104（未校正 P=0.152）。因此故事不是“TCC 越多越高”，而是 T-richness、端点位置、gap/wing 上下文与相邻碱基共同形成的趋势。

## 模型怎样把这些线索合在一起

三个 outer partition 的成员基线平均为 2.221。相对该基线，motif/context block 合计贡献 -0.514，Registry 的逐位 token block 为 -0.300，sequence×chemistry/position block 为 -0.136，三者共同把输出拉到 1.271；最大 TreeSHAP 加和残差为 5.93e-07。这里的 chemistry block 主要是“某个碱基位于固定 OMe wing 或 DNA gap”的交互编码，不是跨 chemistry 的比较。

消融给出了更直观的功能依赖：仅删除 motif block 并在相同 outer-train 重拟合，案例预测升到 1.695（Δ=+0.425），几乎回到实测 1.70；删除 Registry 后为 1.325，删除 chemistry-interaction block 后为 1.358。这说明本例的低估主要由整组 sequence-context 表示压低，而不是某一个 feature 出错。Endpoint 层面也一致：完整模型 Spearman 0.709；删除 motif 或 Registry 后分别为 0.643 和 0.640。但删除 chemistry block 后反而为 0.726，略高于完整模型，说明固定 chemistry 数据中这组交互没有稳定的独立增益，不能把它包装成确定机制。

## 两个实测 analogue 帮助读懂这条 sequence

最近 analogue `GGTCCATACCCAAGGCATCC` 与查询的 sequence identity 为 0.60，chemistry 完全相同；二者都只有 15% T、没有 TT，并各含两个 TCC，analogue 实测 1.00。信息性 contrast `TCCCATCCCTTCTGCTGCCA` 也有两个 TCC，却有 30% T、一个 TT 和两个 TG，实测 3.95。三者把关键 insight 说得很清楚：**TCC 数量相同并不产生相同 response；更宽的 T-rich 与位置语境更能区分这个小集合中的低、中、高读数。**近邻只帮助判断适用域和形成实验对照，不参与预测，也不证明共享机制。

## 下一步怎样证伪

最直接的实验不是盲目删除最高 SHAP feature，而是在相同 5–10–5 OMe/DNA/PS 架构下构建 composition-matched pairs：保持总 T/C/G/A 与 TCC 数不变，把 TT/T-rich 片段在 wing 与 gap 间 reciprocal relocation；再做 TCC-disrupting、总组成不变的配对。联合测 TLR8 reporter、receptor-negative reporter、uptake 与 viability。只有位置化 sequence signal 可重复改变 TLR8 reporter、且一般细胞效应不随之同向变化时，才支持候选 TLR8-context 假说。LNA/MOE 替换必须作为独立的跨 chemistry 实验，不能由本 endpoint 模型外推。

## 证据边界

这是模型依赖与可证伪假说，不是生物因果证明；R848 reporter association 不等于临床 immunotoxicity。Registry block 保存可复用的逐位 HELM token；只有整组位置 token 的组合才接近完整输入身份，单个 token 既不是精确分子身份，也不是机制。TreeSHAP、family deletion、数据相关和 analogue similarity 是四类不同证据，不能互相替代。
