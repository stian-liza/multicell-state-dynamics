# 原型流程代码导读

这份文档整理的是当前仓库里最重要的三条分析链：

1. **细胞级动力学原型**
   从单细胞表达矩阵中提取 `state variables z`、`functional modules m`，再近似 `dm/dt`。
2. **donor-level 分阶段方向网络**
   用固定 signature 节点比较 `A -> B` 和 `B -> A` 哪个方向在 held-out donor 上更有解释力。
3. **局部链条方向性与中介检验**
   对指定链条做 hop-by-hop 双向比较，以及 `A -> B -> C` 的联合增益检验。

这三条链分别对应了你面试里最常讲的三个层次：

- 原型是怎么从真实数据跑起来的
- “方向性网络”到底怎么定义
- 为什么当前结果是**候选机制链**而不是**最终因果证明**

---

## 1. 当前原型里到底有哪两种“模型”

### 1.1 细胞级动力学原型

对应代码：

- [src/multicell_dynamics/real_data.py](../src/multicell_dynamics/real_data.py)
- [src/multicell_dynamics/module_learning.py](../src/multicell_dynamics/module_learning.py)
- [src/multicell_dynamics/trajectory.py](../src/multicell_dynamics/trajectory.py)
- [src/multicell_dynamics/dynamics.py](../src/multicell_dynamics/dynamics.py)
- [scripts/run_scp2154_stromal_to_hepatocyte_coupling.py](../scripts/run_scp2154_stromal_to_hepatocyte_coupling.py)

这一条的核心目标是：

> 先把单细胞表达压缩成模块和低维状态，再用 pseudotime 邻域差分近似模块变化率 `dm/dt`，测试一个粗粒化动力学方程是否能解释这种变化率。

它更接近你研究计划里“系统建模”的原始版本。

### 1.2 donor-level 分阶段方向网络

对应代码：

- [scripts/run_scp2154_fixed_signature_tumor_validation.py](../scripts/run_scp2154_fixed_signature_tumor_validation.py)
- [scripts/run_scp2154_stagewise_bidirectional_network.py](../scripts/run_scp2154_stagewise_bidirectional_network.py)
- [scripts/run_scp2154_chain_directionality.py](../scripts/run_scp2154_chain_directionality.py)

这一条的核心目标是：

> 用 donor 为基本单位，在不同 disease stage 下比较节点之间的双向预测强弱，从而筛出更像 sender/receiver 的候选方向边。

它更接近你现在面试里会讲的“真实数据初步验证”版本。

---

## 2. 数据输入是什么

### 2.1 细胞级动力学原型

典型脚本：

- [scripts/run_gse185477_demo.py](../scripts/run_gse185477_demo.py)
- [scripts/run_scp2154_stromal_to_hepatocyte_coupling.py](../scripts/run_scp2154_stromal_to_hepatocyte_coupling.py)

输入：

- 细胞 × 基因计数矩阵
- 每个细胞的 metadata
- 至少包含 cell id、cell type、donor、phenotype 或 stage 标签

输出：

- 模块活性 `module_activity`
- 模块权重 `module_weights`
- 低维状态 `embedding = z`
- `pseudotime`
- 局部“速度” `local_velocity`
- 拟合后的动力学系数矩阵

### 2.2 donor-level 分阶段方向网络

典型脚本：

- [scripts/run_scp2154_stagewise_bidirectional_network.py](../scripts/run_scp2154_stagewise_bidirectional_network.py)

输入：

- SCP2154 metadata
- 10x mtx/features/barcodes
- stage 定义
- 每个细胞类型的一组固定 signature

输出：

- [results/scp2154_stagewise_network/nodes.tsv](../results/scp2154_stagewise_network/nodes.tsv)
- [results/scp2154_stagewise_network/bidirectional_edges.tsv](../results/scp2154_stagewise_network/bidirectional_edges.tsv)
- [results/scp2154_stagewise_network/stage_chain.tsv](../results/scp2154_stagewise_network/stage_chain.tsv)
- [results/scp2154_stagewise_network/network_summary.tsv](../results/scp2154_stagewise_network/network_summary.tsv)

---

## 3. 第一步：表达矩阵如何预处理

对应代码：

- [src/multicell_dynamics/real_data.py](../src/multicell_dynamics/real_data.py)

### 定义

把原始计数矩阵变成适合后续模块学习或 signature 打分的输入矩阵。

### 输入

- 原始 count matrix `X`

### 输出

- 归一化矩阵 `X_norm`
- HVG 子矩阵 `X_hvg`

### 公式

#### 3.1 library normalization + log1p

在 `log1p_library_normalize()` 中：

```text
x'_cg = log(1 + x_cg / sum_g x_cg * target_sum)
```

含义：

- `x_cg`：细胞 `c` 中基因 `g` 的原始计数
- 先按每个细胞总 reads 做归一化
- 再乘一个固定目标总量
- 最后 `log1p`

#### 3.2 HVG 选择

在 `select_highly_variable_genes()` 中：

```text
dispersion_g = var_g / mean_g
```

按 `dispersion` 排序，取前 `top_k` 个基因。

### 对应函数

- `log1p_library_normalize()`
- `select_highly_variable_genes()`

---

## 4. 第二步：功能模块 `m` 是怎么定义的

对应代码：

- [src/multicell_dynamics/module_learning.py](../src/multicell_dynamics/module_learning.py)

### 定义

模块是把高维基因表达压缩成更少、更稳定、更容易解释的功能程序。

### 输入

- 预处理后的细胞 × 基因矩阵 `X`
- 设定的模块数 `K`

### 输出

- `module_activity = W`，形状 `(n_cells, K)`
- `module_weights = H`，形状 `(n_genes, K)`

### 公式

代码里做的是一个非负低秩分解近似：

```text
X ≈ W H^T
```

其中：

- `W`：每个细胞在每个模块上的活性
- `H`：每个模块由哪些基因构成

更新规则是经典 multiplicative NMF：

```text
H <- H * (W^T X) / (W^T W H)
W <- W * (X H^T) / (W H H^T)
```

### 你面试时怎么讲

可以直接说：

> 我先不用单个基因直接建模，而是把表达矩阵压缩成一组功能模块，每个细胞都有一个模块活性向量，每个模块都有一组代表基因。

---

## 5. 第三步：状态变量 `z` 是怎么定义的

对应代码：

- [src/multicell_dynamics/trajectory.py](../src/multicell_dynamics/trajectory.py)

### 定义

`z` 是细胞在低维状态空间中的位置，是对原始高维表达的 coarse-grained state representation。

### 输入

- HVG 子矩阵 `X_hvg`

### 输出

- `embedding = z`

### 公式

当前实现里直接用了 PCA/SVD 坐标：

```text
X_centered = X - mean(X)
X_centered = U S V^T
z = U[:, 1:d] * S[1:d]
```

对应函数：

- `pca_embedding()`

### 说明

所以现在的 `z` 不是复杂深度模型出来的 latent state，而是第一版最稳定的 PCA 低维状态。

---

## 6. 第四步：pseudotime 和“速度”是怎么定义的

对应代码：

- [src/multicell_dynamics/trajectory.py](../src/multicell_dynamics/trajectory.py)

### 6.1 pseudotime

#### 定义

把细胞按某个低维轴排成一个 `0 到 1` 的连续顺序。

#### 公式

```text
t_i = (z_i - min(z)) / (max(z) - min(z))
```

这里的 `z_i` 是 embedding 某个轴上的值。

然后再根据已知标签把方向定成：

- `healthy -> Tumor`
- 或 `NonInfMac -> InfMac`

对应函数：

- `pseudotime_from_embedding()`
- `orient_pseudotime_by_labels()`

### 6.2 速度 `dm/dt`

#### 定义

当前原型里**没有用真正 RNA velocity**。  
这里的速度是：

> 沿 pseudotime 前向邻居的模块活性差分，近似局部模块变化率。

#### 公式

对每个细胞 `i`：

```text
velocity_i = mean_j ((m_j - m_i) / (t_j - t_i))
```

其中：

- `m_i`：细胞 `i` 的模块活性向量
- `t_i`：细胞 `i` 的 pseudotime
- `j`：在 pseudotime 上更靠前的邻居

这其实是在近似：

```text
dm/dt ≈ Δm / Δt
```

#### 输入

- 模块活性矩阵 `M`
- pseudotime `t`

#### 输出

- `local_velocity`

#### 对应函数

- `local_direction_from_pseudotime()`

### 说明

这一点你面试时一定要讲清：

> 当前原型中的速度是 pseudotime 邻域差分得到的经验变化率，不是严格的 RNA velocity。

---

## 7. 第五步：动力学模型是怎么拟合的

对应代码：

- [src/multicell_dynamics/dynamics.py](../src/multicell_dynamics/dynamics.py)

### 定义

在细胞级动力学原型里，模型尝试解释：

> 当前模块状态、低维状态和外部输入，能否解释模块变化率 `dm/dt`。

### 输入

- `module_activity = M`
- `state_embedding = Z`
- `module_velocity = dM/dt`
- 可选 `external_input = E`
- 可选 `genetics = G`

### 输出

- ridge 回归系数矩阵
- 训练集 `R^2`
- 每个 feature 对每个模块变化率的权重

### 公式

设计矩阵：

```text
X = [M, Z, E, G]
```

目标：

```text
Y = dM/dt
```

ridge 估计：

```text
W = (X^T X + alpha I)^(-1) X^T Y
```

预测：

```text
Y_hat = X W + b
```

### 对应函数

- `fit_population_dynamics()`
- `velocity_r2_score()`
- `velocity_sign_agreement()`

### 输出怎么读

- `train_r2`：模型对局部速度的拟合度
- `top_edges()`：哪些变量最强地预测了哪些模块变化率

---

## 8. 第六步：为什么后来又换成固定 signature 节点

对应代码：

- [scripts/run_scp2154_fixed_signature_tumor_validation.py](../scripts/run_scp2154_fixed_signature_tumor_validation.py)

### 原因

细胞级 NMF 模块适合做原型，但在跨 stage、跨 donor 的真实 atlas 里，模块容易漂移。

所以后面的方向网络用了**固定 signature 节点**，例如：

- `Stromal.caf_contractile`
- `Stromal.ecm_matrix`
- `Hepatocyte.secretory_stress`
- `Hepatocyte.malignant_like`

### signature score 怎么算

对一个 signature `S`：

```text
score_i(S) = mean_{g in S} z_ig
```

这里的 `z_ig` 是基因 `g` 在细胞 `i` 中的标准化表达值。

也就是：

1. 先对每个基因做 z-score
2. 再把 signature 里匹配到的基因求平均

### 输入

- 归一化矩阵
- 基因名
- 一个 signature 字典

### 输出

- 每个 signature 在每个细胞上的分数

### 对应函数

- `signature_scores()`

---

## 9. 第七步：节点是怎么从单细胞变成 donor-level 的

对应代码：

- [scripts/run_scp2154_stagewise_bidirectional_network.py](../scripts/run_scp2154_stagewise_bidirectional_network.py)

### 定义

你后面做方向网络时，基本单位不是单个细胞，而是：

> 某个 donor 在某个 stage 下、某个节点上的平均分数

### 输入

- 细胞级 signature scores
- 每个细胞的 donor id
- stage 标签

### 输出

- donor-level node score tables

### 关键量

对于某个 node（即对应的模块），在某个 stage 下：

```text
delta = mean(condition donors) - mean(healthy donors)
```

这个 `delta` 用来判断节点在该阶段是否明显偏离健康基线。

### 对应函数

- `donor_score_means()`
- `stage_node_tables()`
- `first_altered_stage()`

---

## 10. 第八步：方向性是怎么定义的

对应代码：

- [scripts/run_scp2154_fixed_signature_tumor_validation.py](../scripts/run_scp2154_fixed_signature_tumor_validation.py)
- [scripts/run_scp2154_stagewise_bidirectional_network.py](../scripts/run_scp2154_stagewise_bidirectional_network.py)

### 关键思想

这里的方向性**不是直接宣称因果**，而是比较：

```text
A -> B
vs
B -> A
```

哪个方向在 donor-held-out 预测里更有解释力。

### 单方向检验

对于 donor-level predictor `x` 和 target `y`：

1. leave-one-donor-out
2. 用训练 donor 拟合 ridge
3. 预测 held-out donor
4. 计算模型 `R^2`
5. 再和“只预测均值”的 baseline 比较

### 公式

```text
loo_delta_r2 = loo_r2 - loo_baseline_r2
```

解释：

- `loo_r2`：真实模型在 held-out donor 上的 `R^2`
- `loo_baseline_r2`：只预测训练集均值时的 `R^2`
- `loo_delta_r2 > 0`：说明这个 predictor 比均值基线更有用

### 双向比较

对每一对节点都算：

```text
forward_delta = delta_r2(A -> B)
reverse_delta = delta_r2(B -> A)
```

如果：

```text
forward_delta > reverse_delta + margin
```

则认为 `A -> B` 更强。

### permutation p-value

把 predictor 打乱后重复做很多次，形成 null distribution：

```text
p = (1 + #{null >= observed}) / (N_perm + 1)
```

### 对应函数

- `ridge_fit_predict()`
- `loo_prediction_test()`
- `evaluate_pair()`
- `empirical_p_value()`
- `direction_call()`

---

## 11. 第九步：什么叫“跨阶段复现”

对应代码：

- [scripts/run_scp2154_stagewise_bidirectional_network.py](../scripts/run_scp2154_stagewise_bidirectional_network.py)

### 定义

如果同一条**胜出的方向边**在两个或以上 stage 都被选中，就记为：

```text
stage_consistency = recurrent
```

否则记为：

```text
stage_consistency = stage_limited
```

### 对应函数

- `attach_stage_consistency()`

### 这一步的作用

它不是时间序列因果证明，而是：

> 看这条方向边是不是只在一个阶段偶然出现，还是在多个阶段重复出现。

---

## 12. 第十步：什么叫 strict mode

对应代码：

- [scripts/run_scp2154_stagewise_bidirectional_network.py](../scripts/run_scp2154_stagewise_bidirectional_network.py)

### 定义

`select_stage_chain()` 会从所有双向边里挑出链条边。

如果把这些条件收紧：

- `selected_delta_r2` 更大
- `empirical_p` 更小
- `n_donors` 更高
- 不允许 `target_earlier`
- 甚至要求 `require_recurrent`

那就得到 strict 版网络。

### 含义

宽松版是**候选链发现器**；  
strict 版是**稳定链条过滤器**。

你目前的结果里：

- 宽松版有 `79` 条链
- strict 版变成 `0`

这恰好说明模型会自我收缩，不会把所有相关性都包装成机制。

---

## 13. 第十一步：局部链条和中介检验怎么做

对应代码：

- [scripts/run_scp2154_chain_directionality.py](../scripts/run_scp2154_chain_directionality.py)

### 13.1 hop 检验

对预先指定的链条，每一跳都做双向比较：

例如：

```text
Myeloid.inflammatory_monocyte -> Hepatocyte.secretory_stress
vs
Hepatocyte.secretory_stress -> Myeloid.inflammatory_monocyte
```

输出：

- `forward_delta_r2`
- `reverse_delta_r2`
- `winner`
- `winner_p`

对应函数：

- `evaluate_hops()`

### 13.2 triplet 中介检验

对 `A -> B -> C`，做三个模型：

```text
source-only:    C ~ A
mediator-only:  C ~ B
joint:          C ~ A + B
```

关键指标：

```text
joint_gain_over_source = R2_joint - R2_source_only
```

如果 joint 比 source-only 明显更好，说明中间节点 `B` 提供了额外解释力。

对应函数：

- `evaluate_triplets()`

### 你现在最常讲的例子

```text
Myeloid.inflammatory_monocyte
-> Hepatocyte.secretory_stress
-> Hepatocyte.malignant_like
```

---

## 14. 最终结果文件怎么读

### 14.1 节点表

- [results/scp2154_stagewise_network/nodes.tsv](../results/scp2154_stagewise_network/nodes.tsv)

关键字段：

- `delta`：该 stage 相对 healthy 的节点偏移
- `first_altered_stage`：首次明显变化的阶段

### 14.2 双向边表

- [results/scp2154_stagewise_network/bidirectional_edges.tsv](../results/scp2154_stagewise_network/bidirectional_edges.tsv)

关键字段：

- `forward_loo_delta_r2`
- `reverse_loo_delta_r2`
- `forward_minus_reverse`
- `direction_call`
- `stage_consistency`

### 14.3 保留链条表

- [results/scp2154_stagewise_network/stage_chain.tsv](../results/scp2154_stagewise_network/stage_chain.tsv)

关键字段：

- `selected_delta_r2`
- `selected_empirical_p`
- `selected_coeff`

### 14.4 局部链条方向检验

- [results/scp2154_chain_directionality/chain_hops.tsv](../results/scp2154_chain_directionality/chain_hops.tsv)
- [results/scp2154_chain_directionality/chain_triplets.tsv](../results/scp2154_chain_directionality/chain_triplets.tsv)

---

## 15. 一句话总结整个原型

如果你明天只记一句，就记这个：

> 当前原型先把单细胞表达压缩成状态变量和功能模块，再在 donor 和 stage 两个层面比较跨细胞节点之间的双向预测强弱，从而提出候选方向链；但由于当前数据仍是横断面 atlas，最终结果应被理解为受约束的机制假说，而不是最终因果证明。

---

## 16. 面试最容易混淆的两件事

### 16.1 “速度”不等于图里的箭头

- `velocity_i = mean_j ((m_j - m_i) / (t_j - t_i))`
  是**细胞级动力学原型**里的局部经验速度
- 图里的蓝色箭头和 `forward ΔR²`
  是**donor-level 方向网络**里的方向比较结果

这两个相关，但不是同一个量。

### 16.2 “方向性”不等于严格因果

这里的方向性表示：

> 在当前 donor-level、stage-aware 分析条件下，某个方向比反方向更能解释 held-out 变化。

它是**候选 driver-like relation**，不是已经被证明的真实因果箭头。
