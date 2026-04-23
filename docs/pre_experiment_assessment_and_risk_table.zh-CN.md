# 预实验评估与课题风险表

这份文档回答两个问题：

1. 目前已经做过的预实验，**到底证明了什么**，还**没有证明什么**
2. 这个课题接下来最主要的风险是什么，对应的对策是什么，如果某一步失败，应该怎么收缩

---

## 一、目前预实验已经证明了哪些问题

### 1. 真实数据上的“模块化表示”是可运行的

对应结果：

- [results/gse185477_demo/module_summary.tsv](../results/gse185477_demo/module_summary.tsv)

代表性现象：

- 不同 macrophage 状态之间能得到明显不同的模块活性偏移
- 模块 top genes 具有一定生物解释性，例如：
  - `m_1` 偏炎症样：`S100A12, PLBD1, IL17RA, FPR2`
  - `m_2` 偏 resident / macrophage identity：`MARCO, VCAM1, CD5L, ITLN1`

说明：

> 这至少证明“把高维表达压成模块”不是纯概念，模块层表示在真实数据上能形成可解释的状态差异。

但还没有证明：

- 模块一定是最优状态表示
- 模块跨数据集一定稳定
- 模块已经足以支撑完整病程建模

---

### 2. donor-aware、stage-aware 的方向网络原型能跑通

对应结果：

- [results/scp2154_stagewise_network/network_summary.tsv](../results/scp2154_stagewise_network/network_summary.tsv)

关键数字：

- `sampled_cells = 20236`
- `nodes_scored = 24`
- `bidirectional_edges = 762`
- `chain_edges = 79`
- 分阶段链条数：
  - `low_steatosis = 33`
  - `cirrhotic = 30`
  - `Tumor = 16`

说明：

> 这证明你的框架已经不仅是“想法”，而是在真实 liver fibrosis / tumor atlas 上，能够输出 stage-wise、bidirectional、module-level 的候选方向网络。

但还没有证明：

- 这些链条都是真实病程主轴
- 这些链条已经跨阶段稳定
- 这些链条已经达到“机制结论”强度

---

### 3. 方向性比较比普通相关更进一步，但目前仍是候选机制层

对应结果：

- [results/scp2154_chain_directionality/chain_summary.tsv](../results/scp2154_chain_directionality/chain_summary.tsv)

最关键的一条局部链：

```text
Myeloid.inflammatory_monocyte
-> Hepatocyte.secretory_stress
-> Hepatocyte.malignant_like
```

关键数字：

- `hop_2`: `Myeloid.inflammatory_monocyte -> Hepatocyte.secretory_stress`
  - `winner_p = 0.0392`
- `hop_3`: `Hepatocyte.secretory_stress -> Hepatocyte.malignant_like`
  - `winner_p = 0.0196`
- `triplet`:
  - `partial_mediation_supported`
  - `joint_gain = 0.6960`
  - `p = 0.0392`

说明：

> 这证明在局部链条层面，你的框架已经能提出比“谁和谁相关”更强一点的命题，即某些跨细胞关系可能通过一个中间状态组织出后续变化。

但还没有证明：

- 这是严格因果链
- 它跨多个阶段都稳定成立
- 它在独立数据集也成立

---

### 4. 框架具有“自我收缩”能力，不会把所有相关性都包装成机制

对应结果：

- [results/scp2154_stagewise_network_strict/network_summary.tsv](../results/scp2154_stagewise_network_strict/network_summary.tsv)

关键数字：

- `bidirectional_edges = 657`
- `chain_edges = 0`

说明：

> 这是一个很重要的正面结果。它说明你的框架在宽松模式下可以提出候选机制链，而在严格模式下会老实收缩，不会强行输出看起来很完整但其实证据不够硬的故事。

这个结果在面试里其实是加分项，因为它说明：

- 模型不是“到处画箭头”
- 你知道证据强弱的边界

---

### 5. 动态原型相对静态基线，目前只有“可行性”而不是强优势

对应结果：

- [results/scp2154_static_vs_dynamic/summary.tsv](../results/scp2154_static_vs_dynamic/summary.tsv)

关键数字：

- `mean_static_delta_r2 = -0.201243`
- `mean_dynamic_delta_r2 = -0.001515`
- `dynamic_only_gain_pairs = 6`
- `dynamic_stronger_than_static_pairs = 1`
- `top_dynamic_pair = Stromal_m0 -> Hepatocyte_m2`

说明：

> 这说明“动态层”不是完全没有信息，但也还没有形成非常强、非常普遍的解释增益。换句话说，目前的动态优势更多是 prototype-level feasibility，而不是已经稳稳压过静态方法。

这恰好告诉你后面应该重点补什么：

- 下一步 short-horizon forecast 要更清楚
- 表示层 `m, z` 可能还需要优化
- 动态目标的定义要比现在更锋利

---

## 二、这些预实验综合起来，已经支撑了什么结论

### 已经能支撑的

1. 这个课题不是空想，真实数据原型已经跑通
2. 模块级表示和 stage-wise directional network 在公开 liver 单细胞数据上是可执行的
3. 框架能提出候选跨细胞方向链，而不只是做静态差异分析
4. 框架在 strict 条件下会收缩，说明你有基本的约束意识

### 还不能支撑的

1. 不能直接说已经证明了“谁驱动了肝癌发生”
2. 不能直接说已经重建了真实病程时间顺序
3. 不能直接说某一条边就是最终因果机制
4. 不能直接说动态模型已经明显优于静态分析

---

## 三、目前最需要改进的地方

### 1. 表示层还需要更稳定

最核心的问题不是方程够不够复杂，而是：

- 细胞状态是否真的适合用当前模块表示
- 模块是否跨 donor / 跨数据集稳定
- hepatocyte 终点定义是否足够干净

### 2. 需要把“state effect”和“abundance effect”分开

目前 donor-level 平均分是有意义的，但仍可能混入：

- broad cell type 内部亚状态比例变化
- sample-level composition difference

所以后面最好加入：

- abundance covariate
- 或者 cell-state composition control

### 3. 需要更强的外部验证

至少要补一类：

- 独立队列复现
- 空间支持
- 公共 perturbation 支持

否则现在更像一个很好的 hypothesis generator。

### 4. 需要把动态目标定义得更锋利

你现在最值得升级的方向不是继续堆层，而是：

- 预测 `m(t + Δt)` 而不是只比较静态方向
- 做 one-step short-horizon forecast
- 看当前微环境输入是否能预测下一步 hepatocyte 模块变化

### 5. 需要把主命题再收缩

现在最容易成功的版本不是：

- “我解释了所有肝病到肝癌演化”

而是：

- “我识别了病程中某一条最有说服力的微环境到 hepatocyte 的候选方向链”

---

## 四、主要风险、对策、失败后怎么收缩

| 风险 | 具体表现 | 对策 | 如果失败，怎么收缩 |
|---|---|---|---|
| 状态表示不稳 | 模块跨数据集漂移，结果难解释 | 固定 signature + data-driven modules 对照；比较不同表示层 | 先不用自由模块，退回固定 signature 节点版本 |
| donor 异质性过强 | 链条在不同 donor 上不稳定 | donor-held-out、提高 donor 门槛、做 recurrent 筛选 | 先只做 donor-level 稳定的局部链，不讲全局网络 |
| 横断面数据无法支撑强动态 claim | 容易被质疑只是 stage difference | 明确只讲 candidate progression structure，不讲真实时间因果 | 把目标收成“stage-aware directional pattern”而不是“dynamics proof” |
| composition artifact | 节点升高可能只是某个亚群比例变化 | 加 abundance covariate；在 broad cell type 内再分层；检查子群比例 | 若难以拆清，先把结论改成“microenvironment remodeling pattern”，不直接说 per-cell activation |
| hepatocyte 终点定义不干净 | `malignant_like` / `secretory_stress` 太宽或有 ambient 污染 | 固定 signature、控制 immune ambient、筛更纯的 hepatocyte 子集 | 若终点不稳，先讲 hepatocyte stress remodeling，不直接讲 malignant transition |
| 动态增益不够强 | 相比静态模型提升有限 | 定义 one-step forecast；优化 `z`、`m` 表示；做更合适 baseline | 如果动态优势始终弱，先把课题收成“directional coupling framework”，不强打 fully dynamic |
| 方向性被质疑成相关性 | 审稿人质疑 shared donor state 或 hidden confounder | 做 `A -> B` vs `B -> A`、permutation、negative control、strict filtering | 若方向性仍不稳，先把结论降为“asymmetric predictive association” |
| 主线太散 | 空间、AI agent、药物设计同时展开导致不聚焦 | 博士前期只保留单细胞 + stage + donor + module + direction | 后续部分全部放成 future work，不在当前主论文主线中展开 |

---

## 五、目前最适合的课题定位

基于已有预实验，我认为目前最稳、也最有希望做成文章的定位是：

> 建立一个 donor-aware、stage-aware 的粗粒化多细胞耦合框架，用于从公开 liver disease 单细胞数据中识别候选的跨细胞方向链，并重点解析哪些微环境模块在病程中更早出现、并与 hepatocyte 的恶性相关状态转变相耦合。

这一定义的好处是：

- 足够具体
- 和你已经做出来的东西一致
- 不会把 claim 说得太满
- 又保留了后续机制深化的空间

---

## 六、一句话总结

目前的预实验已经证明：

> 这个课题的最小闭环是成立的：模块级表示、donor-aware 方向比较和 stage-wise 网络筛选在真实 liver 单细胞数据上可以跑通，并能提出候选机制链。

目前还没有证明：

> 这些链条已经构成跨阶段、跨队列稳定的最终病程机制。

所以下一步最重要的，不是继续把框架做得更大，而是：

> 让表示层更稳、让动态目标更清楚、让局部链条更硬、让最终命题更锋利。
