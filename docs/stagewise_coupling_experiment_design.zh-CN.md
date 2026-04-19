# 阶段性 Stromal-Hepatocyte Coupling 实验重设计

## 核心创新命题

不要把文章命题写成：

```text
stromal 影响 hepatocyte
```

这太宽，也不新。

更好的命题是：

```text
在慢性肝病到 HCC 的进展过程中，stromal-hepatocyte 串话不是静态存在，
而是经历了阶段性重排：早期以炎症/分泌扰动为主，肝硬化阶段出现 ECM-rich stromal
对 hepatocyte secretory/stress state 的稳定耦合，肿瘤阶段进一步转向 CAF-like
stromal module 对 hepatocyte tumor-up state 的预测。
```

一句话版本：

```text
创新不在于“有串话”，而在于“串话主轴如何沿病程演化”。
```

## 当前模块事实

### Alcohol / alcoholic cirrhosis exploratory

数据定义：

```text
condition_col = indication
condition = alcohol
receiver = Hepatocyte
sender = Stromal
```

当前最佳边：

```text
Stromal_m1 -> Hepatocyte_m2
delta_r2 = +0.0403
coeff = +56.9538
direction = candidate_supports_condition_up_module
```

Stromal_m1：

```text
SAA1, SAA2, MT1G, SERPINA1, APOC3, HP, MT1X, ORM1, APOA2, APOC1, ALB, FGB
```

解释：

- 这是 stromal compartment 中的 acute-phase / secretory-like signal。
- 它在 alcohol condition 中是 `Condition-down`，因此不能简单解释为 “ECM stromal 上升推动 hepatocyte 上升”。
- 它更像是 stromal 中某类急性期/肝细胞样背景信号下降，同时 hepatocyte secretory module 上升。

Hepatocyte_m2：

```text
ALB, MALAT1, SERPINA1, APOA1, VTN, APOC1, APOE, C3, APOC3, FGB, AMBP, HP
```

解释：

- hepatocyte secretory / plasma-protein / acute-phase-like module。
- 在 alcohol condition 中明显 `Condition-up`。

重要补充：

Alcohol 里也存在 condition-up 的 ECM-like stromal module：

```text
Stromal_m2: TIMP1, DCN, IGFBP7, COL3A1, BGN, COL1A1, COL1A2, MGP, LGALS1, VIM, S100A6, SPARC
```

但这条 ECM module 到 hepatocyte condition-up module 的当前系数是负向，增益也较弱。因此 alcohol 结果只能作为探索性线索，不能当作主证据。

### Cirrhotic / fibrosis-like stage

数据定义：

```text
condition_col = health
condition = cirrhotic
receiver = Hepatocyte
sender = Stromal
```

当前最佳边：

```text
Stromal_m1 -> Hepatocyte_m1
delta_r2 = +0.0246
coeff = +23.6270
direction = candidate_supports_condition_up_module
```

Stromal_m1：

```text
IGFBP7, TIMP1, DCN, VIM, COL1A1, BGN, COL3A1, S100A6, COL1A2, MGP, SPARC, C1R
```

解释：

- ECM-rich / matrix fibroblast / fibrosis-associated stromal module。
- 在 cirrhotic 中强烈 `Condition-up`。
- 这是当前最适合作为“纤维化阶段 stromal sender”的模块。

Hepatocyte_m1：

```text
MALAT1, ALB, APOA1, SERPINA1, APOC1, APOE, VTN, APOA2, AMBP, FGB, RBP4, APOC3
```

解释：

- hepatocyte secretory / lipoprotein / plasma protein / stress-adaptive module。
- 在 cirrhotic 中强烈 `Condition-up`。

另外，cirrhotic 中还有一个 condition-up 的 contractile myofibroblast module：

```text
Stromal_m0: TAGLN, IGFBP7, ACTA2, ADIRF, MYL9, JUNB, TPM2, VIM, FOS, RGS5, SPARCL1, GADD45B
```

但它对 Hepatocyte_m1 的增益较小：

```text
delta_r2 = +0.0033
```

因此 cirrhotic 阶段主线应优先讲 ECM-rich matrix fibroblast，而不是 contractile myofibroblast。

## 重新设计后的实验结构

### Aim 1：建立阶段性模块图谱

问题：

```text
从 healthy -> low_steatosis / NAFLD -> cirrhotic -> Tumor，
stromal 和 hepatocyte 的模块状态如何变化？
```

做法：

- 对每个阶段分别学习 stromal modules 和 hepatocyte modules。
- 同时构建一套跨阶段 reference module score，避免每个阶段的 `m_0/m_1` 名字漂移。
- 把模块归并成少数可解释状态：

```text
Stromal states:
1. inflammatory/secretory-like stromal signal
2. ECM-rich matrix fibroblast
3. contractile myofibroblast / CAF-like
4. immune/ambient-like stromal signal

Hepatocyte states:
1. acute-phase inflammatory
2. secretory/lipoprotein/plasma-protein
3. metabolic-maintenance
4. immune/stress/doublet-like
```

预期产出：

- stage x module heatmap。
- 每个模块的 donor-level score boxplot。
- stage trajectory summary。

### Aim 2：比较各阶段的 stromal-hepatocyte 主导耦合轴

问题：

```text
串话主轴是否随病程发生重排？
```

当前可放入阶段：

```text
healthy -> low_steatosis
healthy -> cirrhotic
healthy -> Tumor
```

探索性阶段：

```text
healthy -> NAFLD
healthy -> alcohol
```

HBV：

```text
SCP2154 里没有 HBV 标签，需要外部 HBV/HCC 数据集验证。
```

每个阶段统一跑：

```text
dM_hep/dt = A * M_hep + B * Z_hep + C * M_stromal
```

输出：

```text
stage
stromal module
hepatocyte module
delta_r2
coefficient
direction
top genes
```

当前初步结果可以组织为：

```text
low_steatosis:
  weak / mixed evidence
  not yet a clean ECM -> hepatocyte condition-up relation

cirrhotic:
  ECM-rich matrix fibroblast -> hepatocyte secretory/stress module
  delta_r2 = +0.0246

Tumor:
  ECM/CAF-like stromal module -> hepatocyte tumor-up secretory/acute-phase module
  delta_r2 = +0.0728
  permutation p ~= 0.0099
```

这就形成了“纤维化阶段出现、肿瘤阶段增强”的主线。

### Aim 3：证明不是 composition artifact

要证明：

```text
不是因为 stromal 多了、hepatocyte 少了、某些 donor 特殊，
而是 stromal state 与 hepatocyte state 的模块级 coupling。
```

需要做：

- donor-held-out。
- leave-one-donor-out。
- permutation control。
- sender comparison。
- cell proportion covariate。
- donor-level module score regression。

当前已经完成：

```text
sender comparison:
  Stromal 排第一，best delta_r2 = +0.0728

permutation:
  selected_pair_empirical_p = 0.0099
  max_tumor_up_empirical_p = 0.0099

leave-one-donor-out:
  23 folds 中 15 folds delta_r2 > 0
  23 folds coupling coefficient 全部为正

static vs dynamic:
  top dynamic pair = Stromal_m0 -> Hepatocyte_m2
  dynamic_delta_r2 = +0.0728
  static_delta_r2 = -1.1813
  static_corr = +0.4616
  说明这条边主要体现在 hepatocyte module velocity / change direction，
  而不是简单的静态模块水平回归。
```

还需要补：

```text
cell proportion covariate
stage-stratified donor-level regression
external dataset validation
```

### Aim 4：把 hepatocyte 端模块重新定义得更尖

当前 hepatocyte module 仍然偏宽，容易稀释故事。

建议把 hepatocyte 端重新整理成固定 signature scores：

```text
secretory-maintenance:
  ALB, APOA1, APOA2, APOC1, APOC3, TTR, RBP4, AMBP

acute-phase inflammatory:
  SAA1, SAA2, HP, ORM1, SERPINA1, CRP, FGA, FGB, FGG

secretory-stress / pre-malignant:
  SERPINA1, IFI27, SPINK1, VTN, C3, APOE, FGB, AMBP

metabolic-maintenance / metabolic-loss:
  CYP3A5, CYP2E1, MLXIPL, SLC22A7, APOB, FTCD

immune/stress contamination control:
  HLA-A, HLA-B, HLA-C, TMSB4X, TMSB10, SRGN, PTPRC
```

这样后面不再只说 `hep_m1/m2`，而是能说：

```text
ECM-rich stromal activation predicts hepatocyte secretory-stress remodeling.
```

这句话会比模块编号强很多。

## 推荐主图设计

### Figure 1：阶段模块图谱

内容：

```text
healthy / low_steatosis / cirrhotic / Tumor
stromal modules + hepatocyte modules
```

目的：

展示 ECM-rich stromal module 不是 HCC 才出现，而是在 cirrhotic 阶段已经上升。

### Figure 2：阶段性 coupling map

内容：

```text
rows = stages
columns = stromal module -> hepatocyte module pairs
color = delta_r2
edge width = coefficient
```

目的：

把创新点压成一张图：串话主轴沿病程重排。

### Figure 3：负对照和稳健性

内容：

```text
sender comparison
permutation null
leave-one-donor-out
cell proportion control
```

目的：

证明不是 composition artifact。

### Figure 4：外部验证

内容：

```text
另一个 HBV/HCC 或 fibrosis/HCC dataset
复现 ECM-rich stromal -> hepatocyte secretory-stress relation
```

目的：

把 SCP2154 里的发现变成可泛化命题。

## 最终文章命题

弱版本：

```text
ECM-rich stromal modules are associated with hepatocyte secretory remodeling in cirrhotic and tumor stages.
```

中等版本：

```text
Stromal-hepatocyte coupling undergoes stage-associated reorganization during chronic liver disease progression,
with ECM-rich stromal modules emerging in cirrhosis and strengthening toward HCC.
```

强版本：

```text
ECM-rich stromal activation is an early microenvironmental event that predicts hepatocyte secretory-stress
state remodeling before and during malignant transition.
```

当前数据最适合先写中等版本。强版本需要外部 HBV/HCC 或 longitudinal fibrosis-HCC 数据集支持。

## 下一步代码任务

1. `run_scp2154_stagewise_coupling_summary.py`
   把 `low_steatosis / cirrhotic / Tumor` 的 module pair 整合成统一 stagewise table。

2. `run_scp2154_signature_scores.py`
   不再只依赖 NMF module 编号，给 hepatocyte 固定打分：secretory-maintenance、acute-phase、secretory-stress、metabolic-loss。

3. `run_scp2154_composition_control.py`
   加入 cell proportion covariates，证明不是细胞比例驱动。

4. `run_scp2154_static_vs_dynamic_comparison.py`
   比较静态模块水平关联和动态模块变化方向预测，证明模型不只是相关性。

5. `run_scp2154_directionality_test.py`
   比较 `Stromal_m0 -> dHepatocyte_m2/dt` 和 `Hepatocyte_m2 -> dStromal_m0/dt`，避免把双向或共同病程信号误写成单向 sender。

6. 外部数据集验证
   优先找有 HBV、fibrosis/cirrhosis、HCC 梯度的数据集。
