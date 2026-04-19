# Stromal-Driven Hepatocyte Malignant-State Remodeling

## 一句话主线

本课题不把目标放在“判断是不是肝癌”，而是问一个更机制化的问题：

> 肿瘤相关 stromal/CAF 激活程序是否能够预测、解释并优先指向 hepatocyte/tumor-cell 恶性状态模块的重塑？

当前最值得推进的候选轴是：

```text
CAF-like stromal myofibroblast program
  -> hepatocyte malignant/metabolic/secretory state remodeling
```

在 SCP2154 当前原型里，最明确的 stromal tumor-up 模块是 `stromal_m1`：

```text
TAGLN, ACTA2, MYL9, IGFBP7, CALD1, TPM2, ADIRF, VIM, SPARCL1
```

它不像普通分类噪声，更像 activated fibroblast / myofibroblast / CAF-like contractile ECM 程序。这个程序和 hepatocyte 的 tumor-up 模块存在 held-out donor 上的预测增益，因此值得作为第一条主线。

## 为什么不只做分类

Stromal 分类 Tumor 很强，但分类本身不能回答“推动了什么”。如果只报告 stromal 能把 Tumor 和 healthy 分开，审稿人会追问：

- 这是 tumor microenvironment 的结果，还是 driver-like signal？
- 这个 stromal 程序对应哪个 hepatocyte/tumor-cell 状态？
- 信号是否只是 donor、batch、study composition 或 cell-type proportion？
- 是否能在 held-out donor 和外部数据集中复现？

所以本课题应该把分类作为入口，把重点放在模块耦合和验证上。

## 核心假设

阶段性重排版本的实验设计见：

```text
docs/stagewise_coupling_experiment_design.zh-CN.md
```

### Hypothesis 1

Tumor-associated stromal cells contain a reproducible CAF/myofibroblast module marked by contractile and ECM-remodeling genes.

可观察指标：

- `ACTA2/TAGLN/MYL9/CALD1/TPM2/IGFBP7` 模块在 Tumor stromal 中升高。
- 该模块在 donor-held-out 设置下仍然能区分 Tumor stromal。
- 该模块在外部 HCC 数据集中复现。

### Hypothesis 2

CAF/myofibroblast activation predicts hepatocyte/tumor-cell state remodeling beyond hepatocyte-intrinsic state alone.

可观察指标：

- 用 hepatocyte 自身模块和状态嵌入预测 `dM_hep/dt` 作为 baseline。
- 加入 stromal donor/phenotype-level 模块输入后，held-out donor 上 `R2` 或 sign agreement 上升。
- 最强关系集中在 hepatocyte tumor-up 模块，而不是随机模块。

### Hypothesis 3

The stromal-to-hepatocyte coupling is cell-type specific and module specific.

可观察指标：

- Stromal/CAF 模块优于 B cell、T/NK、随机 sender 或 permutation control。
- 不是所有 stromal 模块都有效，最强信号集中在 CAF-like myofibroblast module。
- 不是所有 hepatocyte 模块都被解释，增益集中在 malignant/metabolic/secretory/inflammatory remodeling modules。

## 当前原型结果

当前脚本：

```bash
./.venv/bin/python scripts/run_scp2154_stromal_to_hepatocyte_coupling.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

当前输出：

```text
results/scp2154_stromal_to_hepatocyte/module_reference.tsv
results/scp2154_stromal_to_hepatocyte/coupling_summary.tsv
results/scp2154_stromal_to_hepatocyte/coupling_pairs.tsv
results/scp2154_sender_comparison/sender_summary.tsv
results/scp2154_sender_comparison/coupling_pairs.tsv
results/scp2154_coupling_permutation/observed_pair.tsv
results/scp2154_coupling_permutation/null_distribution.tsv
results/scp2154_leave_one_donor_out/fold_results.tsv
results/scp2154_leave_one_donor_out/summary.tsv
results/scp2154_pretumor_coupling/condition_summary.tsv
results/scp2154_pretumor_coupling/coupling_pairs.tsv
```

关键观察：

- `stromal_m1` 是 Tumor-up stromal 模块，标志基因为 `TAGLN, ACTA2, MYL9, IGFBP7, CALD1, TPM2, VIM`。
- 加入 selected stromal module 后，整体 held-out donor `R2` 从 `0.1989` 到 `0.2109`。
- sign agreement 从 `0.7093` 到 `0.7194`。
- 最值得关注的候选对是 `stromal_m1 -> hep_m0` 和 `stromal_m1 -> hep_m3`。
- 多 sender 比较中，Stromal 排名第一，最佳单模块增益 `delta_r2=+0.0728`，高于 TNKcell、Bcell、Myeloid 和 Endothelial。
- 多 sender 比较里最强 stromal 模块是 ECM/CAF-like 程序：`TIMP1, SPARC, COL1A1, COL1A2, MGP, DCN, BGN, COL3A1, TAGLN, FN1`。
- Stromal permutation control 支持这条边不是随机打乱即可复现：`selected_pair_empirical_p=0.0099`，更严格的 `max_tumor_up_empirical_p=0.0099`。
- Leave-one-donor-out 显示该边不是单个 donor 独撑：23 个 fold 中 15 个 `delta_r2` 为正，所有 23 个 fold 的 coupling coefficient 都为正；测试细胞数不少于 60 的 fold 中 12/16 为正。
- 早期/背景肝病验证中，SCP2154 没有 HBV 标签；可测的是 steatosis、NAFLD/NASH、alcoholic cirrhosis/cirrhotic。`health:cirrhotic` 中出现正向候选关系：ECM/CAF-like stromal module -> hepatocyte condition-up secretory module，`delta_r2=+0.0246`。`low_steatosis` 有增益但方向不是清晰的 condition-up 关系，`indication:NAFLD` 增益接近 0，`indication:alcohol` 有探索性正向信号但 hepatocyte 数较少。

这些结果不能证明因果，但足以说明：这个方向比单纯分类更值得推进。

## 模型设计

### 第一阶段：可解释统计模型

主模型保持简单、可解释：

```text
dM_hep/dt = A * M_hep + B * Z_hep + C * M_sender
```

其中：

- `M_hep`：hepatocyte 模块活性。
- `Z_hep`：hepatocyte 低维状态嵌入。
- `M_sender`：sender 细胞群的 donor/phenotype-level 模块活性。
- `dM_hep/dt`：由 `healthy -> Tumor` 方向化伪时序和局部邻域差分得到的模块变化方向。
- `C`：最关键的 sender-to-hepatocyte coupling coefficient。

这不是深度学习，而是模块分解 + 局部动力学 + 正则化线性耦合模型。它适合当前阶段，因为 donor 数比 cell 数更限制结论强度，解释性比复杂模型更重要。

### 第二阶段：负对照和稳健性

需要加入以下验证：

- donor label permutation：打乱 stromal donor/phenotype 映射，确认增益消失。
- random sender module：随机 stromal 模块或随机噪声输入。
- alternative sender cell types：Myeloid、Endothelial、T/NK、B。
- alternative receiver modules：只看 hepatocyte malignant signature 模块，而不是所有模块一起看。
- leave-one-donor-out CV：报告每个 donor fold 的 coupling 稳定性。

### 第三阶段：外部验证

在另一个 HCC scRNA-seq 数据集上复现：

- CAF-like stromal module 是否仍然 Tumor-up。
- hepatocyte/tumor-cell malignant modules 是否相似。
- `CAF module -> hepatocyte malignant module` 的 donor/sample-level 关系是否复现。

如果外部验证成立，课题从“有趣原型”变成“可以写作的机制线索”。

## 模块参考解释

### Stromal m1：CAF-like myofibroblast / contractile ECM

代表基因：

```text
TAGLN, ACTA2, MYL9, IGFBP7, CALD1, TPM2, VIM, SPARCL1
```

可能含义：

- 肌成纤维样激活。
- 细胞收缩、基质张力、ECM remodeling。
- 可能关联 TGF-beta、integrin、mechanotransduction、fibrosis-like wound healing。

这是当前最适合作为 sender driver candidate 的模块。

### Hepatocyte m0：mature metabolic / transport remodeling

代表基因：

```text
CYP3A5, MLXIPL, C3, ALB, APOB, SLC22A7, TF, FTCD, CYP2A7
```

可能含义：

- 肝细胞代谢和转运状态改变。
- 可能反映 tumor/healthy 之间的代谢重编程或残余 mature hepatocyte program。
- 当前与 `stromal_m1` 的 coupling 增益最明显。

### Hepatocyte m1：acute phase inflammatory secretory

代表基因：

```text
SAA1, SAA2, HP, SERPINA1, ORM1, FGA, FGB, MT2A
```

可能含义：

- 急性期反应、炎症分泌、肝损伤反应。
- 当前在 Tumor 中下降，且 `stromal_m1` 对它是负向候选耦合。
- 可以解释为 tumor microenvironment 下正常炎症分泌程序被重塑或替代。

### Hepatocyte m3：secretory / lipoprotein / acute-phase-like remodeling

代表基因：

```text
ALB, APOA2, APOC3, APOC1, RBP4, TTR, SERPINA1, AMBP, ORM1, APOH, VTN
```

可能含义：

- 肝细胞分泌、脂蛋白、血浆蛋白相关程序。
- 当前是 Tumor-up。
- 与 `stromal_m1` 有正向候选耦合，但增益小于 `hep_m0`，需要更多 fold 验证。

## 最小可发表路线

### Aim 1：定义并验证 Tumor-associated stromal/CAF module

问题：

> 肿瘤 stromal 中是否存在稳定的 CAF-like myofibroblast 模块？

方法：

- Stromal-only healthy vs Tumor donor-held-out baseline。
- NMF/module learning 提取 stromal modules。
- 统计 `ACTA2/TAGLN/MYL9` 模块在 Tumor 中的升高。

产出：

- Stromal module heatmap。
- Donor-level module score plot。
- Tumor vs healthy effect size。

### Aim 2：建立 CAF module 到 hepatocyte state remodeling 的耦合模型

问题：

> CAF-like module 是否提高 hepatocyte 模块变化方向预测？

方法：

- 构建 hepatocyte modules。
- 用 healthy -> Tumor 伪时序估计 `dM_hep/dt`。
- 比较 baseline 和 coupled model。
- 输出 `stromal_mX -> hep_mY` 排名。

产出：

- Coupling pair table。
- `delta_r2` barplot。
- Top module gene reference。

### Aim 3：证明这个关系不是 composition/batch artifact

问题：

> 这个关系是否比随机、其他 cell type、donor leakage 更稳？

方法：

- Donor-held-out 和 leave-one-donor-out。
- Permutation control。
- Sender cell type comparison。
- External HCC dataset validation。

产出：

- Sender comparison table。
- Permutation null distribution。
- External validation summary。

## 风险和边界

当前不能声称：

- Stromal 导致 hepatocyte 癌变。
- 某个配体-受体轴已经被证明。
- 这个模型能替代实验验证。

当前可以稳妥声称：

- Tumor stromal 中存在 CAF-like myofibroblast activation module。
- 该模块在 donor-held-out coupling model 中提高了部分 hepatocyte tumor-up 模块变化方向的预测。
- 结果提示 CAF-like stromal activation 与 hepatocyte malignant-state remodeling 存在可检验的关联。

## 下一步代码任务

优先级最高的三个脚本：

1. `run_scp2154_sender_comparison.py`
   比较 Stromal、Myeloid、Endothelial、T/NK、B 对 hepatocyte modules 的预测增益。当前已实现，输出在 `results/scp2154_sender_comparison/`。

2. `run_scp2154_coupling_permutation.py`
   打乱 donor/phenotype-level sender input，构建 `delta_r2` 的 null distribution。当前已实现，输出在 `results/scp2154_coupling_permutation/`。

3. `run_scp2154_leave_one_donor_out.py`
   做 leave-one-donor-out 稳健性，输出每个 fold 的 coupling coefficient 和 `delta_r2`。当前已实现，输出在 `results/scp2154_leave_one_donor_out/`。

如果这三步都支持 `stromal_m1 -> hepatocyte tumor-up modules`，这个课题就真正站起来了。
