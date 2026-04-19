# Multicell State Dynamics

这是一个把博士研究计划收缩成可运行原型的 GitHub 项目，用来展示“多细胞群体耦合的粗粒化状态动力学建模”这一方向的最小可行闭环。

## 这个仓库在做什么

第一版只解决四件事：

- 从高维细胞特征中提取可解释的功能模块
- 在模块层拟合局部速度动力学
- 引入跨细胞群体的外部耦合输入
- 输出稀疏、可解释的候选驱动边和关键模块

这样做的目的，是先证明“模块级动力学建模”本身有解释价值，再逐步扩展到空间组学、遗传学先验、文献证据和干预设计。

## 为什么说原计划可行

原计划的科学问题是成立的，真正需要控制的是范围。

适合博士前期先做的部分：

- 单疾病场景
- 单个关键状态转变
- 模块级 coarse-grained 动力学
- 局部速度监督而不是长期外推
- 稀疏线性模型作为第一版基线

不适合在第一年同时展开的部分：

- 空间组学、遗传学、文献先验、结构生物学、候选分子设计全部并行
- 直接做很强的因果或干预结论
- 在没有稳定动力学闭环前就进入复杂证据融合

## 目录说明

```text
multicell-state-dynamics/
├── docs/                  研究路线与阶段规划
├── scripts/               可直接运行的 demo 脚本
├── src/multicell_dynamics/ 核心建模代码
├── tests/                 基础测试
├── README.md              英文说明
├── README.zh-CN.md        中文说明
├── LICENSE
└── pyproject.toml
```

## 运行方式

如果你的终端里启用了 Conda `base`，推荐直接使用项目内的 `.venv`：

```bash
cd "/Users/einstian/Documents/New project/multicell-state-dynamics"
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -e . --no-build-isolation
./.venv/bin/python scripts/run_synthetic_demo.py
./.venv/bin/python -m unittest discover -s tests
```

## 下一步怎么扩展

- 换成一个真实疾病数据集
- 加入 RNA velocity 或伪时序方向监督
- 按细胞群体分别估计动力学模型
- 引入空间邻域图构建跨细胞耦合项
- 把文献和数据库证据作为独立排序层接到模型之后

## 当前主线：stromal 影响 hepatocyte 恶性状态重塑

当前最值得推进的真实数据方向不是“用 stromal 判断癌症”，而是：

> CAF-like stromal activation 是否能够预测 hepatocyte/tumor-cell malignant-state remodeling？

设计文档见：

```text
docs/stromal_hepatocyte_malignant_remodeling_plan.zh-CN.md
docs/stagewise_coupling_experiment_design.zh-CN.md
```

当前 SCP2154 原型已经输出：

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
results/scp2154_static_vs_dynamic/comparison_pairs.tsv
results/scp2154_static_vs_dynamic/summary.tsv
results/scp2154_directionality_test/summary.tsv
results/scp2154_directionality_test/direction_details.tsv
results/scp2154_bidirectional_key_tests/bidirectional_pairs.tsv
results/scp2154_bidirectional_key_tests/directed_edges.tsv
results/scp2154_bidirectional_key_tests/summary.tsv
results/scp2154_bidirectional_key_tests/module_reference.tsv
results/scp2154_fixed_signature_tumor_validation/summary.tsv
results/scp2154_fixed_signature_tumor_validation/signature_pair_tests.tsv
results/scp2154_fixed_signature_tumor_validation/bidirectional_signature_pairs.tsv
results/scp2154_fixed_signature_tumor_validation/donor_signature_table.tsv
results/scp2154_coupling_driver_scan/driver_scan.tsv
results/scp2154_coupling_driver_scan/bidirectional_driver_scan.tsv
results/scp2154_coupling_driver_scan/driver_donor_table.tsv
results/scp2154_coupling_driver_scan/summary.tsv
```

初步结果提示 `TAGLN/ACTA2/MYL9/IGFBP7/CALD1/TPM2` 这一 CAF-like stromal 模块与 hepatocyte tumor-up 模块变化方向存在可检验的耦合关系。

多 sender 比较脚本：

```bash
./.venv/bin/python scripts/run_scp2154_sender_comparison.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

Permutation control：

```bash
./.venv/bin/python scripts/run_scp2154_coupling_permutation.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

Leave-one-donor-out 稳健性验证：

```bash
./.venv/bin/python scripts/run_scp2154_leave_one_donor_out.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

早期/背景肝病验证：

```bash
./.venv/bin/python scripts/run_scp2154_pretumor_coupling.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

关键阶段双向测试：

```bash
./.venv/bin/python scripts/run_scp2154_bidirectional_key_tests.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

Tumor 固定 signature 验证：

```bash
./.venv/bin/python scripts/run_scp2154_fixed_signature_tumor_validation.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

Tumor coupling driver scan：

```bash
./.venv/bin/python scripts/run_scp2154_coupling_driver_scan.py \
  --metadata "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/metadata.tsv" \
  --matrix "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/matrix.mtx.gz" \
  --features "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/features.tsv.gz" \
  --barcodes "/Users/einstian/study/SYSU/xiong/pARDH_liver/eclip/profile/barcodes.tsv.gz"
```

## GSE185477 真实数据原型

当前仓库已经补上了一个 `velocity-free` 的真实数据入口，思路是：

- 先用 `C41` 样本
- 先只看 `Macrophage`
- 用 `NonInfMac -> InfMac` 作为方向化伪时序
- 用局部邻域差分近似模块变化方向
- 再拟合模块级稀疏动力学

运行顺序：

```bash
./.venv/bin/python scripts/inspect_gse185477.py
./.venv/bin/python scripts/run_gse185477_demo.py
```

这一步的定位不是严格意义上的 RNA velocity，而是一个适合 Mac 本地先跑通的第一阶段真实数据闭环。

真实数据上的指标通常会明显低于 synthetic demo，这是正常的。这里追求的是“流程成立、信号存在、结果可解释”，而不是像合成数据那样接近完美拟合。
