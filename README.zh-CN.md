# Multicell State Dynamics

这是一个围绕**多细胞状态转变粗粒化建模**的研究原型仓库，目标是从单细胞数据出发，把分析从静态关联推进到更有方向感的模块级比较。

英文版见：[README.md](README.md)

## 仓库现在在做什么

这个仓库当前的核心问题是：

> 能否从横断面单细胞数据中，提取出更接近“状态变化结构”的信息，并据此构建可解释的多细胞方向性原型？

目前的代码主线分为三个层次：

1. **合成数据 demo**  
   验证模块学习和稀疏动力学拟合能够端到端跑通。
2. **小规模真实数据原型（`GSE185477`）**  
   测试伪时间局部方向信号能否支持真实数据上的模块级动力学建模。
3. **分阶段肝病原型（`SCP2154`）**  
   在 donor-aware 的前提下，比较 `模块状态 -> 模块速度` 的双向关系，并筛选更稳定的候选方向边。

这个仓库的目标不是直接证明因果，而是建立一个可信的建模闭环，用于生成可解释、可继续验证的候选机制结构。

## 当前主线：SCP2154 速度耦合原型

当前最重要的版本是 **SCP2154 stagewise velocity-coupling prototype**。

它基于按疾病阶段分层的肝病单细胞数据，比较：

- 一个细胞群的模块状态，能否预测另一个细胞群的模块速度
- 这一方向是否强于反方向
- 在 donor-held-out 和 permutation 过滤下，哪些边还能保留下来

当前实现包含：

- 固定、可解释的细胞类型功能模块
- 用伪时间邻域差分近似局部模块速度
- 在每个 disease stage 内按 donor 交集配对
- 双向方向性比较：
  - `A score -> dB/dt`
  - `B score -> dA/dt`
- 进一步组织成跨阶段的候选机制链

项目摘要：

- [docs/project2_velocity_coupling_summary.zh-CN.md](docs/project2_velocity_coupling_summary.zh-CN.md)

面试版讲稿摘要：

- [docs/scp2154_velocity_coupling_interview_summary.zh-CN.md](docs/scp2154_velocity_coupling_interview_summary.zh-CN.md)

已追踪的正式图：

- 流程图：[docs/assets/project2/workflow_figure.svg](docs/assets/project2/workflow_figure.svg)
- 结果链条图：[docs/assets/project2/conclusion_chain_figure.svg](docs/assets/project2/conclusion_chain_figure.svg)

## 代表性结果

在更严格的单向标准下，当前原型保留了一条候选方向链：

`炎症性内皮 -> 干扰素应答型髓系 -> 炎症型成纤维细胞 -> 肝细胞恶性样程序`

这条链应当理解为**候选方向结构**，而不是严格因果证明。

更稳妥的解释是：肝细胞恶性样程序的加速，更可能与前序微环境重塑有关，而不是孤立发生。

## 目录结构

```text
multicell-state-dynamics/
├── docs/                         说明文档、项目摘要和正式图
├── scripts/                      可直接运行的原型脚本
├── src/multicell_dynamics/       核心建模代码
├── tests/                        单元测试
├── README.md
├── README.zh-CN.md
├── pyproject.toml
└── LICENSE
```

## 关键脚本

### 1. 合成数据 demo

```bash
python scripts/run_synthetic_demo.py
```

### 2. 真实数据 macrophage 原型（`GSE185477`）

```bash
python scripts/run_gse185477_demo.py
```

### 3. SCP2154 phenotype baseline

```bash
python scripts/run_scp2154_phenotype_baseline.py \
  --metadata data/raw/scp2154/metadata.tsv.gz \
  --matrix data/raw/scp2154/counts.tsv.gz
```

### 4. SCP2154 stagewise velocity coupling

```bash
python scripts/run_scp2154_stagewise_velocity_coupling.py
```

## 快速开始

```bash
cd "/Users/einstian/Documents/New project/multicell-state-dynamics"
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python -m unittest discover -s tests
```

如果你的终端里已经启用了其他 Python 环境，可以使用更稳妥的本地方式：

```bash
cd "/Users/einstian/Documents/New project/multicell-state-dynamics"
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -e . --no-build-isolation
./.venv/bin/python -m unittest discover -s tests
```

## 结果解释边界

这个仓库对结论边界是有意识控制的：

- 这里使用的速度信号**不是 RNA velocity**
- 这里的跨阶段链条**不是同一批 donor 的纵向追踪证明**
- 保留下来的边更适合解释为**候选方向关系**

这也是整个仓库当前的设计原则。

## 相关文档

- phenotype baseline: [docs/scp2154_phenotype_baseline.md](docs/scp2154_phenotype_baseline.md)
- 过程/代码说明： [docs/prototype_process_code_guide.zh-CN.md](docs/prototype_process_code_guide.zh-CN.md)
- 风险评估： [docs/pre_experiment_assessment_and_risk_table.zh-CN.md](docs/pre_experiment_assessment_and_risk_table.zh-CN.md)

## License

MIT
