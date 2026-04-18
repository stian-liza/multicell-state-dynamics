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
