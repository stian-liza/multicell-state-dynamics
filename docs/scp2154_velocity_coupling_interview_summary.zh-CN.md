# SCP2154 速度耦合原型：面试讲稿摘要

## 这次我具体做了什么

这一次我把之前分开的两条思路真正接起来了。

第一步，仍然是先用固定的细胞类型功能签名，对每个细胞计算模块分数。这里一共用了 24 个节点，也就是 6 类主要细胞群对应的一组固定功能模块。

第二步，在每个细胞类型内部，我不再只看 donor-level 的静态分数，而是先用伪时间邻域差分，给每个细胞估计模块变化率，也就是一个近似的 `dm/dt`。这一步的含义是：我不是只问某个模块高不高，而是问它当前更像在上升还是下降。

第三步，我把每个 donor、每个 stage 的 sender 模块分数，与 receiver 模块速度进行配对，做双向检验：

- `A score -> dB/dt`
- `B score -> dA/dt`

然后用 leave-one-donor-out 的预测增益和置换检验，比较哪一个方向更稳定。

所以这次的主结果，已经不是之前那种“静态 donor-level 相关网络”，而是一个真正的 `score -> velocity` 粗粒化方向网络。

## 主要统计结果

- 采样细胞数：`24,547`
- 固定签名基因数：`165`
- 节点数：`24`
- 双向边测试数：`483`
- 保留下来的 stage-chain 边：`80`
- hepatocyte 相关重点边：`41`

分阶段保留边数：

- `low_steatosis`: `23`
- `cirrhotic`: `24`
- `Tumor`: `33`

## 最值得讲的几条结果

### 1. Tumor 阶段最强的直接肝细胞输入来自 stromal

最清楚的一条直接边是：

- `Stromal.inflammatory_caf -> Hepatocyte.malignant_like`
  - `delta R^2 = 0.8557`
  - `p = 0.0099`

同时还有：

- `Stromal.inflammatory_caf -> Hepatocyte.secretory_stress`
  - `delta R^2 = 0.9211`
  - `p = 0.0099`

这说明在 tumor 阶段，最稳定的直接输入不是泛泛的“免疫细胞很多”，而是一个更具体的炎症型 CAF 状态，去解释 hepatocyte stress / malignant-like 模块速度。

### 2. Endothelial / myeloid 更像前驱链条，而不是最后一跳

这次更有意思的不是单一一条边，而是一条跨阶段的链条开始有结构了：

- `Endothelial.inflammatory_endothelial -> Myeloid.interferon_myeloid`
  - `delta R^2 = 1.3024`
  - `p = 0.0495`
  - `recurrent`

- `Endothelial.sinusoidal_identity -> Stromal.caf_contractile`
  - `delta R^2 = 0.9807`
  - `p = 0.0693`
  - `recurrent`

- `Myeloid.interferon_myeloid -> Stromal.inflammatory_caf`
  - `delta R^2 = 0.3452`
  - `p = 0.0495`

- `Stromal.inflammatory_caf -> Hepatocyte.malignant_like`
  - `delta R^2 = 0.8557`
  - `p = 0.0099`

这给出一个更像病程演化的图景：

早期更像是 endothelial / immune 的扰动，
中期逐渐落到 stromal contractile / inflammatory remodeling，
肿瘤阶段再由 stromal 端更直接地对应 hepatocyte malignant-like 速度。

### 3. 早期最强信号里有一条明显的 B-cell 轴

在 `low_steatosis` 阶段，最强的 hepatocyte stress 速度预测边反而来自 B 细胞相关状态：

- `Bcell.naive_memory -> Hepatocyte.secretory_stress`
  - `delta R^2 = 1.1296`
  - `p = 0.0297`

- `Bcell.plasma -> Hepatocyte.secretory_stress`
  - `delta R^2 = 1.0517`
  - `p = 0.0990`
  - `recurrent`

这个结果统计上是存在的，但生物学解释还需要更谨慎。更适合面试里表述成：

“早期最强输入里出现了免疫轴，尤其是 B-cell 相关状态；这提示早期炎症/抗原呈递环境可能参与 hepatocyte stress 的启动，但具体是哪类细胞真正起主要作用，还需要后续验证。”

## 目前最适合讲成什么生物学故事

我觉得最稳的一版不是说：

“我已经证明某个细胞推动了肝癌发生。”

而是说：

“这个速度耦合原型提示，肝病向肿瘤阶段演化时，存在一个从 endothelial / immune 重排，到 stromal inflammatory CAF 强化，再到 hepatocyte stress / malignant-like 程序加速的阶段性链条。其中 tumor 阶段最直接、最稳定的肝细胞输入来自 stromal inflammatory CAF。”

这样讲有几个好处：

1. 和当前统计结果一致；
2. 有机制味道，但不过度宣称；
3. 能自然引出后续实验验证。

## 面试时要主动说明的边界

### 1. 这是伪速度，不是 RNA velocity

我这里的速度不是 spliced/unspliced RNA velocity，而是基于伪时间前向邻域差分构造的近似 `dm/dt`。它的作用是给横断面数据一个局部动态方向感，因此更适合作为原型验证和假设生成。

### 2. 低阶段出现的 malignant-like 边不能直接解释成“已经癌变”

例如：

- `Endothelial.sinusoidal_identity -> Hepatocyte.malignant_like`
  - `delta R^2 = 1.1574`
  - `p = 0.0099`

这更适合解释成：

“malignant-like signature 在早期已有扰动”

而不是：

“早期 hepatocyte 已经是肝癌细胞”

### 3. 极大的负向 reverse `delta R^2` 不是生物学效应大小

有些反向边会出现很大的负值，那表示的是 reverse 方向在 held-out donor 上非常不稳定，而不是说明存在一个同等强度的负调控。

## 一分钟口头版

如果老师问“你前期做了什么”，我会这样讲：

“我最近做了一个真实数据原型，把多细胞耦合里的动态部分真正接进来了。具体来说，我先用固定的细胞类型功能签名把每个细胞表示成可解释模块，然后在每个细胞类型内部用伪时间邻域差分估计模块变化率，也就是一个近似的 `dm/dt`。在这个基础上，我不再只做静态相关，而是比较 `A 模块分数能不能预测 B 模块速度`，并和反方向 `B -> dA/dt` 做 leave-one-donor-out 的双向检验。  
在 SCP2154 的 liver 数据里，这个原型一共测试了 483 条双向边，保留了 80 条阶段链。当前最清楚的结果是，tumor 阶段最直接的 hepatocyte 输入来自 stromal inflammatory CAF，它既对应 hepatocyte secretory-stress，也对应 malignant-like 模块速度；而 endothelial 和 myeloid 更像位于前面的病程链条里。这说明这个框架至少已经能从横断面单细胞数据里提炼出具有阶段结构的候选动力学机制，下一步就可以围绕这些边去做共培养和阻断验证。” 

## PPT 上最适合放的 take-home message

可以写成三句：

1. `This prototype moves from static donor-level association to score-to-velocity directional testing.`
2. `Tumor-stage direct hepatocyte inputs are strongest for inflammatory CAF-like stromal states.`
3. `Endothelial / immune remodeling appears earlier and may feed into stromal activation before hepatocyte malignant-like acceleration.`
