# v3.0 Mesh-aligned PL-NVP：从 rollback 问题到完整管线

本文按“旧方法为什么不足 → 直接修改会产生什么新问题 → 如何解决 → 又产生什么问题”的顺序，说明 `research/mesh_pl_nvp` 的动机、结构保证、数值实现和完整运行管线。

这套方法是 v3.0 的独立研究管线。它没有改动保留的 v2.4 Affine/Spline NVP 训练代码；只复用了 v2.4 的 Tutte 初值、网格 I/O、评价指标、可视化和汇总工具。v3.0 的 PL-NVP 训练器不含 rollback。

## 1. 问题链总览

当前方案可以概括为下面这条因果链：

1. **连续 NVP 与最终离散网格不是同一个映射。** 连续 Jacobian 为正，并不能保证只映射顶点后得到的线性三角形不翻转、不重叠，所以 v2.4 需要检测和 rollback。
2. **让网络直接产生 mesh-aligned 分片线性映射。** 这样网络优化的映射与最终输出完全一致；但如果直接预测顶点位置，网络仍可能把三角形翻转。
3. **把每个顶点限制在一环合法域 (K_i) 内。** 该域是若干半平面的交，点留在其中即可保持相邻三角形为正；但相邻顶点同时移动会让彼此的合法域同时变化，显式逆无法重建。
4. **用 proper coloring 分批更新独立顶点。** 同色顶点互不相邻，每个三角形在一个子层中至多移动一个顶点，正向和反向都能重算相同的 (K_i)；但 (K_i) 有界，而 NVP latent 空间是无界的。
5. **用径向 homeomorphism (psi_{K_i}) 共轭 NVP coupling。** 它在 (operatorname{int}K_i) 与 (mathbb R^2) 之间建立显式双射，有限网络输出必定回到合法域；但局部正面积本身还不足以在一般边界条件下推出全局无自交。
6. **固定扩展网格的简单外边界，并用 scaffold 释放原边界。** 扩展圆盘上“所有面正定向 + 边界双射”给出全局单射；原边界成为可移动内点，结束后丢弃 scaffold。但径向坐标靠近合法域边界时会病态。
7. **限制层输出并监测 (q)、面积和梯度。** 使用 `float64`、小尺度 coupling、梯度裁剪和只调学习率的自适应策略提高数值稳定性；任何合法性失败都会终止运行，而不是回退参数。

因此，v3.0 的核心不是“检测到非法后把一步撤销”，而是“每一个可执行的网络层在结构上只能产生合法扩展网格”。

## 2. 为什么连续 NVP 仍然需要 rollback

v2.4 的 Affine/Spline NVP 定义连续可逆映射

\[
f_\theta:\mathbb R^2\rightarrow\mathbb R^2,
\qquad \det J_{f_\theta}(x)>0.
\]

这一结论针对的是连续函数 (f_\theta)。实际参数化结果只保存网格顶点的像 (f_\theta(u_i))，三角形内部则由三个像点做线性插值，得到另一个分片线性映射 (widehat f_h)。除非 (f_\theta) 在每个输入三角形上本来就是仿射函数，否则

\[
\widehat f_h\neq f_\theta.
\]

这会造成两个离散化缺口：

- 连续映射下的一条边是曲线，而输出网格使用连接两个端点的弦；三个顶点的弦三角形仍可能翻转。
- 两条连续像曲线互不相交，不代表连接各自端点的两条直线段也互不相交。

所以“连续 NVP 可逆”和“顶点采样后的 PL 网格合法”不是同一结论。v2.4 必须在候选更新后检查翻转与自交，失败时 rollback 到旧参数。该机制有效，但它是一种优化器外部的安全网，传统方法也可以采用，因此不是我们希望依赖的核心创新。

Affine coupling 的显式可逆结构来自 [Real NVP](https://arxiv.org/abs/1605.08803)；v2.4 的单调分段有理二次样条来自 [Neural Spline Flows](https://arxiv.org/abs/1906.04032)。两者保证各自构造的连续变换性质，并不自动消除上述离散化缺口。

## 3. 第一步：让每一层就是网格上的 PL 映射

设第 (l) 层顶点为 (U^{(l)}=\{u_i^{(l)}\})，拓扑连接始终不变。对有向面 (f=(i,j,k))，定义

\[
D_f^{(l)}=
\begin{bmatrix}
u_j^{(l)}-u_i^{(l)} & u_k^{(l)}-u_i^{(l)}
\end{bmatrix}.
\]

该层在三角形 (f) 上的常 Jacobian 为

\[
A_f=D_f^{(l+1)}\left(D_f^{(l)}\right)^{-1},
\qquad
\det A_f=
\frac{\det D_f^{(l+1)}}{\det D_f^{(l)}}.
\]

只要源层合法，且目标层所有有向面积为正，就有 (det A_f>0)。网络层、训练中检查的网格和最终保存的网格现在是同一个 PL 对象，不再存在“连续曲线与离散弦”之间的差异。

这一总体方向受到 [TutteNet](https://arxiv.org/abs/2406.12121) 的启发：通过组合可注入的二维 PL 网格变形构造深层可逆变换。区别是 TutteNet 的二维层采用可学习 Tutte embedding，而本原型研究基于输入网格一环合法域的局部 NVP coupling。

**随之出现的问题：** 如果 MLP 直接预测所有新顶点，即使层被定义为 PL，预测值仍然可能翻转三角形。因此还必须限制每个活动顶点的可达位置。

## 4. 第二步：一环合法域 (K_i)

固定顶点 (i) 的邻点，只把 (u_i) 移到 (x)。对包含它的有向三角形 (f=(i,j,k))，不翻转条件为

\[
\det(u_j-x,\,u_k-x)>0.
\]

邻点固定时，该式关于 (x) 是线性不等式，定义一个开半平面。所有关联面的约束相交得到

\[
K_i=
\bigcap_{(i,j,k)\in F}
\left\{x:\det(u_j-x,u_k-x)>0\right\}.
\]

(K_i) 是凸多边形，当前合法位置位于其中；任何 (x\in\operatorname{int}K_i) 都保持顶点 (i) 的全部关联面为正。将单顶点的无翻转位置描述为局部可行域，和 Amenta、Bern、Eppstein 的 [Optimal Point Placement for Mesh Smoothing](https://arxiv.org/abs/cs/9809081) 中 feasible-region 思想一致；这里将其专门化为有向面积半平面交。

**随之出现的问题：** 如果两个相邻顶点同时更新，定义 (K_i) 的邻点也在变化。各自基于旧网格合法并不能保证组合更新合法；反向时也不能只由当前状态重建正向所用的两个动态合法域。

## 5. 第三步：图着色把更新拆成独立集

对网格一阶邻接图做 proper vertex coloring，使相邻顶点颜色不同。一个子层只更新一种颜色的活动顶点，其余顶点固定。因此

\[
\text{同色顶点互不相邻}
\Longrightarrow
\text{每个三角形在该子层至多更新一个顶点}.
\]

同色顶点可以并行地在各自 (K_i) 内更新；邻点在整个子层中不动，正向和反向都能重新计算完全相同的 (K_i)。一轮依次遍历所有颜色构成一个 coupling cycle，求逆时按相反颜色顺序执行。

Proper coloring 是标准图论工具。把“着色独立集 + 动态一环合法域 + NVP 显式逆”组合为 mesh coupling layer，是当前原型实际研究的结构；现阶段不把这一组合直接宣称为已经完成新颖性证明。

**随之出现的问题：** 神经网络自然输出 (mathbb R^2) 中的无约束值，而合法位置只能在有界 (K_i) 内。直接 clamp 或投影到 (K_i) 会让多个输入落到同一点，破坏双射和显式逆。

## 6. 第四步：径向 (psi_K) 连接有界合法域和无界 latent 空间

我们需要显式双射

\[
\psi_K:\operatorname{int}K\rightarrow\mathbb R^2,
\qquad
\psi_K^{-1}:\mathbb R^2\rightarrow\operatorname{int}K.
\]

其动机直接来自 Chen、Amos、Nickel 的 [Semi-Discrete Normalizing Flows through Differentiable Tessellation](https://arxiv.org/abs/2203.06832)：该工作使用径向 homeomorphism 在无界空间和有界凸 cell 之间转换。本实现保留核心径向构造，但把 Voronoi cell 替换为每一步由网格一环生成的动态 (K_i)。

### 6.1 只由 (K) 决定的中心

将凸域写成

\[
K=\{x\in\mathbb R^2:Ax<b\}.
\]

中心不能取当前活动顶点，否则反向过程无法重建正向中心。本实现使用 log-barrier analytic center：

\[
c(K)=\arg\min_{x\in K}
-\sum_m\log(b_m-a_m^Tx).
\]

它只依赖半平面集合，正反过程可重复计算同一中心。代码用带可行回溯的 Newton 迭代求解；理论和数值方法参考 Boyd 与 Vandenberghe 的开放教材 [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/)。

### 6.2 径向边界距离和双射

对从 (c) 出发的单位方向 (d)，到边界的距离为

\[
\rho_K(d)=
\min_{a_m^Td>0}
\frac{b_m-a_m^Tc}{a_m^Td}.
\]

对 (x\in\operatorname{int}K)，令

\[
r=\|x-c\|,
\qquad d=\frac{x-c}{\|x-c\|},
\qquad q=\frac{r}{\rho_K(d)}\in[0,1).
\]

使用正半轴 softsign 及其逆，得到

\[
\boxed{
\psi_K(x)=c+\rho_K(d)\frac{q}{1-q}d
}
\]

和

\[
\boxed{
\psi_K^{-1}(z)=c+\rho_K(d)
\frac{\|z-c\|}{\rho_K(d)+\|z-c\|}d
}.
\]

任意有限 (z) 都被映到 (K) 的严格内部，所以网络不可能把顶点放到合法域边界或外部。代码中的命名为：

| 数学映射 | 代码函数 | 方向 |
|---|---|---|
| (psi_K) | `from_polytope` | (K\rightarrow\mathbb R^2) |
| (psi_K^{-1}) | `to_polytope` | (mathbb R^2\rightarrow K) |

## 7. 第五步：在 latent 空间执行 Real NVP coupling

活动顶点 (i) 的邻点在当前颜色子层中固定。先计算 (K_i)，再由冻结邻域产生 (s_i,t_i)：

\[
z_i=\psi_{K_i}(u_i),
\qquad
z_i'=z_i\odot\exp(s_i)+t_i,
\qquad
u_i'=\psi_{K_i}^{-1}(z_i').
\]

反向时重算同一个 (K_i,s_i,t_i)：

\[
z_i'=\psi_{K_i}(u_i'),
\qquad
z_i=(z_i'-t_i)\odot\exp(-s_i),
\qquad
u_i=\psi_{K_i}^{-1}(z_i).
\]

中间的 affine coupling 直接沿用 [Real NVP](https://arxiv.org/abs/1605.08803)。让 conditioner 遵循图邻接关系与 [Graphical Normalizing Flows](https://arxiv.org/abs/2006.02548) 有关，但几何合法性来自 (K_i)，不是来自图网络本身。

### 7.1 Conditioner 的实际输入和输出

默认 `basic` 特征每个顶点共 8 维：

- 归一化后的 3D 坐标：3 维；
- 邻点当前 UV 的均值：2 维；
- 邻点当前 UV 的标准差：2 维；
- 归一化 degree（`degree / 12`）：1 维。

MLP 为

```text
Linear(input_dim, hidden_dim)
→ SiLU
→ Linear(hidden_dim, hidden_dim)
→ SiLU
→ Linear(hidden_dim, 4)
```

最后一层零初始化，使初始网络接近恒等映射。前两个输出形成 log-scale，后两个输出形成 shift：

\[
s=s_{\max}\tanh(\widehat s),
\qquad
t=t_{\max}\,L_{uv}\tanh(\widehat t).
\]

其中 `max_log_scale` 是 (s_{\max})，`max_shift_fraction` 是相对 UV 尺度的 (t_{\max})。可选 `local-geometry` 再加入 6 个静态 3D 局部统计：邻边长度的 mean/std/min/max，以及关联面面积的 mean/sum。现有消融中它没有改善最终 area-weighted SD，所以默认仍为 `basic`。

**随之出现的问题：** 所有原始面局部正定向，在一般拓扑和任意边界映射下仍不能单独推出全局无重叠；必须把边界条件和拓扑假设写清楚。

## 8. 第六步：从局部无翻转到全局无自交

当前理论和实现针对拓扑圆盘。管线验证：网格连通、边流形、只有一条有序边界环，并满足 Euler characteristic (V-E+F=1)。若扩展网格同时满足：

1. 每个三角形保持一致正定向；
2. 外边界是固定的简单闭多边形，从而边界映射为双射；

则 PL 映射可由 degree 理论推出全局单射。理论依据是 Lipman 的 [Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing](https://arxiv.org/abs/1310.0955)。所以无自交保证并不是“局部面积为正”的经验外推，固定简单边界是必要环节。

**随之出现的问题：** 如果直接固定原网格边界，优化只能在固定凸边界内进行，而高质量参数化的最优边界通常并非圆或凸多边形。

## 9. 第七步：scaffold 释放原网格边界

本实现采用一圈 scaffold：

1. 提取原网格有序边界；
2. 在其外部建立与边界顶点一一对应的固定凸外环；
3. 将原边界与外环三角化连接成 annulus；
4. 原边界因此成为扩展圆盘的内部顶点，可以参与 PL-NVP 更新；
5. 最终只输出原始顶点和原始面，丢弃 scaffold。

这种“用外部单纯复形把全局碰撞约束转成扩展网格局部可注入约束”的动机来自 Jiang、Schaefer、Panozzo 的 [Simplicial Complex Augmentation Framework for Bijective Maps](https://people.engr.tamu.edu/schaefer/research/scaffold.pdf)。当前实现使用较简单的一一对应凸外环，而不是复现论文的全部 scaffold 优化算法。

SD loss 只在原始三角形上计算；scaffold 面不追求形状质量，只负责让原边界可动并维持全局单射证明。因此评价时也应重点看原始面，不能让辅助三角形的高畸变掩盖原网格质量。

**随之出现的问题：** 顶点虽然始终在 (K_i) 内，但接近其边界时径向坐标会变得病态。

## 10. 第八步：数值条件控制，但不使用 rollback

径向比例

\[
q_i=\frac{\|u_i-c_i\|}{\rho_{K_i}(d_i)}
\]

表示顶点从 analytic center 到当前方向边界所走的比例。(q_i=0) 位于中心，(q_i\to1) 表示逼近合法域边界。代码记录

\[
q_{\max}=\max_i q_i.
\]

由于 (q/(1-q)\to\infty)，高 (q) 意味着 latent 值、逆向误差和梯度敏感性可能迅速增大。它主要是**数值条件与退化风险指标**，不是畸变指标：低 (q) 不保证低 SD，高 (q<1) 也仍然合法。

本实现采用以下措施：

- 全流程使用 `torch.float64`；
- `max_log_scale` 和 `max_shift_fraction` 限制每层 coupling 强度；
- 使用 gradient clipping；
- 每一步断言扩展网格所有有向面积为正且数值有限；
- 持续记录 (q_{max})、最小面积、最小面积比和 slack；
- adaptive plateau 只在损失停滞或几何接近风险阈值时降低学习率；
- 若硬合法性断言失败，运行直接报错且不保存非法结果，不撤销一步继续训练。

这与 rollback 有本质区别：学习率调整只改变后续步长，合法性仍由层结构保证；训练器不存在“候选非法 → 恢复旧参数”的分支。

## 11. 优化目标

当前唯一训练目标是原始表面三角形上的 area-weighted symmetric Dirichlet（SD）能量。对每个面从 3D 局部坐标到 UV 的 Jacobian (J_f)，奇异值为 (sigma_{f,1},\sigma_{f,2})，面能量为

\[
E_f=\sum_{k=1}^{2}
\left(\sigma_{f,k}^{2}+(\sigma_{f,k}+10^{-4})^{-2}\right).
\]

总损失按原始 3D 面积加权：

\[
\mathcal L_{SD}=
\frac{\sum_f A_f^{3D}E_f}{\sum_f A_f^{3D}}.
\]

当前没有加入：

- 翻转 barrier；
- 自交 penalty；
- rollback penalty；
- scaffold 面质量项；
- identity、边界光顺或正则项。

这样可以清楚地区分两件事：合法性由网络结构和拓扑边界条件提供，优化器只负责降低原始网格的几何畸变。

## 12. 完整运行管线

通用入口为：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/David328/David328.usda `
  --output data/output/mesh_pl_nvp/David328/David328_mesh_pl_nvp.usda
```

执行顺序如下：

1. 读取 OBJ/USD/USDA 网格和 3D 几何；
2. 检查连通、边流形、单边界环和圆盘 Euler characteristic；
3. 若未显式传入 `--initial-uv`，无论文件中是否已有 UV，都重新计算 Tutte 圆边界初值；当前不接入 Mean-Value 或 ABF++；
4. 根据 `geometry_scale` 归一化几何尺度并验证初值 0 翻转、0 自交；
5. 默认建立 scale 1.1 的 scaffold，并固定最外凸环；
6. 对扩展网格着色，建立多 cycle 的 mesh-aligned PL-NVP；
7. 使用 Adam 优化原始面 SD，默认 1000 iterations；每一步保持硬合法，不使用 rollback；
8. 反向通过全部 coupling layers，测量 round-trip 误差；
9. 对最终原网格和扩展网格进行翻转、自交、面积、(q) 和畸变检查；
10. 保存网格、配置、模型状态、JSON/CSV 指标、汇总和诊断图。

若使用已有模型继续训练，可以额外指定：

```powershell
--initial-model-state path\to\model_state.pt
```

## 13. 默认参数及含义

默认配置文件是 `research/mesh_pl_nvp/default.yaml`。

### 13.1 初值

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `init.boundary` | `circle` | Tutte 初值的边界形状 |
| `init.geometry_scale` | `true` | 按几何尺度归一化，减少不同 mesh 的量纲差异 |
| `init.initial_uv` | `null` | 不提供外部 UV，强制使用 Tutte；命令行可覆盖 |

### 13.2 模型

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `model.cycles` | `4` | 完整遍历所有颜色的次数；越大表达能力越强、计算越慢 |
| `model.hidden_dim` | `32` | conditioner MLP 的隐藏维度 |
| `model.conditioner_features` | `basic` | 使用 8 维基本特征；可选 `local-geometry` |
| `model.max_log_scale` | `0.08` | 限制 affine coupling 的 log-scale 绝对值 |
| `model.max_shift_fraction` | `0.04` | 限制 shift 相对 UV 尺度的比例 |
| `model.center_iterations` | `12` | 每次动态 (K_i) analytic center 的 Newton 迭代数 |

### 13.3 训练和自适应学习率

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `train.seed` | `20260830` | 随机种子 |
| `train.iterations` | `1000` | Adam 更新次数，与 v2.4 的主要方法对齐 |
| `train.lr` | `0.003` | 初始学习率 |
| `train.device` | `cpu` | 默认设备；可按环境改为 CUDA |
| `train.check_interval` | `10` | 记录完整诊断指标的步数间隔 |
| `train.gradient_clip` | `10.0` | 全局梯度范数裁剪阈值 |
| `train.lr_schedule` | `adaptive-plateau` | 按 mesh 实际损失平台自适应降学习率 |
| `train.min_lr` | `0.0001` | 学习率下限 |
| `train.plateau_window` | `100` | 比较损失改善所用窗口长度 |
| `train.plateau_patience` | `2` | 连续多少个窗口改善不足才衰减 |
| `train.plateau_relative_threshold` | `0.008` | 窗口相对改善低于 0.8% 视为平台 |
| `train.plateau_factor` | `0.5` | 每次衰减乘数 |
| `train.plateau_q_threshold` | `0.97` | 接近合法域边界的风险阈值 |
| `train.plateau_minimum_area_ratio` | `0.25` | 相对初始最小面积的风险阈值 |
| `train.intersection_batch_size` | `262144` | 全局边相交检查的分批规模，主要影响内存和速度 |

### 13.4 Scaffold 与输出

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `scaffold.enabled` | `true` | 使用自由原边界的扩展网格 |
| `scaffold.scale` | `1.1` | 固定凸外环相对原边界包围尺度 |
| `io.prim_path` | `null` | USD prim 路径；为空时按读取器默认规则选择 |
| `io.save_model` | `true` | 保存 `model_state.pt` 以便复现或续训 |

## 14. 输出、评价指标与结果报告

每次通用运行会保存：

- 优化后的 mesh；
- 实际使用的 config；
- `metrics.json`；
- JSON/CSV summary；
- runtime 和训练轨迹；
- `model_state.pt`；
- UV、SD/面积/相交、loss 和 (q)-hotspot 图。

主要质量指标包括 area-weighted SD、mean/median/p95/max SD、最小有向二倍面积、翻转数、非邻接边相交数和 round-trip 误差；同时记录训练耗时、学习率事件、(q_{max}) 及热点一环，便于区分“畸变未充分优化”和“径向映射接近数值边界”。

本地结果报告与结果放在一起：

```text
data/output/mesh_pl_nvp/BALLS_EXPERIMENT_CN.md
data/output/mesh_pl_nvp/SIMPLE_MODELS_1000_CN.md
```

`data/output/` 被 Git 忽略，因此这些实验结果和报告仅保留在本机；v3.0 仓库只提交实现、配置、测试和方法说明。

## 15. 当前保证与限制

在输入为合法拓扑圆盘、scaffold 外边界固定且简单、浮点计算没有失效的前提下，当前层结构给出：

- 每次颜色子层后所有扩展面正定向；
- 每层是 mesh-aligned PL 映射；
- 正向和反向按相反颜色顺序显式计算；
- 扩展圆盘全局单射，因而原网格部分也无翻转、无自交；
- 训练无需 rollback、翻转 barrier 或相交 penalty。

当前限制包括：

- (ho_K) 中的 `min` 在命中边切换方向上只分片可微；
- (q\to1) 时条件数恶化，理论合法不等于数值稳健或低畸变；
- 动态合法域和 analytic center 是主要运行成本，现有 profile 中占绝大部分时间；
- 运行速度明显慢于 v2.4 Spline；
- David328、Isis 等模型的数值精度仍与成熟方法有差距；
- 单圈 scaffold 可能产生锯齿边界和局部高 (q)；
- 当前只支持具有单边界的圆盘拓扑，复杂拓扑需要先切割或扩展理论。

这些限制属于下一阶段的数值精度和效率问题，不改变 v3.0 已验证的核心目标：把合法性从 rollback 的事后修正，转成网络层本身的可达集合约束。

## 16. 思想来源对应表

| 组件 | 论文或资料 | 当前方案中的作用 |
|---|---|---|
| Affine coupling 与显式逆 | [Real NVP](https://arxiv.org/abs/1605.08803) | latent 空间中的可逆更新 |
| v2.4 连续样条背景 | [Neural Spline Flows](https://arxiv.org/abs/1906.04032) | 说明旧连续 NVP 与离散 PL 输出的差异 |
| 组合可注入 PL 网格层 | [TutteNet](https://arxiv.org/abs/2406.12121) | mesh-aligned 多层变形的总体动机 |
| 单顶点可行域 | [Optimal Point Placement for Mesh Smoothing](https://arxiv.org/abs/cs/9809081) | 将一环合法位置写成半平面交 (K_i) |
| 有界凸域径向双射 | [Semi-Discrete Normalizing Flows](https://arxiv.org/abs/2203.06832) | (mathbb R^2) 与动态 (K_i) 间的显式 homeomorphism |
| Analytic center | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) | 为动态 (K_i) 选择可重算的唯一内部 anchor |
| 图结构 conditioner | [Graphical Normalizing Flows](https://arxiv.org/abs/2006.02548) | 邻域条件网络的相关设计思路 |
| 局部保向到全局双射 | [Lipman 2014](https://arxiv.org/abs/1310.0955) | 固定简单边界下的全局无重叠依据 |
| Scaffold 扩展 | [Jiang et al. 2017](https://people.engr.tamu.edu/schaefer/research/scaffold.pdf) | 释放原边界，同时固定扩展网格外边界 |
