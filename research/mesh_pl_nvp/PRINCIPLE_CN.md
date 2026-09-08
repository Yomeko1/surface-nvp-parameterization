# v3.1 Mesh-aligned PL-NVP：从 rollback 问题到完整管线

本文按“旧方法为什么不足 → 直接修改会产生什么新问题 → 如何解决 → 又产生什么问题”的顺序，说明 `research/mesh_pl_nvp` 的动机、结构保证、数值实现和完整运行管线。

这套方法是独立于 v2.4 的研究管线。v3.0 首先实现了 mesh-aligned PL-NVP 的结构合法性；v3.1 在不改变该保证的前提下，加入 H1–Spline–H2 三阶段组合、多圈 scaffold、多尺度 C0 边界模式、exact legal-interval spline 和风险缓解后的学习率恢复。它没有改动保留的 v2.4 Affine/Spline NVP 训练代码，只复用 Tutte 初值、网格 I/O、评价指标、可视化和汇总工具。v3.1 训练器仍不含 rollback。

## 1. 问题链总览

当前方案可以概括为下面这条因果链：

1. **连续 NVP 与最终离散网格不是同一个映射。** 连续 Jacobian 为正，并不能保证只映射顶点后得到的线性三角形不翻转、不重叠，所以 v2.4 需要检测和 rollback。
2. **让网络直接产生 mesh-aligned 分片线性映射。** 这样网络优化的映射与最终输出完全一致；但如果直接预测顶点位置，网络仍可能把三角形翻转。
3. **把每个顶点限制在一环合法域 (K_i) 内。** 该域是若干半平面的交，点留在其中即可保持相邻三角形为正；但相邻顶点同时移动会让彼此的合法域同时变化，显式逆无法重建。
4. **用 proper coloring 分批更新独立顶点。** 同色顶点互不相邻，每个三角形在一个子层中至多移动一个顶点，正向和反向都能重算相同的 (K_i)；但 (K_i) 有界，而 NVP latent 空间是无界的。
5. **用径向 homeomorphism (psi_{K_i}) 共轭 NVP coupling。** 它在 (operatorname{int}K_i) 与 (mathbb R^2) 之间建立显式双射，有限网络输出必定回到合法域；但局部正面积本身还不足以在一般边界条件下推出全局无自交。
6. **固定扩展网格的简单外边界，并用 scaffold 释放原边界。** 扩展圆盘上“所有面正定向 + 边界双射”给出全局单射；原边界成为可移动内点，结束后丢弃 scaffold。但单圈 scaffold 和纯局部更新仍难以产生足够丰富的整体边界变化。
7. **加入可认证的全局 harmonic/boundary modes，并用多圈 scaffold 传递边界运动。** 每个全局模式只更新一个标量系数，其保持所有面积为正的连通区间可以精确求出；H1 先调整全局形状，局部层优化细节，H2 再精修。C0 boundary hats 不要求边界平滑，并按 mesh 边界采样自动筛掉不受支持的尺度。
8. **把局部 affine coupling 扩展为 exact legal-interval spline。** 固定一个坐标后，另一个坐标在 (K_i) 中的合法区间可以精确求出；通过 `atanh` chart 和单调 rational-quadratic spline 更新该坐标，再映回区间，增强表达能力且不离开合法域。
9. **限制层输出并监测 (q)、面积和梯度。** 使用 `float64`、梯度裁剪、mesh-specific 安全峰值、风险降速和带滞回的 LR recovery；任何合法性失败都会终止运行，而不是回退参数。

因此，v3.1 的核心仍不是“检测到非法后把一步撤销”，而是“每一个可执行的网络层在结构上只能产生合法扩展网格”；新增组件只扩大这一合法可达集合中的表达能力和改善优化效率。

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

v3.0 使用正半轴 softsign。v3.1 的 affine 消融默认改用更缓和的 `atanh/tanh`
径向对：

\[
\boxed{
\psi_K(x)=c+\rho_K(d)\operatorname{atanh}(q)d
}
\]

和

\[
\boxed{
\psi_K^{-1}(z)=c+\rho_K(d)
\tanh\!\left(\frac{\|z-c\|}{\rho_K(d)}\right)d
}.
\]

任意有限 (z) 都被映到 (K) 的严格内部，所以网络不可能把顶点放到合法域边界或外部。softsign 仍作为可选消融保留。代码中的命名为：

| 数学映射 | 代码函数 | 方向 |
|---|---|---|
| (psi_K) | `from_polytope` | (K\rightarrow\mathbb R^2) |
| (psi_K^{-1}) | `to_polytope` | (mathbb R^2\rightarrow K) |

## 7. 第五步：在合法 latent 坐标中执行 affine 或 spline coupling

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

### 7.1 v3.1 的 exact legal-interval spline

完整二维径向 affine coupling 合法但表达能力有限。v3.1 默认采用一维条件
rational-quadratic spline，并在不同子层交替更新 x/y 坐标。以更新 x 为例，固定当前
y 后，将 (K_i) 的所有半平面约束沿水平线求交，可精确得到

\[
x_i\in(\ell_i(y_i),u_i(y_i)).
\]

先把合法区间标准化到 (-1,1)，再用

\[
z_i=\operatorname{atanh}\!\left(
2\frac{x_i-\ell_i}{u_i-\ell_i}-1
\right)
\]

映射到实数轴。conditioner 根据冻结邻域和保留坐标预测单调 RQS 的 widths、heights
和 derivatives；默认 8 个 bins。最后通过 `tanh` 和区间仿射变换映回
((\ell_i,u_i))。单调 RQS 的构造来自
[Neural Spline Flows](https://arxiv.org/abs/1906.04032)，而“每步重算精确合法区间”是
本 mesh-aligned 版本维持离散合法性的关键。任何有限 latent 输出都严格落在区间内部，
所以 spline 增强表达能力而不需要 rollback。

### 7.2 Conditioner 的实际输入和输出

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

最后一层零初始化，使初始网络接近恒等映射。affine 路径的前两个输出形成
log-scale、后两个输出形成 shift：

\[
s=s_{\max}\tanh(\widehat s),
\qquad
t=t_{\max}\,L_{uv}\tanh(\widehat t).
\]

其中 `max_log_scale` 是 (s_{\max})，`max_shift_fraction` 是相对 UV 尺度的
(t_{\max})。spline 路径则输出 8 个 bins 对应的 widths、heights 和内部
derivatives，并由 softmax/softplus 保证单调。可选 `local-geometry` 再加入 6 个静态
3D 局部统计；现有消融中它没有改善最终 area-weighted SD，所以默认仍为 `basic`。

**随之出现的问题：** 所有原始面局部正定向，在一般拓扑和任意边界映射下仍不能单独推出全局无重叠；必须把边界条件和拓扑假设写清楚。

## 8. 第六步：从局部无翻转到全局无自交

当前理论和实现针对拓扑圆盘。管线验证：网格连通、边流形、只有一条有序边界环，并满足 Euler characteristic (V-E+F=1)。若扩展网格同时满足：

1. 每个三角形保持一致正定向；
2. 外边界是固定的简单闭多边形，从而边界映射为双射；

则 PL 映射可由 degree 理论推出全局单射。理论依据是 Lipman 的 [Bijective Mappings of Meshes with Boundary and the Degree in Mesh Processing](https://arxiv.org/abs/1310.0955)。所以无自交保证并不是“局部面积为正”的经验外推，固定简单边界是必要环节。

**随之出现的问题：** 如果直接固定原网格边界，优化只能在固定凸边界内进行，而高质量参数化的最优边界通常并非圆或凸多边形。

## 9. 第七步：scaffold 释放原网格边界

v3.0 采用一圈 scaffold；v3.1 默认采用 6 圈过渡 scaffold：

1. 提取原网格有序边界；
2. 在其外部建立与边界顶点一一对应的固定凸外环，默认 scale 为 1.1；
3. 在原边界和外环间插入 5 个可移动过渡环并三角化连接；
4. 过渡位置按 ((r/6)^3) 分布，全局模式位移按对应的几何权重逐圈衰减到 0；
5. 原边界因此成为扩展圆盘的内部顶点，可以参与所有 PL-NVP 更新；
6. 最终只输出原始顶点和原始面，丢弃 scaffold。

这种“用外部单纯复形把全局碰撞约束转成扩展网格局部可注入约束”的动机来自 Jiang、Schaefer、Panozzo 的 [Simplicial Complex Augmentation Framework for Bijective Maps](https://people.engr.tamu.edu/schaefer/research/scaffold.pdf)。当前实现使用较简单的一一对应凸外环，而不是复现论文的全部 scaffold 优化算法。

SD loss 只在原始三角形上计算；scaffold 面不追求形状质量，只负责让原边界可动并维持全局单射证明。因此评价时也应重点看原始面，不能让辅助三角形的高畸变掩盖原网格质量。

### 9.1 为什么增加 H1 和 H2

纯局部颜色更新能够表达复杂变形，但整体边界运动传播慢，容易在较少训练轮数内停留在
Tutte 圆附近。v3.1 预计算一组离散 harmonic 位移模式：边界端指定运动，内部通过
Laplacian 调和延拓。沿一个固定模式改变标量系数时，每个三角形面积是该系数的二次
多项式，因此可以精确求出包含当前位置、且所有面保持正面积的连通合法区间。将系数经
`atanh/tanh` 映到实轴做有界 affine coupling，仍然具有显式逆和结构合法性。

H1 在局部 spline 前快速调整全局形状与边界；H2 在局部细节优化后再做全局精修。两者
优化同一个原始面 area-weighted SD，不增加边界或 scaffold 质量损失。

### 9.2 C0 boundary hats

低频 Fourier 模式较平滑，难以形成 SLIM 常见的非光滑分段边界。v3.1 额外加入
4/8/16/32 控制数的多尺度分片线性 hat modes。它们只要求边界连续，不要求导数连续。
若某个 mesh 的边界顶点不足以实际采样某一级 hats，支持度预检会自动删除该尺度；例如
David328 和 Isis 的发布实验自动使用 4/8/16，而不是人为设置 mesh 特供参数。

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

v3.1 采用以下措施：

- 全流程使用 `torch.float64`；
- `max_log_scale`、`max_shift_fraction` 和 spline 单调参数化限制每层行为；
- 使用 gradient clipping；
- 每一步断言扩展网格所有有向面积为正且数值有限；
- 持续记录 (q_{max})、最小面积、最小面积比和 slack；
- 根据初始 `conditioning_risk_max` 把配置 LR 缩放成该 mesh 的安全峰值；
- 高风险初值使用 100-step warm-up，从安全峰值的 25% 逐步升高；
- 训练中风险超过 0.95 时动态降低 LR；
- 风险降到 0.94 以下并连续保持 20 steps 后，每 10 steps 将 LR 乘 1.05，最高不超过安全峰值；风险重新升高就重置 patience；
- 若硬合法性断言失败，运行直接报错且不保存非法结果，不撤销一步继续训练。

`softsign` 下直接用 (q) 作为风险；`atanh` 下使用
(1-\sqrt{1-q^2}) 将其换算到共同的 ([0,1]) 灵敏度尺度。该量仍只是条件风险，不是
SD。风险恢复机制在五模型验证中只在 Cow 上触发；其余模型条件不满足时保持原轨迹，
所以它是通用控制规则而非 Cow 名称分支。

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
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/David328/David328.usda `
  --output-dir data/output/mesh_pl_nvp/v3.1/David328
```

执行顺序如下：

1. 读取 OBJ/USD/USDA 网格和 3D 几何；
2. 检查连通、边流形、单边界环和圆盘 Euler characteristic；
3. 若未显式传入 `--initial-uv`，无论文件中是否已有 UV，都重新计算 Tutte 圆边界初值；当前不接入 Mean-Value 或 ABF++；
4. 根据 `geometry_scale` 归一化几何尺度并验证初值 0 翻转、0 自交；
5. 建立 scale 1.1、6 圈、指数 3 的 scaffold，并固定最外凸环；
6. 构造 Fourier 与 C0 boundary-hat harmonic modes，自动去掉边界采样不足的 hat 尺度；
7. H1 使用 Adam 优化 100 iterations，所有模式系数限制在精确正面积区间；
8. 对扩展网格着色，使用 exact legal-interval spline PL-NVP 优化 500 iterations；局部 LR 经过初始风险缩放、warm-up、动态降速和可选恢复；
9. H2 再使用同一组可认证 harmonic modes 优化 100 iterations；
10. 反向通过全部 coupling layers，测量 round-trip 误差；
11. 对最终原网格和扩展网格进行翻转、自交、面积、(q)、risk 和畸变检查；
12. 在 `two_stage/` 与 `three_stage/` 分别保存中间和最终结果、模型、实际配置、JSON/CSV 指标与诊断图。

## 13. v3.1 默认参数及含义

可执行配置文件是 `research/mesh_pl_nvp/v3_1_default.yaml`。YAML 提供默认值，
显式命令行参数优先；未知配置键会报错，避免静默拼写错误。

### 13.1 初值

| 参数 | 默认值 | 含义 |
|---|---:|---|
| 初值方法 | `Tutte` | 固定使用具有凸圆边界的合法初值 |
| 边界 | `circle` | Tutte 初值边界形状 |
| geometry scale | `true` | 按几何尺度归一化，减少不同 mesh 的量纲差异 |

### 13.2 三阶段与局部模型

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `harmonic_iters` | `100` | H1 全局/边界优化轮数 |
| `local_iters` | `500` | 中间 exact-interval spline 优化轮数 |
| `final_harmonic_iters` | `100` | H2 全局精修轮数 |
| `harmonic_cycles` | `2` | 每个 harmonic stage 的模式循环数 |
| `local_cycles` | `4` | 局部层完整颜色循环数 |
| `local_hidden_dim` | `32` | spline conditioner 隐藏维度 |
| `local_latent_transform` | `spline` | 使用单调 RQS；`affine` 保留为消融 |
| `local_spline_bins` | `8` | 一维 spline 分段数 |
| `local_center_iterations` | `12` | 每次动态 (K_i) analytic center Newton 轮数 |

### 13.3 训练和风险学习率

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `seed` | `20260906` | 五模型发布实验随机种子 |
| `harmonic_lr` | `0.02` | H1/H2 Adam 学习率 |
| `local_lr` | `0.003` | 局部配置 LR；实际安全峰值由初始风险缩放 |
| `local_risk_threshold` | `0.85` | 启用初始风险缩放和 warm-up 的阈值 |
| `local_risk_warmup_iters` | `100` | 高风险初值 warm-up 长度 |
| `local_dynamic_risk_threshold` | `0.95` | 训练中超过它就动态降速 |
| `local_risk_recovery_threshold` | `0.94` | 低于它才累计恢复 patience，与 0.95 形成滞回 |
| `local_risk_recovery_patience` | `20` | 首次恢复前连续安全 steps |
| `local_risk_recovery_interval` | `10` | 后续恢复间隔 |
| `local_risk_recovery_factor` | `1.05` | 每次 LR 恢复乘数；不超过安全峰值 |
| `gradient_clip` | `10.0` | 全局梯度范数裁剪阈值 |
| `device` | `cuda` | 默认设备；无 CUDA 时显式改为 `cpu` |

### 13.4 边界与 Scaffold

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `frequencies` | `2,3,4,5` | radial Fourier 全局模式频率 |
| `boundary_hat_counts` | `4,8,16,32` | 多尺度 C0 边界控制数 |
| `auto_boundary_hat_counts` | `true` | 自动删除采样不足的 hats |
| `scaffold_scale` | `1.1` | 固定凸外环包围尺度 |
| `scaffold_rings` | `6` | 原边界到固定外环的总圈数 |
| `scaffold_transition_exponent` | `3.0` | 过渡几何与模式衰减指数 |
| `scaffold_mode_profile` | `geometric` | 模式位移与 scaffold 几何一致过渡 |

## 14. 输出、评价指标与结果报告

每次通用运行在 `two_stage/` 与 `three_stage/` 分别保存 H1+local 和
H1+local+H2。发布结果以 `three_stage/` 为准，包括：

- 优化后的 mesh；
- 实际使用的 config；
- `metrics.json`；
- JSON/CSV summary；
- runtime 和训练轨迹；
- `model_state.pt`；
- UV、边界、SD、翻转、相交和 loss 图。

当前主指标是原始面的 area-weighted mean SD。median/p95/p99/max 只作为分布诊断，
因为 v3.1 没有 tail-aware loss；因此极端 SD 可以轻微变差而整体 SD 下降。合法性还要
同时检查原网格与扩展 scaffold 的翻转、自交和最小面积，并确认 round-trip error 与
`rollback_enabled=false`。metrics 同时记录训练耗时、风险降速/恢复事件和 (q_{max})。

本地结果报告与结果放在一起：

```text
data/output/mesh_pl_nvp/BALLS_EXPERIMENT_CN.md
data/output/mesh_pl_nvp/SIMPLE_MODELS_1000_CN.md
data/output/mesh_pl_nvp/latent_spline_local_iterations_20260907/LOCAL_ITERATIONS_500_CN.md
data/output/mesh_pl_nvp/risk_lr_recovery_20260908/RISK_LR_RECOVERY_CN.md
```

`data/output/` 被 Git 忽略，因此这些实验结果和报告仅保留在本机；v3.1 仓库只提交实现、配置、测试和方法说明。

## 15. 当前保证与限制

在输入为合法拓扑圆盘、scaffold 外边界固定且简单、浮点计算没有失效的前提下，当前层结构给出：

- 每次颜色子层后所有扩展面正定向；
- 每层是 mesh-aligned PL 映射；
- 正向和反向按相反颜色顺序显式计算；
- 扩展圆盘全局单射，因而原网格部分也无翻转、无自交；
- 训练无需 rollback、翻转 barrier 或相交 penalty。

发布配置在 Balls、Cow、David328、Isis、NefertitiFace 上均得到原网格和 scaffold
0 翻转、0 自交。局部 200 增至 500 iterations 后，五个模型的整体 SD 均下降；风险
恢复只在 Cow 上触发，其他四个模型没有副作用。NefertitiFace 的最大 SD 略升，但整体
area-weighted SD 从 4.8850 降至 4.8069，符合当前只关注整体 SD 的评价口径。

当前限制包括：

- (\rho_K) 中的 `min` 在命中边切换方向上只分片可微；
- (q\to1) 时条件数恶化，理论合法不等于数值稳健或低畸变；
- 动态合法域和 analytic center 是主要运行成本，现有 profile 中占绝大部分时间；
- 运行速度明显慢于 v2.4 Spline；
- 整体 SD 尚未在每个 mesh 上超过 v3.0 或成熟方法；
- boundary hats 增强了非光滑边界表达，但通常仍不能复现完全自由的 SLIM 边界；
- 当前不优化尾部 SD，也不支持 00027 一类需进一步拓扑处理的复杂输入；
- 当前只支持具有单边界的圆盘拓扑，其他拓扑需要先切割或扩展理论。

这些限制不改变 v3.1 的核心闭环：合法性由可达集合和拓扑边界条件提供，优化器只在合法映射族中降低原始面 SD。

## 16. 思想来源对应表

| 组件 | 论文或资料 | 当前方案中的作用 |
|---|---|---|
| Affine coupling 与显式逆 | [Real NVP](https://arxiv.org/abs/1605.08803) | latent 空间中的可逆更新 |
| 单调 rational-quadratic spline | [Neural Spline Flows](https://arxiv.org/abs/1906.04032) | v2.4 连续样条背景与 v3.1 exact legal-interval spline 的单调变换 |
| 组合可注入 PL 网格层 | [TutteNet](https://arxiv.org/abs/2406.12121) | mesh-aligned 多层变形的总体动机 |
| 单顶点可行域 | [Optimal Point Placement for Mesh Smoothing](https://arxiv.org/abs/cs/9809081) | 将一环合法位置写成半平面交 (K_i) |
| 有界凸域径向双射 | [Semi-Discrete Normalizing Flows](https://arxiv.org/abs/2203.06832) | (mathbb R^2) 与动态 (K_i) 间的显式 homeomorphism |
| Analytic center | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) | 为动态 (K_i) 选择可重算的唯一内部 anchor |
| 图结构 conditioner | [Graphical Normalizing Flows](https://arxiv.org/abs/2006.02548) | 邻域条件网络的相关设计思路 |
| 局部保向到全局双射 | [Lipman 2014](https://arxiv.org/abs/1310.0955) | 固定简单边界下的全局无重叠依据 |
| Scaffold 扩展 | [Jiang et al. 2017](https://people.engr.tamu.edu/schaefer/research/scaffold.pdf) | 释放原边界，同时固定扩展网格外边界 |
