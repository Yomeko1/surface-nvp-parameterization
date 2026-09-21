# 误差审计参考资料

核对日期：2026-09-17。以下链接为来源定位，正式论文按实际引用版本补齐章节和页码。
这些资料都不提供本项目网络往返误差必须小于 1e-8 的普适依据。

1. Behrmann, J.; Vicol, P.; Wang, K.-C.; Grosse, R.; Jacobsen, J.-H. (2021).
   Understanding and Mitigating Exploding Inverses in Invertible Neural Networks.
   AISTATS, PMLR 130:1792-1800.
   https://proceedings.mlr.press/v130/behrmann21a.html
   用途：理论可逆与数值逆稳定的区别。不能据此认定本项目的具体失稳机制或设定 1e-8。
2. CGAL 6.2.1, Planar Parameterization of Triangulated Surface Meshes, User Manual.
   https://doc.cgal.org/latest/Surface_mesh_parameterization/index.html
   用途：分片线性参数化、双射与畸变目标的区分。latest 地址未来会变化，记录本次版本。
3. Open CASCADE Technology, Precision Class Reference.
   https://dev.opencascade.org/doc/refman/html/class_precision.html
   用途：几何距离和参数域精度转换需要尺度/导数和应用语境。
   不能把库内 Confusion 等常量直接移植成本项目的无量纲阈值。
4. Pharr, M.; Jakob, W.; Humphreys, G. Physically Based Rendering, Fourth Edition,
   Shapes / Triangle Meshes.
   https://pbr-book.org/4ed/Shapes/Triangle_Meshes
   用途：重心坐标、三角形查询稳健性与浮点位置误差界，不是参数化误差通用标准。
5. Microsoft, Direct3D 11.3 Functional Specification, section 7.18.16.1.
   https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm
   用途：纹理坐标精度与 subtexel 量级。硬件最低精度要求不等于应用质量预算。
6. Shewchuk, J. R. (1997). Adaptive Precision Floating-Point Arithmetic and Fast
   Robust Geometric Predicates. Discrete & Computational Geometry 18:305-363.
   https://www.cs.cmu.edu/~quake/robust.html
   用途：方向符号的舍入风险与稳健谓词。此实验 Python 有理数回退不是论文 C 实现。
7. Shapely 2.1.2, STRtree documentation.
   https://shapely.readthedocs.io/en/2.1.2/strtree.html
   用途：只读空间索引与点定位、面相交候选。STRtree 不自动使 GEOS 浮点构造变为精确算术。

本项目约定：固定 D3D/LUV0 归一化、三类误差分离、0.1 或 0.01 texel 的示例预算
属于我们的实验/工程选择，不应表述为上述文献给出的普适建议。
