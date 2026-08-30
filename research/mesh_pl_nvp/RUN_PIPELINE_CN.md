# Mesh PL-NVP 通用运行说明

## 1. 隔离范围

Mesh PL-NVP 的模型、scaffold 和训练器仍全部位于：

```text
research/mesh_pl_nvp/
```

它不会替换或修改 v2.4 的 Affine、Spline、Direct 和 SLIM 方法。通用入口只复用
v2.4 已经稳定的公共功能：

- OBJ、USD、USDA 输入输出；
- Tutte 初值；
- 翻转和自交检查；
- symmetric Dirichlet、condition number 等评价指标；
- UV、畸变、面积、翻转、自交和 loss 可视化；
- `config.json`、`metrics.json`、`summary.json/csv` 输出格式。

PL-NVP 训练保持 `float64`，损失只计算原始三角形。训练不包含 Jacobian barrier、
自交惩罚和 rollback。若硬合法性断言失败，程序会立即停止并且不保存错误结果。

## 2. 当前默认配置

默认配置文件为：

```text
research/mesh_pl_nvp/default.yaml
```

主要参数为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `cycles` | 4 | 完整颜色更新循环数 |
| `hidden_dim` | 32 | 每个 conditioner 的隐藏宽度 |
| `conditioner_features` | `basic` | 使用当前表现更好的 8 维输入 |
| `iters` | 1000 | 与 v2.4 Direct、Affine、Spline 默认训练轮数对齐 |
| `lr` | 0.003 | Adam 初始学习率 |
| `lr_schedule` | `adaptive-plateau` | 根据相对 loss 改善自适应降学习率 |
| `scaffold.enabled` | `true` | 释放原网格边界 |
| `scaffold.scale` | 1.1 | 当前较稳健的外环尺度 |
| `geometry_scale` | `true` | 与 v2.4 一致的几何尺度归一化 |

## 3. 初值规则

当前只复用 Tutte，不使用 Mean-Value、ABF++ 或自动选择。

不传 `--initial-uv` 时，程序一定重新计算 Tutte。即使输入 USDA 自带 dABF 或其他
UV，也不会直接采用它：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --input data/input/David328/David328.usda `
  --output data/output/mesh_pl_nvp/David328/David328_mesh_pl_nvp.usda
```

若要与某个已经保存的 Tutte 初值严格共用同一组坐标，可以显式指定该文件：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --input data/input/David328/David328.usda `
  --initial-uv data/output/v2.4/David328/David328/initial/David328_initial.usda `
  --output data/output/mesh_pl_nvp/David328/David328_mesh_pl_nvp.usda
```

显式初值仍会检查顶点数、三角形拓扑、翻转和自交。

## 4. 推荐的小模型验证命令

所有命令都应在仓库根目录执行：

```powershell
Set-Location F:\Juyong_Zhang\1\try\CODE
```

David328：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/David328/David328.usda `
  --output data/output/mesh_pl_nvp/David328/David328_mesh_pl_nvp.usda
```

NefertitiFace：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/NefertitiFace/NefertitiFace.usda `
  --output data/output/mesh_pl_nvp/NefertitiFace/NefertitiFace_mesh_pl_nvp.usda
```

Isis：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/Isis/Isis_dABF.usda `
  --output data/output/mesh_pl_nvp/Isis/Isis_mesh_pl_nvp.usda
```

Cow 规模较大，建议前三个完成后再运行：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/Cow/Cow_dABF.usda `
  --output data/output/mesh_pl_nvp/Cow/Cow_mesh_pl_nvp.usda
```

## 5. 固定边界对照

默认启用 scaffold。增加 `--no-scaffold` 即可得到其他参数完全相同的固定边界对照：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/David328/David328.usda `
  --output data/output/mesh_pl_nvp/David328_fixed/David328_mesh_pl_nvp.usda `
  --no-scaffold
```

## 6. 缩短测试或使用 GPU

默认已经是与 v2.4 对齐的 1000 轮。若只检查环境、输入和输出是否正常，可以临时缩短为 300 轮：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/David328/David328.usda `
  --output data/output/mesh_pl_nvp/David328_smoke/David328_mesh_pl_nvp.usda `
  --iters 300
```

若 CUDA 环境支持 PyTorch，可增加：

```powershell
--device cuda
```

例如：

```powershell
python -m research.mesh_pl_nvp.run_pipeline `
  --config research/mesh_pl_nvp/default.yaml `
  --input data/input/Cow/Cow_dABF.usda `
  --output data/output/mesh_pl_nvp/Cow/Cow_mesh_pl_nvp.usda `
  --device cuda
```

## 7. 输出文件

假设输出为 `David328_mesh_pl_nvp.usda`，同目录还会生成：

- `.config.json`：本次实际配置；
- `.metrics.json`：完整初值、最终结果、训练轨迹、q 和扩展网格审计；
- `.runtime.json`：Tutte、训练、最终审计与制图、完整流程的分段耗时；
- `.summary.json`、`.summary.csv`：与 v2.4 相同口径的摘要；
- `.model.pt`：模型参数；
- `.compare.png`、`.uv.png`：UV 对比和最终网格；
- `.distortion.png`、`.distortion_compare.png`：畸变分布；
- `.area_compare.png`：面积变化；
- `.flip_heatmap.png`、`.intersection_heatmap.png`：合法性结果；
- `.loss.png`：训练曲线；
- `.q_hotspots.png`：接近局部合法域边界的热点。

评价优化效果时以原始三角形的 SD、condition number、最小面积、翻转和自交为主。
scaffold 面不进入损失。`q` 只用于判断合法域余量和数值条件，不是参数化质量指标；
metrics 中会分别记录原网格内部更新和释放后的原边界更新。
