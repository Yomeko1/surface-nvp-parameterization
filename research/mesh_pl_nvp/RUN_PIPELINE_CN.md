# v3.1 Mesh PL-NVP 运行说明

## 1. 版本范围

v3.1 的入口是：

```text
research/mesh_pl_nvp/run_harmonic_local.py
```

它与 v2.4 Affine、Spline、Direct、SLIM 管线隔离，只复用网格 I/O、Tutte
初值、SD 指标、合法性检查、可视化和汇总工具。v3.1 全程使用 `float64`，
只优化原始三角形的 3D 面积加权 symmetric Dirichlet；没有 rollback、翻转
barrier、自交 penalty、scaffold 质量项或 tail-aware 项。

## 2. 安装与环境检查

Windows PowerShell：

```powershell
Set-Location F:\Juyong_Zhang\1\try\CODE
python -m pip install -e ".[test,usd]"
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"
python -m pytest research/mesh_pl_nvp/tests -q
```

Linux 服务器：

```bash
cd /你的路径/CODE
python -m pip install -e '.[test,usd]'
python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available())"
python -m pytest research/mesh_pl_nvp/tests -q
```

如果 `cuda=False`，运行时添加 `--device cpu`；CPU 可以运行，但会明显更慢。

## 3. 发布配置

实际可读取的发布配置为：

```text
research/mesh_pl_nvp/v3_1_default.yaml
```

核心值如下：

| 组件 | 默认值 | 作用 |
|---|---:|---|
| `harmonic_iters` | 100 | H1 全局/边界模式优化轮数 |
| `local_iters` | 500 | 中间 mesh-local spline 优化轮数 |
| `final_harmonic_iters` | 100 | H2 全局精修轮数 |
| `harmonic_cycles` | 2 | 每个 harmonic stage 的模式循环数 |
| `local_cycles` | 4 | 局部层的完整颜色更新循环数 |
| `local_hidden_dim` | 32 | 局部 spline conditioner 隐藏宽度 |
| `local_latent_transform` | `spline` | exact legal interval 内的条件 RQS |
| `local_spline_bins` | 8 | 每个一维 RQS 的分段数 |
| `local_radial_map` | `atanh` | q 风险归一化口径；affine 消融时也控制径向映射 |
| `frequencies` | 2,3,4,5 | 平滑全局边界 Fourier 模式 |
| `boundary_hat_counts` | 4,8,16,32 | 多尺度 C0 分片线性边界模式 |
| `auto_boundary_hat_counts` | `true` | 自动去掉边界采样不足的 hat 尺度 |
| `scaffold_rings` | 6 | 原边界到固定外环之间的 scaffold 圈数 |
| `scaffold_scale` | 1.1 | 固定外环包围尺度 |
| `scaffold_transition_exponent` | 3 | scaffold 几何与位移的过渡指数 |
| `local_lr` | 0.003 | 局部阶段配置学习率；实际值会经风险预检缩放 |
| `local_dynamic_risk_threshold` | 0.95 | 超过时动态降低 LR |
| `local_risk_recovery_threshold` | 0.94 | 低于它且持续稳定后才允许恢复 LR |
| `local_risk_recovery_patience` | 20 | 恢复前连续安全轮数 |
| `local_risk_recovery_interval` | 10 | 恢复更新间隔 |
| `local_risk_recovery_factor` | 1.05 | 每次恢复的 LR 乘数 |
| `seed` | 20260906 | 五模型发布实验使用的随机种子 |
| `device` | `cuda` | 默认运行设备 |

YAML 提供默认值，显式命令行选项优先。例如：

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/Balls/Balls.obj `
  --output-dir data/output/mesh_pl_nvp/v3.1/Balls_cpu `
  --device cpu `
  --local-iters 50
```

## 4. 五个模型的完整命令

每个命令使用不同输出目录，避免覆盖已有结果。所有命令都在仓库根目录执行。

### 4.1 Balls

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/Balls/Balls.obj `
  --output-dir data/output/mesh_pl_nvp/v3.1/Balls
```

### 4.2 David328

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/David328/David328.usda `
  --output-dir data/output/mesh_pl_nvp/v3.1/David328
```

### 4.3 Isis

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/Isis/Isis_dABF.usda `
  --output-dir data/output/mesh_pl_nvp/v3.1/Isis
```

### 4.4 NefertitiFace

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/NefertitiFace/NefertitiFace.usda `
  --output-dir data/output/mesh_pl_nvp/v3.1/NefertitiFace
```

### 4.5 Cow

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/Cow/Cow_dABF.usda `
  --output-dir data/output/mesh_pl_nvp/v3.1/Cow
```

上述 USDA 文件即使自带 UV，v3.1 也不会直接使用；它会重新生成 Tutte 圆初值。

## 5. 快速冒烟测试

下面只验证安装、输入、前向/反向和输出，不用于比较精度：

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local `
  --config research/mesh_pl_nvp/v3_1_default.yaml `
  --input data/input/Balls/Balls.obj `
  --output-dir data/output/mesh_pl_nvp/smoke/Balls `
  --harmonic-iters 5 `
  --local-iters 10 `
  --final-harmonic-iters 5
```

## 6. 消融开关

关闭 LR recovery，但保留风险降速：

```powershell
--no-local-risk-recovery
```

同时关闭风险自适应与 recovery：

```powershell
--no-local-risk-adaptive --no-local-risk-recovery
```

改回 affine 局部层：

```powershell
--local-latent-transform affine
```

缩短或延长中间层：

```powershell
--local-iters 200
--local-iters 1000
```

查看全部选项：

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local --help
```

## 7. 输出结构

假设 `--output-dir data/output/mesh_pl_nvp/v3.1/Balls`，程序生成：

```text
Balls/
  two_stage/    # H1 + local spline
  three_stage/  # H1 + local spline + H2，发布结果以此为准
```

每个阶段包含：

- `.usda`：参数化网格；
- `.model.pt`：模型权重；
- `.config.json`：本次实际参数，包括 YAML 路径和自动筛选后的 hats；
- `.metrics.json`：初值、训练历史、SD 分布、q/risk、LR 事件、合法性和耗时；
- `.summary.json`、`.summary.csv`：摘要；
- `.uv.png`、`.compare.png`、`.boundary_compare.png`：UV 与边界；
- `.distortion_compare.png`：SD 热图；
- `.flip_heatmap.png`、`.intersection_heatmap.png`：合法性可视化；
- `.loss.png`：三阶段 loss 曲线。

发布结果首先看 `three_stage/*.summary.json` 的 area-weighted SD；p95、p99、max
用于观察尾部，但 v3.1 不显式优化它们。还必须检查原网格和
`final_extended_injectivity` 均为 0 翻转、0 自交，并确认
`rollback_enabled=false`。

## 8. 失败处理

- 输入拓扑不是单边界圆盘：程序会在训练前拒绝；需要先切割网格。
- CUDA OOM：换一个新输出目录并降低 `--intersection-batch-size`，或者改用
  `--device cpu`；不要覆盖半成品目录。
- hard area assertion、非有限 loss 或最终合法性失败：保留输出和终端错误用于
  诊断，不要把它当作可发布结果，也不要用 rollback 掩盖。
- 中断后当前 runner 不支持精确续训；使用新目录重新运行。

`data/output/` 被 Git 忽略，运行结果只保存在本地。发布到 GitHub 的是代码、
配置、测试和说明，输入网格仍位于 `data/input/`。
