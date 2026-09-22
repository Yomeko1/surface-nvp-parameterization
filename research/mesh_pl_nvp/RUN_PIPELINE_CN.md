# v3.2 Mesh PL-NVP 运行说明

## 1. 安装与默认值

正式入口 `research.mesh_pl_nvp.run_v32`，安装命令为 `surface-nvp-pl`。
Python >=3.10，训练 float64。保留 v2.4 训练行为，v3.1 配置文件不变。

```powershell
python -m pip install -e ".[test,usd]"
python -m pytest tests research/mesh_pl_nvp/tests -q
```

OBJ 输入输出不需要 usd-core，此时可安装 `.[test]`；无 CUDA 添加 `--device cpu`。
Linux 使用相同命令，多行续行符换成反斜杠。

| 阶段 | 更新数 | 设置 |
|---|---:|---|
| H1 | 100 | affine，max_log_scale=0.08，max_shift=0.20 |
| Local | 500 | RQS，4 cycles，hidden=32，8 bins，bound=8 |
| H2 | 300 | affine，max_log_scale=0.16，max_shift=0.40 |

Tutte circle、geometry_scale=true、六圈 scaffold、scale=1.1、指数=3、Fourier
2/3/4/5、自动支持 hats 4/8/16/32、LR=0.02/0.003、风险降速/恢复、seed=20260906
均保留 v3.1。输入自带 UV 也会重新生成 Tutte 初值。
`v3_2_default.yaml` 与新入口内建值一致，CLI 覆盖 YAML。

## 2. 正式运行

```powershell
python -m research.mesh_pl_nvp.run_v32 --config research/mesh_pl_nvp/v3_2_default.yaml --input "data/input/Cow#/Cow_dABF.usda" --output-dir data/output/v3.2/Cow/ours
python -m research.mesh_pl_nvp.run_v32 --input data/input/00027/Input.obj --output-dir data/output/v3.2/00027/ours --export-format usda
```

其他保留输入：`Balls/Balls.obj`、`David328/David328.usda`、`Isis/Isis_dABF.usda`、
`NefertitiFace/NefertitiFace.usda`，相对于 `data/input/`。
不表示这些模型均已完成 v3.2 全预算验证。多 mesh USD 场景用 `--prim-path`。

输出目录必须不存在。默认导出跟随输入：OBJ -> OBJ，USD -> USDA。
00027 历史导出审计为 USDA，因此上例显式指定；不能混用 OBJ 与 USDA 量化误差。
训练期间不用实时查看，逐步日志和状态会落盘。

## 3. 短诊断与分开审计

```powershell
surface-nvp-pl --input data/input/Balls/Balls.obj --output-dir data/output/v3.2/smoke --harmonic-iters 2 --local-iters 2 --final-harmonic-iters 2
```

短诊断只验证运行，不用于精度对照。仅训练、稍后审计：

```powershell
python -m research.mesh_pl_nvp.run_v32 train --input "data/input/Cow#/Cow_dABF.usda" --output-dir data/output/v3.2/Cow_deferred
python -m research.mesh_pl_nvp.run_v32 audit data/output/v3.2/Cow_deferred --device cuda
```

重复审计/汇总需新名称，不能覆盖：

```powershell
python -m research.mesh_pl_nvp.run_v32 audit data/output/v3.2/Cow_deferred --audit-name audit_again --report-name report_again
python -m research.mesh_pl_nvp.run_v32 report data/output/v3.2/Cow_deferred --report-name report_new
```

## 4. 输出和加载

2026-09-22 起将展示结果与完整复现记录分开，不改变训练配置或 v3.2 tag。

`data/output/v3.2/00xxx/ours/` 与 `slim/` 并列，参照 SLIM 保留：

- `final.obj`/`final.usda` 和 `initial.obj`/`initial.usda`：终点与初值；终点在逆审计之前保存。
- 初始/最终 UV、畸变热图，以及 UV、畸变和面积对比图。
- 初始/最终翻转热图、最终自交热图、SD 曲线、四指标图。
- `summary.json`/CSV、`config.json`、`completion.json`、`RESULTS.md` 和绘图尺度记录。
- 若有相邻 `slim/00xxx.obj`，自动生成 SLIM 对照图；也可指定 `--slim-result PATH`。
- `run.json`：指向独立归档的相对路径。目录内没有 checkpoint、逐步日志或源码 ZIP。

完整归档默认为 `data/archive/v3.2/00xxx/ours/`：

- `final.model.pt`、`reference.pt`、`snapshots/`：模型、原生 UV、Adam、RNG、调度状态。
- `training.jsonl`、`gradients.jsonl`：逐步记录；`manifest.json`：配置、版本、设备和 hash。
- `audit/`、`report/`：原始审计、每面数据和完整报告。`status.json` 为训练状态。
- `source.ref.json` 指向 `data/archive/_sources/<SHA256>.zip`，按原 ZIP 字节去重。

可用 `--archive-dir PATH` 指定独立且不存在的归档目录，不能与结果目录嵌套。
输出不在 `data/output/` 下时，默认归档为旁边的 `_run_archives/<结果目录名>/`。
两者须在同一磁盘上以保存相对路径；搬迁时保持相对结构。
`train` 模式只生成网格和待审计状态；随后 `audit` 自动补图和指标。
图使用 native float64 数据，不改变 UV 或 SD；截色只影响热图显示，对比图共用色标。
相交计数只在实际检查点画圆圈，未检查的步骤不是零；圆圈不是失败符号。

已有完整运行可不训练，直接生成新的展示目录（旧式原始运行目录也支持）：

```powershell
python -m research.mesh_pl_nvp.run_v32 present data/output/v3.2/00027/ours --output-dir data/output/v3.2/00027/ours_figures
```

再次审计或生成原始报告要选新名称；旧结果不覆盖。重新绘图须选新展示目录。
`present` 使用已完成的审计和报告；若要更新报告图，先执行 `report --report-name NEW`，
再给 `present` 传相同 `--report-name NEW`。

仅加载可信 `.pt` 文件，它们使用 pickle：

```python
from research.mesh_pl_nvp.run_v31_audit import load_checkpoint
model, payload = load_checkpoint("data/output/v3.2/Cow/ours", "cuda")
model.eval()
```

重新审计使用保存的参考几何，不依赖原输入绝对路径。独立审计导出只负责几何/UV，
不负责搬运原纹理资源。原 OBJ I/O 的材质库复制行为保留；纹理图片仍需自行提供。
最终网格和图可单独使用；再次审计或加载模型则需保留归档及 `run.json` 路径关系。

## 5. 误差和失败

定义见 `AUDIT_PROTOCOL_CN.md`。几何/网络分别用固定 D3D/LUV0 归一化。
SD 只按原始 3D 面积加权，不重新缩放；strict 与 regularized 分列。

- 非单边界圆盘输入被拒绝，先处理拓扑。
- 正向非法区间、非有限或面积断言失败时停止，保留现场，不回滚。
- 网络逆失败只记 failed，不阻断合法结果保存，不改变训练或选择终点。
- 逆向 ok 仅表示计算有限，不是精度认证；需看残差。
- 几何审计发现翻转、自交或退化，总体状态失败，诊断仍保留。
- OOM 停止并记录；先研究等价有限内存方法，在新目录重试。不自动降低模型、精度或约束集合。
- 等价方法仍 OOM 则停止汇报。当前没有用户级通用断点恢复命令，failure_state 不作为正常续训点。

## 6. 旧版复现

```powershell
python -m research.mesh_pl_nvp.run_harmonic_local --config research/mesh_pl_nvp/v3_1_default.yaml --input "data/input/Cow#/Cow_dABF.usda" --output-dir data/output/v3.1/Cow_new
```

旧入口写 `two_stage/`、`three_stage/`，部分旧指标为 float32；新表格应使用 v3.2 审计。
严格历史复现使用 v3.1 tag 和对应依赖。不要向 v3.2 入口传 v3.1 YAML 期待旧行为，
其未列出的 H2 参数仍会继承新默认。
