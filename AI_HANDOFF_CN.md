# Surface NVP v3.2 交接说明

更新：2026-09-21。本文件替代此前停在 v3.2 探索状态的本地说明。

## 当前版本

正式发布为 v3.2；默认 H1 affine100、Local RQS500、H2 affine300。
只有 H2 max_log_scale/max_shift 从 0.08/0.20 改为 0.16/0.40，H1 不变。
正式入口 `python -m research.mesh_pl_nvp.run_v32`，安装命令 `surface-nvp-pl`。
默认 preset `research/mesh_pl_nvp/v3_2_default.yaml`。
v3.1 配置/历史标签保留，v2.4 训练行为不变。

## 目标和已有成果

在单边界圆盘三角网格上，用 mesh-aligned PL-NVP 结构维持正定向与全局单射，
不通过 rollback、翻转 barrier 或自交 penalty 实现。目标仍为原始 3D 面积加权
regularized SD，strict SD 仅另列审计。
固定简单 scaffold 外边界和全部扩展面正定向是结构保证的条件，数值执行仍需审计。

同 seed=20260906：00027 的 SD 54.434032 -> 47.082369，Cow 15.062480 -> 13.992588。
两例原/扩展/导出终点均零翻转、自交。Cow 历史原版 UV 逐位复现。
这些是两例单种子证据，不能扩展为任意 mesh 均改善。

## 未解决问题

- 00027 组合网络浮点逆仍失败，但最终正向 PL 几何映射的审计通过。
- 理论可逆不等于浮点恢复稳定。应用若要求全网络逆，必须额外处理。
- 仅优化整体 SD，尾部畸变仍大；scaffold 可能限制边界，速度仍有改善空间。
- 采样几何误差不是连续最坏界，GEOS 相交不是任意输入精确算术证书。
- H2 RQS 曾在 00027 第 261 次更新后的前向合法区间计算失败，未纳入发布。

## 不能做的事

- 不用网络往返误差阈值阻断训练、挑选终点或覆盖成功参数化；前向非法仍停止。
- 不混淆几何与网络往返，不把 status=ok 当精度认证。
- 不把未查相交的 null 当成 0，不混用 float32/float64 或 strict/regularized SD。
- 不把辅助面加进目标，不删除其合法性约束，不私自加 rollback 或改变理论前提。
- 不覆盖发布标签、输入数据和历史结果；新运行必须新目录。
- OOM 先等价有限内存方案，仍失败就停；不自动降规模/精度/约束。
- 先短诊断再长实验，报告预计时间，长运行低频检查。
- 不提交原始 checkpoint、临时文件或大体积输出；新增输入需单独确认，不随发布误提交。

## 数据和清理

保留 `data/output/v3_1_audit_00027/20260917_full_01`、Cow 验证、00027 后续实验中的
原幅度/宽幅对照和逐位续训验证，以及 v3.0/v3.1 历史基线。
发布验证在 `data/output/v3_2_release/`。旧探索源码备份、分支 bundle、
清理清单及小体积指标记录在 `data/archive/v3_2_release_20260921/`，均不上传 GitHub。
旧探索分支/worktree 和未采用的大体积输出在发布后清理；原始输入及旧 tag 保留。
本地若仍显示未跟踪的新 mesh 输入，视为用户数据，不删除、不自动纳入发布。

优先阅读 `v3.2.md`、模块 `README.md`、`RUN_PIPELINE_CN.md`、`AUDIT_PROTOCOL_CN.md`。
方法推导在 `PRINCIPLE_CN.md`；引用记录在 `AUDIT_REFERENCES_CN.md` 和 `.bib`。
