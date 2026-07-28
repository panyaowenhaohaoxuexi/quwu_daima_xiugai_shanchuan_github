# 训练可观测性设计

## 目标

为 Teacher 与 EMA 训练补充启动诊断、控制台进度展示及可直接分析的持久化指标日志，不改变模型、损失、数据路径、训练调度或 checkpoint 格式。

## 输出契约

每次训练在各自的 `exp_dir` 中创建或覆盖以下文件：

- `run_summary.json`：启动时的设备、数据集规模、批配置、逻辑训练步数、参数量、关键资源加载状态和输出目录。
- `metrics.jsonl`：逐行追加的事件日志。`train_step` 事件记录 epoch、step、全局 step、学习率、耗时及全部损失分量；`validation` 事件记录 PSNR、SSIM、是否刷新 best checkpoint 与学习率。
- `metrics.csv`：与 JSONL 同源的扁平表格，供 Excel 直接打开。未知字段留空。

所有浮点值在写入前转换为普通 Python `float`，路径转换为字符串，保证 JSON/CSV 可读。重新开始一轮非 resume 训练时覆盖日志；resume 时追加并保留原有记录。

## 训练行为

Teacher 每个逻辑训练 step 写一次训练事件，并按现有 50-step 节奏打印包含总损失、主要分量、LR、耗时及训练指标的紧凑摘要。EMA 每个逻辑 step 写一次并按同一节奏打印真实域损失、CLIP、源域锚定、总损失、LR 和耗时。两者在每个 epoch 的验证后写 validation 事件并打印是否得到新的最佳模型。

启动摘要中的资源状态由训练入口明确提供：Teacher 记录 RGB Res2Net 预训练加载状态；EMA 记录 Res2Net、ViT-B/32、RN101 与 haze prompt 的初始化状态。不会重新加载或探测权重文件。

## 边界与验证

日志器保持为独立的 `training/observability.py`，只依赖 Python 标准库，训练入口通过小型函数调用它。测试覆盖：启动摘要可序列化；训练/验证事件同时写入 JSONL 与 CSV；字段随事件扩展时 CSV 保持所有已出现列；恢复模式追加，非恢复模式覆盖。现有端到端测试补充断言输出文件及关键事件。
