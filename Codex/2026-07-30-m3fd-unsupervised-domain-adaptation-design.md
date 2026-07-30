# FLIR→M3FD 无监督域适应设计：目标统计引导的物理路由微调

> 状态：设计已确认，尚未实施。
>
> 目标是从当前 V2 `source_best.pt` 出发，改善模型在 M3FD 真实雾图上的密度预测、路由激活和去雾效果。该方案属于无监督目标域适应（UDA），不是纯 Source-only 泛化评估。

## 1. 已确认问题与约束

### 1.1 诊断证据

对 M3FD `00343` 的当前 V2 `source_best.pt` 推理表明：

```text
M3FD：density_map mean = 0.275，route_hard fraction = 0.000
FLIR dense 对照：density_map mean = 0.759，route_hard fraction = 0.391
```

模型在真实 M3FD 上没有完全忽略雾，但 HDE 的密度标定偏低，FeatureGuidedRouter 的 hard route 全为融合，补全分支没有被启用。将整张图强制送入补全分支会产生过曝和伪边缘，表明问题同时存在于路由和补全解码器的真实域适配，而不是简单阈值问题。

### 1.2 数据条件

```text
FLIR：clear RGB、hazy RGB、TIR、M_GT=1-T、mask_GT
M3FD：真实 hazy RGB、TIR；没有对应 clear RGB、M_GT 或 mask_GT
```

M3FD 的 361 张图像可作为无标注目标域数据。项目不引入第三方数据集、额外预训练网络、CycleGAN 或扩散式风格迁移模型，也不将 M3FD 的雾图伪造成物理 `mask_GT`。

## 2. 总体流程

```text
当前 source_best.pt
        │
        ▼
阶段 A：FLIR 目标统计引导的监督微调
        │  使用 FLIR 五元组的真实 M_GT / mask_GT；
        │  M3FD 只提供无标注 RGB/TIR 成像统计。
        ▼
阶段 B：M3FD 无监督 EMA 适应
        │  真实域一致性 + CLIP 约束 + FLIR source anchor；
        │  路由一致性延后启用，避免固化初始全零路由。
        ▼
EMA teacher checkpoint
```

不从零训练。阶段 A、B 都以当前 V2 Source checkpoint 初始化；阶段 B 的 source anchor 持续保留 FLIR 的物理密度、mask 路由和分区域重建监督。

## 3. 阶段 A：目标统计引导的 Source 微调

### 3.1 目标

让带有严格 `M_GT/mask_GT` 的 FLIR 样本呈现更接近 M3FD 的 RGB/TIR 数值分布。这样路由器能在接近真实相机响应的输入上仍接受物理 mask 监督，而无需对 M3FD 生成伪标签。

### 3.2 变换原则

对一个 FLIR 五元组 `(hazy, clear, tir, density_gt, completion_mask_gt)`，随机采样一个 M3FD RGB–TIR 对作为统计参考。仅使用每个模态的低阶、全局成像统计：RGB 各通道的均值和标准差，TIR 单通道等效信号的均值和标准差。

由源 `hazy` 与目标参考图计算有界仿射变换，且以随机强度 `beta` 与恒等变换插值：

```text
x' = clip(a * x + b, 0, 1)
```

约束如下：

- 同一个 RGB `(a, b)` 同步应用于 `hazy` 与 `clear`，以保持成对关系；
- TIR 只使用独立的单调标量增益/偏移，保证热结构的空间位置不变；
- `density_gt` 与 `completion_mask_gt` 不作颜色变换；它们只接受与 RGB/TIR 相同的几何增强；
- 变换增益、偏移与 `beta` 都设上界，防止将目标图的雾层作为像素内容复制到 FLIR 图中；
- 一部分 batch 保持原始 FLIR，不做目标统计变换，防止遗忘源域。

此过程不是从 M3FD 提取雾，也不是图像到图像风格生成。M3FD 仅提供无标签的相机响应和动态范围参考。

### 3.3 优化目标

阶段 A 的优化目标不新加伪标签项，继续使用当前 V2 的完整 Source loss：

```text
L_A = L_rec + lambda_density * L_density + lambda_route * L_route
```

其中：

- `L_rec` 为 CoA 风格全局 `L1 + SSIM + CR`，加融合、补全和边界区域重建；
- `L_density` 由 FLIR 的 `M_GT` 监督；
- `L_route` 为 `route_logits` 对 FLIR 物理 `mask_GT` 的加权 BCE；
- 训练期 GT→预测路由的 teacher-gate 退火保持现有定义。

### 3.4 阶段 A 验收

必须同时满足：

1. FLIR 验证 PSNR/SSIM 不出现明显回退；
2. 固定 M3FD 探针集（至少含 `00343`）的 `route_hard` 不再全部为零；
3. 探针图输出不出现强制全补全时的过曝和伪边缘；
4. 目标统计变换在 mask/density 上保持恒等，且在 `beta=0` 时严格等于原 Source 训练路径。

阶段 A 未通过时，不进入 EMA 阶段 B；应先检查统计变换幅度、源域监督稳定性和探针路由，而不通过降低 hard-route 阈值掩盖问题。

## 4. 阶段 B：M3FD 无监督 EMA 适应

### 4.1 数据与基本目标

M3FD 仅输入真实 `(hazy, tir)`。teacher 和 student 在不同几何增强下预测去雾图、密度图和软路由，现有 EMA 的图像/密度稳定性项保持：

```text
L_real_core = lambda_ema_j * L_J + lambda_ema_m * L_M
```

CLIP 去雾先验保持为可选项。每一步仍计算 FLIR source-anchor 的 `L_A`，总目标为：

```text
L_B = L_real_core + w_route(s) * lambda_ema_r * L_R
    + w_loss_Clip * L_clip
    + lambda_anchor * L_source_anchor
```

### 4.2 延后路由一致性

当前 M3FD 初始路由是全零。若从第一个 EMA step 就施加 `L_R`，teacher 的全零 soft route 会被作为 student 目标，进一步固定错误状态。

因此在阶段 B 的起始窗口：

```text
w_route(s) = 0
```

阶段 A 的 checkpoint 已在真实探针上产生非退化路由后，再将 `w_route(s)` 从 0 平滑升至 1。启用前的人工质量门槛为：固定浓雾 M3FD 探针上有非零补全区域，且该区域与视觉浓雾位置一致，不是全图错误激活。

这不是人为设定一个雾浓度阈值，也不更改路由的物理含义；它只避免无标签 teacher 在错误初态下自蒸馏。

### 4.3 评估协议

使用 M3FD 参与阶段 A 的统计采样和阶段 B 的训练后，实验必须报告为“无监督 M3FD 目标域适应”。不应将同一批 M3FD 图像的结果称为纯合成域零样本泛化。

如果需要更严格的目标域报告，应预先划分目标域适应子集与可视化/诊断保留子集；M3FD 无清晰 GT 时，保留子集用于不参与训练的定性评估和路由诊断，而非 PSNR 计算。

## 5. 实现边界

新增内容限定为：

```text
1. M3FD RGB/TIR 统计采样与有界同步光度变换；
2. 阶段 A 的 Source checkpoint 微调入口或模式；
3. 阶段 B 中路由一致性 L_R 的延迟/ramp 调度；
4. 固定 M3FD 探针的密度、soft/hard route、去雾图监控。
```

不新增或不改变：

```text
不新增 GAN、扩散模型、域判别器、第三方数据或预训练模型；
不为 M3FD 生成/伪造 M_GT 或 mask_GT；
不改变 FeatureGuidedRouter 的 195 通道接口；
不降低 hard route 的 0.5 判定阈值；
不以强制补全替代路由学习；
不删除 FLIR source anchor 的物理监督。
```

## 6. 测试与可观测性

实现后至少覆盖：

1. 相同 RGB 光度变换同步作用于 FLIR `hazy/clear`，而 `density_gt/mask_GT` 不被改变；
2. TIR 变换单调且三通道灰度输入的通道一致性保持；
3. `beta=0` 是严格恒等变换；
4. 统计范围裁剪避免 NaN、反转对比度与越界值；
5. 阶段 A 从 V2 Source checkpoint 恢复并保留原 Source 验证；
6. 阶段 B 在 `w_route=0` 时不计算/反传 `L_R`，ramp 后才恢复；
7. M3FD 探针输出保存 `prediction / density_map / route_soft / route_hard / boundary_map`；
8. 对照实验至少包含：Source-only、阶段 A、阶段 A+B。

## 7. 风险与停止条件

- 低阶统计不能完全消除真实散射模型、场景和传感器响应差异；它是可验证的低风险起点，不保证完全解决目标域问题。
- 若阶段 A 的 FLIR 验证显著下降，应降低变换概率/强度或增加原始 FLIR batch 比例，而非继续扩大目标风格。
- 若阶段 A 后 M3FD hard route 仍全零，停止进入阶段 B；应重新检查 HDE 的真实域密度标定，而不是依赖 EMA 自蒸馏。
- 若 route 激活后出现大面积补全伪影，应优先改善阶段 A 的监督覆盖与补全域适配，而不是提高路由覆盖率。
