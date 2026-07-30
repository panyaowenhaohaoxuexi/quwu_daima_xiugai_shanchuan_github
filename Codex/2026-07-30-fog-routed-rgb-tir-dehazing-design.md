# 物理 mask 引导的 RGB--TIR 分区去雾：当前实现方案（V2）

> 状态：已实现并作为当前 Source、EMA 与验证主链路的唯一方案。
>
> 本文记录 2026-07-30 的实际接口、训练行为和损失定义。`2026-07-29-fog-routed-rgb-tir-dehazing-design.md` 保留为历史设计，不作为当前实现依据。

## 1. 目标与推理约束

输入为有雾 RGB 图像 `I_rgb` 与配对 TIR 图像 `I_tir`，输出为去雾 RGB 图像 `J_hat`。推理阶段不读取清晰 RGB、雾浓度 GT、`Transmission_Map_GT` 或 `mask_GT`。

模型用一个二值路由表达像素级处理方式：

```text
R = 0：融合。保留仍可信的 RGB 外观和纹理，并由 TIR 提供结构辅助。
R = 1：补全。局部 RGB 已不可用，以 TIR 结构、可靠区域上下文和解码器先验生成结果。
```

`R=1` 并不承诺恢复物理上已完全丢失的真实细节；当 TIR 也缺乏局部结构时，补全分支应给出由上下文支持的平滑、合理结果。

## 2. 合成数据接口与物理标签

`SynthMultiModalDataset` 的固定返回值为：

```python
(hazy, clear, tir, density_gt, completion_mask_gt)
```

默认雾级目录为：

```python
("1_mist", "2_middle", "3_dense", "4_local_extreme")
```

对每一个 `<level>/<stem>`，数据加载器分别读取：

```text
clear/<stem>.<ext>
ir/<stem>.<ext>
hazy/<level>/<stem>.<ext>
Transmission_Map_GT/<level>/<stem>.<ext>
mask_GT/<level>/<stem>.<ext>
```

所有配对以文件主名 `stem` 匹配，缺少任意一项的样本不会进入合成训练集。

### 2.1 两种 GT 的语义

- `Transmission_Map_GT` 当前保存的是雾浓度 `M = 1 - T`，因此 `density_gt_semantics` 固定为 `"density"`；它是连续值监督。
- `mask_GT` 保存物理补全区域标签。它在加雾过程中由未量化的最终透射率生成并落盘；其定义为 `Completion_Mask_GT = 1[T_final <= 0.05]`。训练加载时不依赖 PNG 的具体灰度编码，而是固定二值化：

```python
completion_mask = (load_scalar_map_as_float_tensor(mask_path) > 0.5).float()
```

因此，`Transmission_Map_GT` 与 `mask_GT` 即使使用相同文件名，也来自不同根目录、承担不同监督职责，不会混淆。

### 2.2 几何增强和插值

RGB、TIR 与连续 `density_gt` 使用双线性 resize；`completion_mask_gt` 必须使用最近邻 resize。裁剪、翻转和旋转对五项同步执行，最终 mask 仍再次二值化。

## 3. 模型结构

### 3.1 HDE：连续雾浓度与路由特征提供者

HDE 接收 `(I_rgb, I_tir)`，正式输出为：

```python
{
    "density_map": density_map,
    "tir_structure_pyramid": structure_pyramid,
    "routing_features": {
        "fm_vis": fm_vis,
        "fm_ir": fm_ir,
        "struct_diff_gap": struct_diff_gap,
        "struct_diff_gmp": struct_diff_gmp,
    },
    "debug": None,
}
```

其中 `density_map` 是连续雾浓度预测，`tir_structure_pyramid` 供后续结构恢复使用。路由器只消费 `routing_features` 中已有的特征，不额外新建跨模态编码器。

### 3.2 FeatureGuidedRouter：局部上下文路由

路由器输入为：

```python
x = torch.cat([
    density_map,                         # [B, 1, H, W]
    routing_features["fm_vis"],          # [B, 96, H, W]
    routing_features["fm_ir"],           # [B, 96, H, W]
    routing_features["struct_diff_gap"], # [B, 1, H, W]
    routing_features["struct_diff_gmp"], # [B, 1, H, W]
], dim=1)                                # [B, 195, H, W]
```

网络为：

```text
Conv2d(195, hidden, 1)
→ SiLU
→ Conv2d(hidden, hidden, 3, padding=1)
→ SiLU
→ Conv2d(hidden, 1, 1)
→ route_logits
```

它输出：

```python
route_soft = sigmoid(route_logits / temperature)
route_hard = (route_soft >= 0.5).float()
```

这里没有单调性约束、没有 `MonotonicFogRouter`，也不使用 straight-through hard route。3×3 卷积提供路由所需的局部上下文；密度仅是输入证据之一，而不是人为阈值规则。

### 3.3 融合、补全与共享解码

融合分支面向 `R=0`：保留 RGB 的外观、颜色和有效纹理，并结合 TIR 结构。

补全分支面向 `R=1`：阻止失效局部 RGB 特征经残差、跳连或注意力键值直接回流；使用 TIR 结构、可靠区域上下文、记忆检索和解码器先验恢复。两条路径在特征层由同一 RGB 解码器解码，而非先生成两张图像再做像素拼接。

## 4. 路由训练与推理

### 4.1 温度与教师门控调度

全局步数 `s` 下，路由温度线性退火：

```python
temperature = linear_anneal(
    route_tau_start,
    route_tau_end,
    global_step,
    route_temperature_anneal_steps,
)
```

训练解码使用连续门控：

```python
alpha = 0.0 if route_teacher_anneal_steps == 0 else max(
    0.0,
    1.0 - global_step / route_teacher_anneal_steps,
)
gate = alpha * completion_mask_gt + (1.0 - alpha) * route_soft
output = decode_with_route(context, route_mode="soft", route_override_value=gate)
```

这是一项受控的训练期 teacher forcing：初期 `alpha=1`，解码区域完全由物理 `mask_GT` 决定；随着训练增加，GT 权重线性降低；后期 `alpha=0`，解码完全由模型预测的 `route_soft` 控制。若 `route_teacher_anneal_steps=0`，第一步即为纯预测路由。

`gate` 还用于训练期融合区、补全区与边界区重建损失的区域划分。路由 BCE 仍始终直接以 `completion_mask_gt` 监督原始 `route_logits`，并不因为教师门控而替换其监督目标。

### 4.2 推理

推理不传 GT mask，始终使用：

```python
context = encode_context(hazy_rgb, tir, route_temperature)
prediction = decode_with_route(context, route_mode="hard")
```

因此推理结果只由输入 RGB、TIR、HDE、预测密度图和预测的 hard route 决定。

## 5. 训练目标

Source 的唯一损失入口是 `compute_physical_mask_batch_losses()`，接收上述五元组及模型输出。总损失为：

```text
L_total = L_rec + lambda_density * L_density + lambda_route * L_route
```

### 5.1 分区域去雾重建损失 `L_rec`

全局重建项使用原 CoA 定义的 SSIM 与 CR：

```text
L_global = 0.8 * L1(pred_clear, clear_rgb)
         + 0.2 * (1 - SSIM(pred_clear, clear_rgb))
         + 0.05 * CR(pred_clear, clear_rgb, hazy_rgb)
```

`CR` 是 CoA 的 VGG19 特征对比比率损失：对五个 VGG19 特征层，以权重
`[1/32, 1/16, 1/8, 1/4, 1]` 拉近预测与清晰图，同时相对拉远预测与有雾输入。VGG19 使用 torchvision 的 ImageNet 预训练权重；实现取消了写死的 `.cuda()`，但计算公式、特征切片和层权重不变。

区域损失 `R_region` 由掩码加权的 L1、梯度误差和局部 SSIM 组成：

```text
R_region = 1.0 * L1_region + 0.2 * L_gradient_region + 0.2 * L_local_ssim_region
```

以训练期 `gate` 划分：

```text
L_fuse     = R_region(1 - gate)
L_comp     = R_region(gate)
L_boundary = R_region(boundary(gate))

L_rec = lambda_global * L_global
      + lambda_fuse * L_fuse
      + lambda_comp * L_comp
      + lambda_boundary * L_boundary
```

默认 `lambda_global`、`lambda_fuse`、`lambda_comp` 和 `lambda_boundary` 均为 `1.0`。区域掩码在计算重建项时不反向更新路由器，避免重建损失通过区域面积捷径改变路由标签学习。

损失代码按职责拆分：

```text
loss/ssim.py      # CoA 风格 SSIM
loss/cr.py        # CoA 风格 VGG19 ContrastLoss / CR
loss/regional.py  # 掩码区域的 L1、梯度、局部 SSIM
loss/source.py    # Source 总损失组合
```

### 5.2 密度与物理 mask 监督

连续雾浓度监督为：

```python
L_density = F.smooth_l1_loss(
    density_map,
    density_gt,
    beta=density_smooth_l1_beta,
)
```

物理路由监督为加权 logit BCE：

```python
positives = completion_mask_gt.sum()
negatives = completion_mask_gt.numel() - positives
pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0)

L_route = F.binary_cross_entropy_with_logits(
    route_logits,
    completion_mask_gt,
    pos_weight=pos_weight,
)
```

这种写法在全零或全一 mask 的 batch 上保持数值稳定，并直接对应物理补全标签。

## 6. Source、EMA、验证与诊断

- Source 与 EMA anchor 都按五元组解包合成域数据，并使用同一物理 mask 损失接口。
- 验证同样解包第五项，但不使用 GT mask 解码或计算推理路由；验证使用 hard predicted route。
- 诊断图按以下顺序展示：

```text
hazy / TIR / clear / prediction /
M_GT / M_pred /
mask_GT / route_soft / route_hard
```

- Source 每个 epoch 后执行验证并保存 `source_last.pt`；验证 PSNR 创新高时保存 `source_best.pt`。EMA 同样按验证结果保存 last/best checkpoint。

## 7. 配置、兼容性与清理边界

保留的路由与物理监督配置包括：

```text
route_tau_start
route_tau_end
route_temperature_anneal_steps
route_teacher_anneal_steps
lambda_density
lambda_route
```

同时保留当前分区重建权重：`lambda_global`、`lambda_fuse`、`lambda_comp`、`lambda_boundary`，以及全局/区域内部权重。

不再存在或不允许进入当前主链路的内容包括：

```text
MonotonicFogRouter
OmegaSampler
compute_q / q 日志
counterfactual predictions 及其启动步数
route_hard 的 straight-through 训练
旧 binary / boundary / q 路由损失
Auto_Route_GT 的生成与可视化工具
```

checkpoint 预检必须拒绝包含旧 `MonotonicFogRouter` 或 old-route 配置的权重，并明确提示：`不兼容 V2 物理 mask 路由架构`。旧 checkpoint 不能静默迁移到当前方案。

## 8. 必要测试

当前方案的测试应覆盖：

1. 合成数据五元组、`mask_GT` 配对、二值化和最近邻增强；
2. HDE `routing_features` 的正式键、形状与路由器 195 通道输入；
3. FeatureGuidedRouter 对 HDE 特征的依赖、3×3 局部上下文、温度与 hard route；
4. 物理 `mask_GT` BCE，以及全零/全一 mask 的稳定性；
5. GT→预测路由的教师门控退火；
6. Source、EMA、验证和诊断图的端到端 smoke test；
7. old-route checkpoint 的明确拒绝。

## 9. 服务器训练前提

除代码和数据目录外，服务器必须具备：

- 项目内的 Res2Net RGB 编码器 ImageNet 预训练权重；
- torchvision VGG19 ImageNet 权重（CR 损失首次初始化时读取；无网络服务器需预置到 Torch 缓存）；
- 运行 EMA 时还需 `clip_model/` 内的 CLIP 权重与 `haze_prompt.pth`；
- 与项目兼容的 PyTorch、torchvision 和 CUDA 环境。

训练根目录必须满足第 2 节的目录结构与 stem 配对规则；首次部署建议运行一个小 batch、一个 epoch、少量 iteration 的 Source 冒烟训练，再启动正式训练。
