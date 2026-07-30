# Fog-routed RGB--TIR Dehazing

This project uses two training stages and one unified evaluation entrypoint.

- `Teacher.py` is the root Source-domain training entrypoint.
- `model/Teacher.py` defines the `FogRoutedRGBTIRDehazer` main network.
- `EMA.py` is the real-domain EMA adaptation entrypoint.
- `Eval.py` is the unified Source/EMA evaluation entrypoint, not a third training stage.
- `model/fog_routed_dehazer.py` does not exist.

## Formal Source training

```powershell
python Teacher.py --train_data_dir <synthetic_root> --saved_model_dir <source_out> --formal_training
```

## Minimum smoke examples

These commands are smoke examples, not formal paper-training configurations:

```powershell
python Teacher.py --train_data_dir <synthetic_root> --saved_model_dir <source_out>
python EMA.py --source_checkpoint <source_out/source_last.pt> --source_anchor_data_dir <synthetic_root> --real_data_dir <real_root> --saved_model_dir <ema_out>
python Eval.py --checkpoint <source_or_ema_checkpoint> --hazy_dir <rgb_dir> --tir_dir <tir_dir> --output_dir <output_dir>
```

`Eval.py` detects `training_stage` automatically. For an EMA checkpoint, use `--ema_model student` or `--ema_model teacher` to choose weights.

## FLIR → M3FD 无监督域适应（UDA）

`UDA.py` 将 M3FD 明确作为无标注目标域：不生成 M3FD 的 clear、`M_GT` 或 `mask_GT`，不新增风格迁移预训练模型，也不改变硬路由的 0.5 阈值。

训练参数、数据路径、输出目录和探针路径都集中在 [option/UDA.py](option/UDA.py) 顶部的“Direct training configuration”区域。通常无需传终端参数：配置完成后直接运行 `python UDA.py`。Stage A 验收后，仅把 `UDA_RUN_STAGE` 改为 `"ema"`；程序会自动改用 `source_style_best.pt` 和 Stage B 输出目录。

先运行 Stage A。它仅用 M3FD RGB/TIR 的全局统计对 FLIR 的 `hazy/clear/tir` 做有界同步变换，FLIR 的五元组物理监督保持不变：

```powershell
D:\anaconda\envs\CoA\python.exe UDA.py --stage source_style `
  --source_checkpoint F:\1_paper_pan\1_Dehaze_Paper\1_pth_model_duibi_experiments\1_Ours\pth\v2\source_best.pt `
  --source_anchor_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\train `
  --validation_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\test `
  --real_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD `
  --real_tir_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\ir `
  --probe_hazy F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\hazy\00343.png `
  --probe_tir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\ir\00343.png `
  --probe_output_dir <stage_a_out>\probe --saved_model_dir <stage_a_out>
```

Stage A 生成 `source_style_best.pt`/`source_style_last.pt`。先检查 FLIR 验证指标与 `probe` 内的 `pred_clear`、`density_map`、`route_soft`、`route_hard`、`boundary_map`；当探针硬路由不再退化为全零且图像没有大面积伪影，再运行 Stage B：

```powershell
D:\anaconda\envs\CoA\python.exe UDA.py --stage ema `
  --source_checkpoint <stage_a_out>\source_style_best.pt `
  --source_anchor_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\train `
  --validation_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\test `
  --real_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD `
  --real_tir_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\ir `
  --probe_hazy F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\hazy\00343.png `
  --probe_tir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\ir\00343.png `
  --probe_output_dir <stage_b_out>\probe --saved_model_dir <stage_b_out>
```

Stage B 输出 `uda_ema_best.pt`/`uda_ema_last.pt`。在 `route_consistency_warmup_steps` 内，目标域 `L_R` 为零，但图像/密度一致性、CLIP 项和 FLIR Source-anchor 损失始终启用；随后 `L_R` 线性升至设定权重。结果应标注为“M3FD 无监督域适应”，不是纯合成域零样本泛化。

The synthetic root contains `clear/`, `ir/`, `hazy/<level>/`, and `Transmission_Map_GT/<level>/`. Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`; files pair by case-insensitive stem. The real root contains `hazy/` and `tir/`.

## Structure--appearance Transformer decoder

The single decoder is a multi-scale structure--appearance cross-attention Transformer decoder. At each scale, the routed structural feature initializes or updates the decoder query state, while the assembled appearance feature serves as Key and Value. h16 starts from routed structure; h8/h4/h2 add the projected, upsampled deeper decoded state to their structural query state. TIR content is never passed directly as a Transformer Value. In completion regions, appearance is assembled by continuously blending RGB appearance retrieved from reliable fusion memory with a TIR-conditioned appearance prior; the prior dominates when reliable memory is absent or retrieval confidence is low.

Each stage uses pre-norm multi-head window cross-attention, learnable 2-D relative-position bias, residual connections and GELU FFN blocks. Windows are right/bottom padded with validity masks and processed in configurable batches (`--decoder_window_chunk_size`), giving bounded local attention rather than a high-resolution global `HW x HW` matrix.

Memory retrieval uses the continuous `memory_retrieval_gate` to blend retrieved RGB appearance with the TIR-conditioned prior. `memory_fallback_mask` is diagnostic only. Formal checkpoints use `format_version=2` and store all decoder configuration. Historical checkpoints cannot be resumed, used to initialize EMA, or evaluated by formal entrypoints; `tools/inspect_legacy_checkpoint.py` remains read-only and reports static v2 compatibility without attempting a model load. Static v2 compatibility does not guarantee successful strict model loading.
