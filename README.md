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

The synthetic root contains `clear/`, `ir/`, `hazy/<level>/`, and `Transmission_Map_GT/<level>/`. Supported image extensions are `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and `.tiff`; files pair by case-insensitive stem. The real root contains `hazy/` and `tir/`.

## Structure--appearance Transformer decoder

The single decoder is a multi-scale structure--appearance cross-attention Transformer decoder. At each scale, the routed structural feature initializes or updates the decoder query state, while the assembled appearance feature serves as Key and Value. h16 starts from routed structure; h8/h4/h2 add the projected, upsampled deeper decoded state to their structural query state. TIR content is never passed directly as a Transformer Value. In completion regions, appearance is assembled by continuously blending RGB appearance retrieved from reliable fusion memory with a TIR-conditioned appearance prior; the prior dominates when reliable memory is absent or retrieval confidence is low.

Each stage uses pre-norm multi-head window cross-attention, learnable 2-D relative-position bias, residual connections and GELU FFN blocks. Windows are right/bottom padded with validity masks and processed in configurable batches (`--decoder_window_chunk_size`), giving bounded local attention rather than a high-resolution global `HW x HW` matrix.

Memory retrieval uses the continuous `memory_retrieval_gate` to blend retrieved RGB appearance with the TIR-conditioned prior. `memory_fallback_mask` is diagnostic only. Formal checkpoints use `format_version=2` and store all decoder configuration. Historical checkpoints cannot be resumed, used to initialize EMA, or evaluated by formal entrypoints; `tools/inspect_legacy_checkpoint.py` remains read-only and reports static v2 compatibility without attempting a model load. Static v2 compatibility does not guarantee successful strict model loading.
