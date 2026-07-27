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
