# Fog-routed RGB--TIR Dehazing

本工程采用两阶段训练和一个统一评估入口：

```powershell
python Teacher.py --train_data_dir <synthetic_root> --saved_model_dir <source_out>
python EMA.py --source_checkpoint <source_out/source_last.pt> --source_anchor_data_dir <synthetic_root> --real_data_dir <real_root> --saved_model_dir <ema_out>
python Eval.py --checkpoint <source_or_ema_checkpoint> --hazy_dir <rgb_dir> --tir_dir <tir_dir> --output_dir <output_dir>
```

`Eval.py` 根据 checkpoint 的 `training_stage` 自动识别 Source 或 EMA。评估 EMA checkpoint 时，可用 `--ema_model student` 或 `--ema_model teacher` 选择权重。

Source 合成数据根目录包含 `clear/`、`ir/`、`hazy/<level>/` 和 `Transmission_Map_GT/<level>/`。真实域数据根目录包含 `hazy/` 与 `tir/`，文件名按 stem 配对。
