import argparse


def test_source_resume_uses_checkpoint_semantics_and_current_runtime_values():
    from Teacher import resolve_source_resume_config
    from option.Teacher import build_parser, persisted_config_from_args

    checkpoint_config = persisted_config_from_args(build_parser().parse_args([]))
    checkpoint_config.update({"base_channels": 12, "train_size": 64, "lambda_router": 0.37})
    raw = build_parser().parse_args([
        "--train_data_dir", "new-data", "--device", "cpu", "--epochs", "9",
        "--learning_rate", "0.002", "--batch_size", "3", "--num_workers", "2",
        "--saved_model_dir", "new-output", "--base_channels", "99", "--train_size", "96",
        "--lambda_router", "4.0",
    ])

    merged = resolve_source_resume_config(raw, checkpoint_config)

    assert merged["base_channels"] == 12
    assert merged["train_size"] == 64
    assert merged["lambda_router"] == 0.37
    assert merged["train_data_dir"] == "new-data"
    assert merged["device"] == "cpu"
    assert merged["epochs"] == 9
    assert merged["learning_rate"] == 0.002
    assert merged["batch_size"] == 3
    assert merged["num_workers"] == 2
    assert merged["saved_model_dir"] == "new-output"


def test_source_resume_does_not_reset_formal_checkpoint_loss_weights():
    from Teacher import resolve_source_resume_config
    from option.Teacher import build_parser, persisted_config_from_args, validate_config

    checkpoint_config = persisted_config_from_args(validate_config(build_parser().parse_args([
        "--formal_training", "--q_l1_weight", "1.7", "--rec_ssim_weight", "0.9",
    ])))
    raw = build_parser().parse_args(["--resume_checkpoint", "source.pt"])
    merged = resolve_source_resume_config(raw, checkpoint_config)
    merged["_resume_checkpoint_semantics"] = True
    resumed = validate_config(argparse.Namespace(**merged))

    assert resumed.q_l1_weight == 1.7
    assert resumed.rec_ssim_weight == 0.9


def test_ema_resume_keeps_checkpoint_ema_semantics_and_current_runtime_values():
    from option.EMA import build_parser, resolve_ema_config
    from option.Teacher import build_parser as build_source_parser, persisted_config_from_args

    checkpoint_config = persisted_config_from_args(build_source_parser().parse_args([]))
    checkpoint_config.update(vars(build_parser().parse_args([])))
    checkpoint_config.update({"ema_decay": 0.91, "lambda_anchor": 0.23, "lambda_ema_j": 0.42})
    raw = build_parser().parse_args([
        "--real_data_dir", "new-real", "--source_anchor_data_dir", "new-source",
        "--device", "cpu", "--epochs", "5", "--learning_rate", "0.003",
        "--ema_decay", "0.5", "--lambda_anchor", "2.0",
    ])

    merged = resolve_ema_config(raw, checkpoint_config, resume=True)

    assert merged["ema_decay"] == 0.91
    assert merged["lambda_anchor"] == 0.23
    assert merged["lambda_ema_j"] == 0.42
    assert merged["real_data_dir"] == "new-real"
    assert merged["source_anchor_data_dir"] == "new-source"
    assert merged["device"] == "cpu"
    assert merged["epochs"] == 5
    assert merged["learning_rate"] == 0.003
