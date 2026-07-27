import numpy as np
import torch
from PIL import Image


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.allclose(actual, expected, atol=0, rtol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for value, expected_value in zip(actual, expected):
            _assert_nested_equal(value, expected_value)
    else:
        assert actual == expected


def _named_buffers(module):
    return {name: buffer.detach().cpu().clone() for name, buffer in module.named_buffers()}


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_ema_entrypoint_runs_real_and_source_anchor_from_strict_source_checkpoint(tmp_path):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist", "real/hazy", "real/tir"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", 90)
    _write_rgb(tmp_path / "ir" / "sample.png", 40)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", 120)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.png"
    )
    _write_rgb(tmp_path / "real" / "hazy" / "real.png", 100)
    _write_rgb(tmp_path / "real" / "tir" / "real.png", 55)

    from model import FogRoutedRGBTIRDehazer
    from option.Teacher import build_parser as build_source_parser
    from training.checkpointing import build_source_checkpoint, capture_rng_state
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    source_args = build_source_parser().parse_args([
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2", "--train_size", "32",
        "--counterfactual_chunk_size", "6",
    ])
    source_checkpoint = tmp_path / "source.pt"
    torch.save(build_source_checkpoint(
        model.state_dict(), torch.optim.AdamW(model.parameters()).state_dict(), None, 0, 0,
        vars(source_args), "transmission", capture_rng_state(),
    ), source_checkpoint)

    from EMA import main
    checkpoint_dir = tmp_path / "ema-checkpoints"
    main([
        "--source_checkpoint", str(source_checkpoint), "--source_anchor_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--epochs", "1", "--device", "cpu",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "ema-experiment"),
    ])
    checkpoint = torch.load(checkpoint_dir / "ema_last.pt", map_location="cpu")
    assert checkpoint["training_stage"] == "ema"
    assert checkpoint["ema_global_step"] == 1
    assert {"student", "teacher", "source_global_step", "ema_global_step"}.issubset(checkpoint)
    assert checkpoint["rng_state"]["omega_generator"] is not None
    assert checkpoint["rng_state"]["geometry_generator"] is not None

    # Both independent real/source cursors are at epoch boundaries after one
    # successful step. EMA resume must advance both and perform the next step.
    main([
        "--resume_checkpoint", str(checkpoint_dir / "ema_last.pt"), "--source_anchor_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--epochs", "2", "--learning_rate", "0.0003", "--device", "cpu",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "ema-resume"),
    ])
    resumed = torch.load(checkpoint_dir / "ema_last.pt", map_location="cpu")
    assert resumed["ema_global_step"] == 2
    assert resumed["optimizer"]["param_groups"][0]["lr"] == 0.0003


def test_ema_failure_after_full_forward_backward_replays_transaction_exactly(tmp_path, monkeypatch):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist", "real/hazy", "real/tir"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", 90)
    _write_rgb(tmp_path / "ir" / "sample.png", 40)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", 120)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.png"
    )
    _write_rgb(tmp_path / "real" / "hazy" / "real.png", 100)
    _write_rgb(tmp_path / "real" / "tir" / "real.png", 55)
    from model import FogRoutedRGBTIRDehazer
    from option.Teacher import build_parser as build_source_parser
    from training.checkpointing import build_source_checkpoint, capture_rng_state
    import EMA

    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    source_args = build_source_parser().parse_args([
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2", "--train_size", "32",
        "--counterfactual_chunk_size", "6",
    ])
    source_checkpoint = tmp_path / "source.pt"
    torch.save(build_source_checkpoint(
        model.state_dict(), torch.optim.AdamW(model.parameters()).state_dict(), None, 0, 0,
        vars(source_args), "transmission", capture_rng_state(),
    ), source_checkpoint)
    common = [
        "--source_checkpoint", str(source_checkpoint), "--source_anchor_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--epochs", "1", "--device", "cpu",
    ]
    reference_dir = tmp_path / "reference"
    EMA.main([*common, "--saved_model_dir", str(reference_dir), "--exp_dir", str(tmp_path / "reference-exp")])
    reference = torch.load(reference_dir / "ema_last.pt", map_location="cpu")
    import training.ema_step as ema_step

    original_perform_optimizer_step = ema_step.perform_optimizer_step
    original_update_teacher_after_success = ema_step.update_teacher_after_success
    original_run_ema_views = EMA.run_ema_views
    original_compute_source_batch_losses = EMA.compute_source_batch_losses
    calls = {"count": 0}
    teacher_update_calls = {"count": 0}
    geometry_records = []
    anchor_records = []
    modules = {}
    failed_step_buffers = {}

    def fail_once_after_backward(optimizer, parameters, **kwargs):
        parameters = list(parameters)
        calls["count"] += 1
        if calls["count"] == 1:
            assert any(
                parameter.grad is not None
                for parameter in parameters
                if parameter.requires_grad
            )
            failed_step_buffers.update({
                "student": _named_buffers(modules["student"]),
                "teacher": _named_buffers(modules["teacher"]),
            })
            return False
        return original_perform_optimizer_step(optimizer, parameters, **kwargs)

    def recorded_update_teacher_after_success(teacher, student, decay):
        teacher_update_calls["count"] += 1
        return original_update_teacher_after_success(teacher, student, decay)

    def recorded_run_ema_views(teacher, student, hazy_rgb, tir, generator, *args, **kwargs):
        modules["student"] = student
        modules["teacher"] = teacher
        state_before = generator.get_state().clone()
        record = {
            "before": state_before,
            "student_buffers": _named_buffers(student),
            "teacher_buffers": _named_buffers(teacher),
        }
        result = original_run_ema_views(teacher, student, hazy_rgb, tir, generator, *args, **kwargs)
        record["after"] = generator.get_state().clone()
        geometry_records.append(record)
        return result

    def recorded_source_step(model, source_batch, args, omega_sampler, global_step, **kwargs):
        generator = kwargs["omega_generator"]
        omega_before = generator.get_state().clone()
        result = original_compute_source_batch_losses(
            model, source_batch, args, omega_sampler, global_step, **kwargs,
        )
        anchor_records.append({
            "omega_before": omega_before,
            "omega_after": generator.get_state().clone(),
            "q": result["q"].detach().cpu().clone(),
            "q_valid_sum": result["q_valid_sum"].detach().cpu().clone(),
            "route_support": result["route_support"].detach().cpu().clone(),
            "omega_support": result["omega"]["omega_support"].detach().cpu().clone(),
            "omega_weight": result["omega"]["omega_weight"].detach().cpu().clone(),
            "owner_index": result["omega"]["owner_index"].detach().cpu().clone(),
            "route_supervision": dict(result["route_supervision"]),
        })
        return result

    monkeypatch.setattr(ema_step, "perform_optimizer_step", fail_once_after_backward)
    monkeypatch.setattr(ema_step, "update_teacher_after_success", recorded_update_teacher_after_success)
    monkeypatch.setattr(EMA, "run_ema_views", recorded_run_ema_views)
    monkeypatch.setattr(EMA, "compute_source_batch_losses", recorded_source_step)
    retry_dir = tmp_path / "retry"
    EMA.main([*common, "--saved_model_dir", str(retry_dir), "--exp_dir", str(tmp_path / "retry-exp")])
    retried = torch.load(retry_dir / "ema_last.pt", map_location="cpu")

    assert calls["count"] == 2
    assert teacher_update_calls["count"] == 1
    assert retried["ema_global_step"] == 1
    assert retried["source_global_step"] == reference["source_global_step"]
    assert retried["epoch"] == 1

    assert len(geometry_records) == 2
    _assert_nested_equal(geometry_records[0]["before"], geometry_records[1]["before"])
    _assert_nested_equal(geometry_records[0]["after"], geometry_records[1]["after"])
    _assert_nested_equal(geometry_records[0]["student_buffers"], geometry_records[1]["student_buffers"])
    _assert_nested_equal(geometry_records[0]["teacher_buffers"], geometry_records[1]["teacher_buffers"])
    assert failed_step_buffers

    assert len(anchor_records) == 2
    _assert_nested_equal(anchor_records[0], anchor_records[1])

    for key in ("student", "teacher", "optimizer", "sampler_states", "empty_omega_streaks"):
        _assert_nested_equal(retried[key], reference[key])
    for key in ("omega_generator", "geometry_generator"):
        _assert_nested_equal(retried["rng_state"][key], reference["rng_state"][key])
    for key in ("real", "source_anchor"):
        _assert_nested_equal(
            retried["rng_state"]["dataloader_generators"][key],
            reference["rng_state"]["dataloader_generators"][key],
        )
