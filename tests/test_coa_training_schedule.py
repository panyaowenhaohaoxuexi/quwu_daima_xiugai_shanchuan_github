import pytest
import torch


def test_coa_source_and_ema_schedule_defaults():
    from option.EMA import build_parser as build_ema_parser
    from option.Teacher import build_parser as build_teacher_parser

    source = build_teacher_parser().parse_args([])
    ema = build_ema_parser().parse_args([])

    assert (source.epochs, source.iters_per_epoch) == (20, 5000)
    assert (source.start_lr, source.end_lr, source.no_lr_sche) == (1e-4, 1e-6, False)
    assert (ema.epochs, ema.iters_per_epoch) == (20, 1000)
    assert (ema.start_lr, ema.end_lr, ema.no_lr_sche, ema.ema_decay) == (1e-7, 1e-8, False, 0.95)


def test_coa_cosine_schedule_reaches_start_and_end_rates():
    from training.schedule import cosine_decay_lr

    assert cosine_decay_lr(0, 100_000, 1e-4, 1e-6) == pytest.approx(1e-4)
    assert cosine_decay_lr(100_000, 100_000, 1e-4, 1e-6) == pytest.approx(1e-6)
    assert cosine_decay_lr(50_000, 100_000, 1e-4, 1e-6) == pytest.approx(5.05e-5)


def test_cycling_iterator_restarts_short_loader_without_losing_order():
    from training.schedule import cycle_batches

    batches = cycle_batches(["a", "b"])
    assert [next(batches) for _ in range(5)] == ["a", "b", "a", "b", "a"]


def test_coa_optimizer_is_adam_without_adamw_weight_decay():
    from training.schedule import build_coa_adam

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = build_coa_adam([parameter], learning_rate=1e-4)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0
