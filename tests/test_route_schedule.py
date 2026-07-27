import pytest

from training.source import source_route_schedule


def test_source_route_schedule_enforces_counterfactual_and_route_loss_contract():
    early = source_route_schedule(
        0,
        tau_start=1.0,
        tau_end=0.2,
        hard_start_step=10,
        counterfactual_start_step=5,
        route_loss_start_step=5,
        route_loss_warmup_steps=10,
        binary_loss_start_step=20,
        binary_loss_warmup_steps=10,
        base_lambda_route=2.0,
        base_lambda_binary=3.0,
    )
    assert early["execute_counterfactual"] is False
    assert early["lambda_route"] == 0.0
    assert early["route_mode"] == "soft"

    started = source_route_schedule(
        10,
        tau_start=1.0,
        tau_end=0.2,
        hard_start_step=10,
        counterfactual_start_step=5,
        route_loss_start_step=5,
        route_loss_warmup_steps=10,
        binary_loss_start_step=20,
        binary_loss_warmup_steps=10,
        base_lambda_route=2.0,
        base_lambda_binary=3.0,
    )
    assert started["execute_counterfactual"] is True
    assert started["route_mode"] == "hard"
    assert started["lambda_route"] == pytest.approx(1.0)
    assert started["lambda_binary"] == 0.0


def test_counterfactual_cannot_start_after_route_loss():
    with pytest.raises(ValueError, match="counterfactual"):
        source_route_schedule(
            0, tau_start=1.0, tau_end=0.2, hard_start_step=1,
            counterfactual_start_step=2, route_loss_start_step=1,
            route_loss_warmup_steps=1, binary_loss_start_step=2,
            binary_loss_warmup_steps=1, base_lambda_route=1.0,
            base_lambda_binary=1.0,
        )
