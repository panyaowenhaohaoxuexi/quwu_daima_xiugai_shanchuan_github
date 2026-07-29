import pytest

from option.Teacher import build_parser, validate_config


def test_v2_parser_exposes_only_physical_mask_route_controls():
    args = validate_config(build_parser().parse_args([]))
    assert args.density_gt_semantics == "density"
    assert args.lambda_density == args.lambda_route == 1.0
    for removed in ("counterfactual_start_step", "q_temperature", "lambda_binary", "boundary_width"):
        assert not hasattr(args, removed)


@pytest.mark.parametrize("name", ("lambda_density", "lambda_route"))
def test_v2_parser_rejects_negative_loss_weights(name):
    with pytest.raises(ValueError, match="non-negative"):
        validate_config(build_parser().parse_args([f"--{name}", "-0.1"]))
