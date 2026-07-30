import pytest

from option.UDA import build_parser, validate_config


def test_uda_parser_exposes_two_stages_and_target_style_defaults():
    args = build_parser().parse_args([])

    assert args.stage == "source_style"
    assert args.style_probability == 0.5
    assert args.style_beta_min == 0.0
    assert args.style_beta_max == 0.6
    assert args.route_consistency_warmup_steps == 1000
    assert args.route_consistency_ramp_steps == 1000


@pytest.mark.parametrize("argv, message", [
    (["--style_probability", "1.1"], "style_probability"),
    (["--style_beta_min", "0.8", "--style_beta_max", "0.2"], "style beta"),
    (["--style_min_gain", "1.5", "--style_max_gain", "1.0"], "style gain"),
    (["--route_consistency_warmup_steps", "-1"], "route consistency"),
])
def test_uda_rejects_invalid_style_and_route_schedule_values(argv, message):
    args = build_parser().parse_args(argv)

    with pytest.raises(ValueError, match=message):
        validate_config(args)
