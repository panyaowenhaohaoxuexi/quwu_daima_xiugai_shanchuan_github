from argparse import Namespace

from option.EMA import build_parser, resolve_ema_config, validate_config
from option.Teacher import build_parser as build_source_parser, persisted_config_from_args, validate_config as validate_source


def test_ema_inherits_only_v2_source_mask_routing_semantics():
    source = persisted_config_from_args(validate_source(build_source_parser().parse_args([])))
    raw = build_parser().parse_args(["--real_data_dir", "real", "--source_anchor_data_dir", "anchor"])
    resolved = validate_config(Namespace(**resolve_ema_config(raw, source)))
    assert resolved.density_gt_semantics == "density"
    assert resolved.lambda_route == source["lambda_route"]
    assert not hasattr(raw, "counterfactual_start_step")
