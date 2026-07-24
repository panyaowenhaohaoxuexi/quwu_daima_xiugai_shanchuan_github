from option.Teacher import build_parser


def test_formal_source_options_expose_route_and_memory_controls():
    args = build_parser().parse_args([])
    assert args.route_tau_start > 0
    assert args.route_tau_end > 0
    assert args.memory_topk >= 2
    assert args.num_structure_renderers >= 1
