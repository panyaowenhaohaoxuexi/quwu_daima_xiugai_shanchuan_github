import torch
import inspect

from model.fog_routed_dehazer import FogRoutedRGBTIRDehazer


def test_tiny_model_cpu_supports_split_context_and_original_output_size():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    hazy = torch.rand(1, 3, 31, 47, requires_grad=True)
    tir = torch.rand(1, 3, 31, 47)

    context = model.encode_context(hazy, tir, route_temperature=0.8)
    soft = model.decode_with_route(context, route_mode="soft")
    hard = model.decode_with_route(context, route_mode="hard")
    full = model(hazy, tir, route_temperature=0.8, route_mode="hard")

    expected_spatial = (31, 47)
    for output in (soft, hard, full):
        for key in (
            "pred_clear", "density_map", "route_logits", "route_soft", "route_hard",
            "boundary_map", "memory_confidence", "memory_fallback_mask",
        ):
            assert output[key].shape[-2:] == expected_spatial
        assert output["memory_reliable_mass"].shape == (1,)
        assert output["memory_reliable_ratio"].shape == (1,)
        assert torch.isfinite(output["pred_clear"]).all()
        assert float(output["pred_clear"].min()) >= 0.0
        assert float(output["pred_clear"].max()) <= 1.0

    full["pred_clear"].mean().backward()
    assert hazy.grad is not None
    assert torch.isfinite(hazy.grad).all()


def test_counterfactual_memory_exclusion_is_dilated_at_memory_scale():
    model = FogRoutedRGBTIRDehazer(
        base_channels=8, memory_max_tokens=16, memory_topk=2,
        deform_max_offset=2.0, memory_exclusion_extra_margin=1,
    )
    hazy, tir = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    context = model.encode_context(hazy, tir)
    full = torch.ones(1, 1, 32, 32)
    exclude = torch.zeros_like(full)
    exclude[..., 16, 16] = 1
    output = model.decode_with_route(
        context, route_mode="soft", route_override_value=torch.zeros_like(full),
        route_override_mask=full, memory_exclude_mask=exclude, return_debug=True,
    )

    reliability = output["debug"]["effective_reliable_mask"]["h2"]
    # One original pixel maps to one h2 token; the appearance receptive-field
    # exclusion must additionally remove neighbouring tokens.
    assert int((reliability == 0).sum()) > 1


def test_complete_cpu_tiny_smoke_backpropagates_through_all_formal_subsystems():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    model.train()
    for height, width in ((32, 32), (31, 47)):
        hazy = torch.rand(1, 3, height, width, requires_grad=True)
        tir = torch.rand(1, 3, height, width)
        context = model.encode_context(hazy, tir, route_temperature=0.7)
        soft = model.decode_with_route(context, route_mode="soft")
        hard = model.decode_with_route(context, route_mode="hard")
        assert soft["pred_clear"].shape[-2:] == (height, width)
        assert hard["route_logits"].shape[-2:] == (height, width)
        (soft["pred_clear"].mean() + hard["pred_clear"].mean() +
         hard["route_soft"].mean() + hard["density_map"].mean()).backward()

        for module_name in ("hde", "router", "rgb_encoder", "tir_encoder", "rgb_output_head"):
            module = getattr(model, module_name)
            assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
                       for parameter in module.parameters())
        model.zero_grad(set_to_none=True)


def test_unique_decoder_consumes_every_routed_structure_scale():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    output = model(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), route_mode="soft")
    output["pred_clear"].mean().backward()

    for scale in ("h2", "h4", "h8", "h16"):
        assert any(parameter.grad is not None for parameter in model.merge[scale].parameters())


def test_full_completion_with_excluded_memory_has_no_rgb_feature_gradient():
    from training.source_counterfactual import detached_context

    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    context = detached_context(model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)))
    context["rgb_pyramid"] = {
        name: feature.detach().clone().requires_grad_(True)
        for name, feature in context["rgb_pyramid"].items()
    }
    full = torch.ones(1, 1, 32, 32)
    output = model.decode_with_route(
        context, route_mode="hard", boundary_mode="soft",
        route_override_value=full, route_override_mask=full, memory_exclude_mask=full,
    )
    output["pred_clear"].sum().backward()

    for feature in context["rgb_pyramid"].values():
        assert feature.grad is None or torch.allclose(feature.grad, torch.zeros_like(feature.grad), atol=1e-8)


def test_formal_model_forward_exposes_no_clear_or_density_supervision_inputs():
    parameters = inspect.signature(FogRoutedRGBTIRDehazer.forward).parameters
    assert "clear_rgb" not in parameters
    assert "density_gt" not in parameters
    assert list(parameters)[:3] == ["self", "hazy_rgb", "tir"]


def test_debug_memory_statistics_are_scale_local_while_formal_statistics_are_h2():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    output = model(torch.rand(2, 3, 31, 47), torch.rand(2, 3, 31, 47), return_debug=True)

    stats = output["debug"]["memory_stats_by_scale"]
    assert set(stats) == {"h2", "h4", "h8", "h16"}
    assert torch.equal(output["memory_reliable_mass"], stats["h2"]["reliable_mass"])
    assert torch.equal(output["memory_reliable_ratio"], stats["h2"]["reliable_ratio"])
