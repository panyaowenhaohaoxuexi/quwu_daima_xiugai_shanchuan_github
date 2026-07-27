import torch
import inspect
from torch.nn import functional as F

from model.Teacher import MemoryRetriever, FogRoutedRGBTIRDehazer, appearance_receptive_field_radius_by_scale


def test_empty_memory_fallback_prior_uses_global_tir_structure_context():
    memory = MemoryRetriever(
        channels=1, structure_channels=1, max_tokens=4, topk=2,
        attention_temperature=0.07, reliability_epsilon=1e-6,
        ratio_threshold=0.01, confidence_threshold=0.1,
    )
    with torch.no_grad():
        memory.prior.layers[0].weight.zero_()
        memory.prior.layers[0].weight[0, 1, 0, 0] = 1.0
        memory.prior.layers[0].bias.zero_()
        memory.prior.layers[2].weight.fill_(1.0)
        memory.prior.layers[2].bias.zero_()

    validity = torch.ones(1, 1, 3, 3)
    reliability = torch.zeros_like(validity)
    structure_a = torch.zeros_like(validity)
    structure_a[..., 1, 1] = 1.0
    structure_b = structure_a.clone()
    structure_b[..., 0, 0] = 2.0
    value_a = torch.zeros_like(validity)
    value_b = torch.full_like(validity, 17.0)

    appearance_a, _, fallback_a, _, _, candidate_count_a, _ = memory(
        structure_a, value_a, reliability, validity,
    )
    appearance_with_other_value, _, fallback_with_other_value, _, _, candidate_count_with_other_value, _ = memory(
        structure_a, value_b, reliability, validity,
    )
    appearance_b, _, fallback_b, _, _, candidate_count_b, _ = memory(
        structure_b, value_a, reliability, validity,
    )

    assert appearance_a.shape == value_a.shape
    assert torch.equal(fallback_a, torch.ones_like(fallback_a))
    assert torch.equal(fallback_with_other_value, torch.ones_like(fallback_with_other_value))
    assert torch.equal(fallback_b, torch.ones_like(fallback_b))
    assert torch.equal(candidate_count_a, torch.zeros_like(candidate_count_a))
    assert torch.equal(candidate_count_with_other_value, torch.zeros_like(candidate_count_with_other_value))
    assert torch.equal(candidate_count_b, torch.zeros_like(candidate_count_b))
    assert torch.allclose(appearance_a, appearance_with_other_value, atol=0, rtol=0)
    assert not torch.allclose(appearance_a[..., 1, 1], appearance_b[..., 1, 1], atol=0, rtol=0)


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
            "boundary_map", "memory_confidence", "memory_fallback_mask", "memory_retrieval_gate",
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
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
                   for parameter in model.decoder_stages[scale].parameters())


def test_transformer_stage_and_projection_hooks_keep_structure_query_and_appearance_key_value_separate():
    from training.source import _detach_context

    torch.manual_seed(41)
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    context = _detach_context(model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)))
    full = torch.ones(1, 1, 32, 32)
    stage = model.decoder_stages["h2"]
    block = stage.blocks[0]
    captured = {}
    handles = [
        stage.register_forward_pre_hook(lambda _module, values: captured.update(
            stage_structure=values[0].detach(), stage_appearance=values[1].detach(), stage_validity=values[2].detach())),
        block.structure_norm.register_forward_pre_hook(lambda _module, values: captured.update(
            structure_norm_raw=values[0].detach())),
        block.structure_norm.register_forward_hook(lambda _module, values, output: captured.update(q_norm=output.detach())),
        block.appearance_norm.register_forward_pre_hook(lambda _module, values: captured.update(
            appearance_norm_raw=values[0].detach())),
        block.appearance_norm.register_forward_hook(lambda _module, values, output: captured.update(kv_norm=output.detach())),
        block.q_proj.register_forward_pre_hook(lambda _module, values: captured.update(q_proj=values[0].detach())),
        block.k_proj.register_forward_pre_hook(lambda _module, values: captured.update(k_proj=values[0].detach())),
        block.v_proj.register_forward_pre_hook(lambda _module, values: captured.update(v_proj=values[0].detach())),
    ]
    try:
        output = model.decode_with_route(context, route_mode="hard", route_override_value=full,
                                         route_override_mask=full, memory_exclude_mask=full, return_debug=True)
    finally:
        for handle in handles:
            handle.remove()
    stage_validity = captured["stage_validity"]
    height, width = captured["stage_structure"].shape[-2:]
    pad_h, pad_w = (-height) % block.window_size, (-width) % block.window_size

    def window_tokens(feature):
        padded = F.pad(feature, (0, pad_w, 0, pad_h)) * F.pad(stage_validity, (0, pad_w, 0, pad_h))
        tokens = padded.permute(0, 2, 3, 1).reshape(padded.shape[0], -1, padded.shape[1])
        return block._partition(tokens, padded.shape[-2], padded.shape[-1])

    torch.testing.assert_close(captured["structure_norm_raw"], window_tokens(captured["stage_structure"]))
    torch.testing.assert_close(captured["appearance_norm_raw"], window_tokens(captured["stage_appearance"]))
    torch.testing.assert_close(captured["q_proj"], captured["q_norm"])
    torch.testing.assert_close(captured["k_proj"], captured["kv_norm"])
    torch.testing.assert_close(captured["v_proj"], captured["kv_norm"])
    torch.testing.assert_close(captured["stage_appearance"], output["debug"]["appearance_tokens"]["h2"])
    assert captured["stage_structure"].shape == output["debug"]["structure_tokens"]["h2"].shape


def test_content_only_completion_change_keeps_prior_and_transformer_key_value_inputs_identical():
    from training.source import _detach_context

    torch.manual_seed(43)
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    base = _detach_context(model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)))
    full = torch.ones(1, 1, 32, 32)

    def run(context):
        recorded = {}
        block = model.decoder_stages["h2"].blocks[0]
        handles = [
            model.memory["h2"].prior.register_forward_pre_hook(lambda _m, values: recorded.update(prior_input=values[0].detach())),
            model.memory["h2"].prior.register_forward_hook(lambda _m, values, output: recorded.update(prior_output=output.detach())),
            block.k_proj.register_forward_pre_hook(lambda _m, values: recorded.update(k=values[0].detach())),
            block.v_proj.register_forward_pre_hook(lambda _m, values: recorded.update(v=values[0].detach())),
        ]
        try:
            model.decode_with_route(context, route_mode="hard", route_override_value=full,
                                    route_override_mask=full, memory_exclude_mask=full)
        finally:
            for handle in handles:
                handle.remove()
        return recorded

    changed = _detach_context(base)
    changed["tir_content_pyramid"]["h2"] = changed["tir_content_pyramid"]["h2"] + 0.25
    reference, actual = run(base), run(changed)
    for key in ("prior_input", "prior_output", "k", "v"):
        torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)


def test_content_tir_has_a_deterministic_completion_to_query_projection_gradient_path():
    from training.source import _detach_context

    torch.manual_seed(53)
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    context = _detach_context(model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)))
    content = context["tir_content_pyramid"]["h2"].detach().clone().requires_grad_(True)
    context["tir_content_pyramid"]["h2"] = content
    full = torch.ones(1, 1, 32, 32)
    block = model.decoder_stages["h2"].blocks[0]
    recorded = {}
    handles = [
        block.q_proj.register_forward_hook(lambda _m, _values, output: recorded.update(q_output=output)),
    ]
    try:
        model.decode_with_route(context, route_mode="hard", route_override_value=full,
                                route_override_mask=full, memory_exclude_mask=full)
        recorded["q_output"].square().mean().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert content.grad is not None
    assert torch.isfinite(content.grad).all()
    assert content.grad.abs().sum() > 0


def test_structure_tir_has_deterministic_query_prior_and_key_value_gradient_paths():
    from training.source import _detach_context

    torch.manual_seed(47)
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    base = _detach_context(model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)))
    half = torch.full((1, 1, 32, 32), 0.5)
    full = torch.ones_like(half)

    for target_name, module_name in (("query", "q_proj"), ("prior", "prior"), ("key", "k_proj"), ("value", "v_proj")):
        context = _detach_context(base)
        structure = context["tir_structure_pyramid"]["h2"].detach().clone().requires_grad_(True)
        context["tir_structure_pyramid"]["h2"] = structure
        block, prior = model.decoder_stages["h2"].blocks[0], model.memory["h2"].prior
        recorded = {}
        target_module = prior if module_name == "prior" else getattr(block, module_name)
        handles = [
            target_module.register_forward_hook(lambda _m, _values, output: recorded.update(target=output)),
        ]
        if target_name in ("key", "value"):
            handles.extend((
                prior.register_forward_hook(lambda _m, _values, output: recorded.update(prior=output)),
                model.decoder_stages["h2"].register_forward_pre_hook(
                    lambda _m, values: recorded.update(stage_appearance=values[1])
                ),
            ))
        try:
            override = full if target_name in ("key", "value") else half
            output = model.decode_with_route(
                context, route_mode="hard", route_override_value=override,
                route_override_mask=full, memory_exclude_mask=full, return_debug=True,
            )
            if target_name in ("key", "value"):
                torch.testing.assert_close(recorded["stage_appearance"], recorded["prior"], atol=0, rtol=0)
                for scale in ("h16", "h8", "h4", "h2"):
                    assert torch.count_nonzero(output["debug"]["memory_attention_candidate_count"][scale]) == 0
                    assert torch.count_nonzero(output["debug"]["memory_stats_by_scale"][scale]["retrieval_gate"]) == 0
            recorded["target"].square().mean().backward()
        finally:
            for handle in handles:
                handle.remove()
        assert structure.grad is not None, target_name
        assert torch.isfinite(structure.grad).all(), target_name
        assert structure.grad.abs().sum() > 0, target_name


def test_full_completion_with_excluded_memory_has_no_rgb_feature_gradient():
    from training.source import _detach_context

    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    context = _detach_context(model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)))
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


def test_memory_query_chunks_match_unchunked_and_bound_score_rows(monkeypatch):
    torch.manual_seed(9)
    reference = FogRoutedRGBTIRDehazer(
        base_channels=8, memory_max_tokens=16, memory_topk=2, memory_query_chunk_size=10_000,
    ).eval()
    chunked = FogRoutedRGBTIRDehazer(
        base_channels=8, memory_max_tokens=16, memory_topk=2, memory_query_chunk_size=7,
    ).eval()
    chunked.load_state_dict(reference.state_dict())
    hazy, tir = torch.rand(1, 3, 31, 47), torch.rand(1, 3, 31, 47)
    max_rows = []
    original_topk = torch.topk

    def checked_topk(value, *args, **kwargs):
        if value.ndim == 2:
            max_rows.append(value.shape[0])
        return original_topk(value, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", checked_topk)
    actual = chunked(hazy, tir, route_mode="soft")
    actual_max_rows = max(max_rows)
    monkeypatch.setattr(torch, "topk", original_topk)
    expected = reference(hazy, tir, route_mode="soft")

    assert actual_max_rows <= 7
    for key in ("pred_clear", "memory_confidence", "memory_fallback_mask", "memory_retrieval_gate", "memory_reliable_mass", "memory_reliable_ratio"):
        assert torch.allclose(actual[key], expected[key], atol=1e-6)


def test_boundary_gradient_uses_effective_hard_override_route():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    context = model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    full = torch.ones(1, 1, 32, 32)
    seen = []
    handle = model.boundary["h2"][0].register_forward_pre_hook(lambda _m, values: seen.append(values[0][:, -1:].detach()))
    try:
        model.decode_with_route(
            context, route_mode="hard", boundary_mode="hard",
            route_override_value=torch.zeros_like(full), route_override_mask=full,
        )
    finally:
        handle.remove()

    assert seen
    assert torch.equal(seen[0], torch.zeros_like(seen[0]))


def test_memory_exclusion_radius_is_derived_from_the_actual_appearance_stack():
    radii = appearance_receptive_field_radius_by_scale(deform_max_offset=2.0, extra_margin=1)

    assert radii == {"h2": 9, "h4": 10, "h8": 11, "h16": 11}
