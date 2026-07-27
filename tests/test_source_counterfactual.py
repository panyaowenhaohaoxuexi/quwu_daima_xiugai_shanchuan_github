import torch

from model.fog_routed_dehazer import FogRoutedRGBTIRDehazer
from training.source_counterfactual import build_counterfactual_routes, detached_context, gather_context
from training.source_counterfactual import run_counterfactual_pair, run_counterfactual_chunks, compute_q


def test_counterfactual_routes_override_only_omega_and_context_is_detached():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    context = model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    omega = torch.zeros(1, 1, 32, 32)
    omega[..., 10:14, 10:14] = 1

    detached = detached_context(context)
    fuse, completion = build_counterfactual_routes(detached, omega)

    assert all(not value.requires_grad for value in detached.values() if torch.is_tensor(value))
    assert torch.equal(fuse["route_override_mask"], omega)
    assert torch.equal(completion["route_override_mask"], omega)
    assert float(fuse["route_override_value"].sum()) == 0.0
    assert float(completion["route_override_value"].min()) == 1.0


def test_counterfactual_pair_uses_detached_context_and_q_prefers_lower_completion_error():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    context = model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    omega = torch.zeros(1, 1, 32, 32)
    omega[..., 8:24, 8:24] = 1
    clear = torch.rand(1, 3, 32, 32)

    fuse, completion = run_counterfactual_pair(model, context, omega)
    q, _ = compute_q(clear, fuse["pred_clear"], completion["pred_clear"], omega, temperature=0.1)

    assert not fuse["pred_clear"].requires_grad
    assert not completion["pred_clear"].requires_grad
    assert not q.requires_grad
    assert torch.all((q >= 0) & (q <= 1))
    perfect_completion, _ = compute_q(clear, clear + 0.3, clear, omega, temperature=0.1)
    assert perfect_completion[omega.bool()].mean() > 0.5


def test_q_local_error_ignores_pixels_outside_omega_support():
    clear = torch.zeros(1, 3, 9, 9)
    omega = torch.zeros(1, 1, 9, 9)
    omega[..., 3:6, 3:6] = 1
    fusion = clear.clone()
    completion_a = clear.clone()
    completion_b = clear.clone()
    completion_b[..., 0, :] = 100.0
    completion_b[..., :, 0] = 100.0

    q_a, _ = compute_q(clear, fusion, completion_a, omega, 0.1, window_size=3, min_valid_support=2)
    q_b, _ = compute_q(clear, fusion, completion_b, omega, 0.1, window_size=3, min_valid_support=2)

    assert torch.allclose(q_a, q_b)


def test_q_ssim_term_is_finite_with_small_valid_support():
    clear = torch.zeros(1, 3, 5, 5)
    omega = torch.zeros(1, 1, 5, 5)
    omega[..., 2, 2] = 1
    q, q_valid = compute_q(clear, clear, clear, omega, 0.1, window_size=3,
                           min_valid_support=4, ssim_weight=1.0)

    assert torch.isfinite(q).all()
    assert q.sum() == 0
    assert q_valid.sum() == 0


def test_q_keeps_l1_supervision_when_ssim_or_gradient_is_invalid():
    clear = torch.zeros(1, 3, 5, 5)
    omega = torch.zeros(1, 1, 5, 5)
    omega[..., 2, 2] = 1
    fusion = clear + 0.4
    completion = clear

    q, valid = compute_q(
        clear, fusion, completion, omega, 0.1, window_size=3, min_valid_support=1,
        l1_weight=1.0, gradient_weight=1.0, ssim_weight=1.0,
    )

    assert valid[omega.bool()].all()
    assert q[omega.bool()].mean() > 0.5


def test_gather_context_selects_only_region_owners_without_reencoding():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    context = model.encode_context(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32))
    gathered = gather_context(detached_context(context), torch.tensor([1, 0, 1]))

    assert gathered["density_map"].shape[0] == 3
    assert torch.equal(gathered["density_map"][0], context["density_map"][1].detach())
    assert gathered["original_size"] == context["original_size"]


def test_counterfactual_chunks_match_single_gathered_decode_and_never_encode_again():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    context = model.encode_context(torch.rand(2, 3, 32, 32), torch.rand(2, 3, 32, 32))
    owners = torch.tensor([0, 1, 0])
    supports = torch.zeros(3, 1, 32, 32)
    supports[0, ..., 2:8, 2:8] = 1
    supports[1, ..., 8:14, 8:14] = 1
    supports[2, ..., 16:24, 16:24] = 1
    calls = {"encode": 0}
    original_encode = model.encode_context

    def counted_encode(*args, **kwargs):
        calls["encode"] += 1
        return original_encode(*args, **kwargs)

    model.encode_context = counted_encode
    fusion_all, completion_all = run_counterfactual_pair(model, gather_context(context, owners), supports)
    fusion_chunk, completion_chunk = run_counterfactual_chunks(model, context, owners, supports, chunk_size=1)

    assert calls["encode"] == 0
    assert torch.allclose(fusion_all["pred_clear"], fusion_chunk["pred_clear"], atol=1e-6)
    assert torch.allclose(completion_all["pred_clear"], completion_chunk["pred_clear"], atol=1e-6)


def test_counterfactual_decode_leaves_all_model_buffers_unchanged():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).train()
    context = model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    before = {name: value.detach().clone() for name, value in model.named_buffers()}
    omega = torch.zeros(1, 1, 32, 32)
    omega[..., 8:16, 8:16] = 1

    run_counterfactual_pair(model, context, omega, route_mode="hard")

    assert all(torch.equal(value, before[name]) for name, value in model.named_buffers())
