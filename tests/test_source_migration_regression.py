from types import SimpleNamespace

import torch


def _args():
    return SimpleNamespace(route_tau_start=1.0, route_tau_end=0.2, route_hard_start_step=1,
        counterfactual_start_step=0, route_loss_start_step=0, route_loss_warmup_steps=0,
        binary_loss_start_step=0, binary_loss_warmup_steps=0, lambda_route=1.0, lambda_binary=1.0,
        q_temperature=0.1, counterfactual_chunk_size=2, density_smooth_l1_beta=0.1,
        lambda_global=1.0, lambda_fuse=1.0, lambda_comp=1.0, lambda_boundary=1.0,
        lambda_router=1.0, lambda_density=1.0, rec_l1_weight=1.0, rec_gradient_weight=0.0,
        rec_ssim_weight=0.0, boundary_l1_weight=1.0, boundary_gradient_weight=0.0,
        reconstruction_ssim_window=3, reconstruction_min_valid_support=2, q_window_size=1,
        q_min_valid_support=1, q_l1_weight=1.0, q_gradient_weight=0.0, q_ssim_weight=0.0)


def test_source_batch_migration_matches_loss_q_route_and_gradients():
    from model import FogRoutedRGBTIRDehazer
    from training.omega_sampler import OmegaSampler as LegacyOmega
    from training.source import OmegaSampler, compute_source_batch_losses
    from training.source_step import compute_source_batch_losses as legacy_step

    torch.manual_seed(73)
    legacy_model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    migrated_model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    migrated_model.load_state_dict(legacy_model.state_dict(), strict=True)
    batch = tuple(torch.rand(1, channels, 32, 32) for channels in (3, 3, 3, 1))
    generator_state = torch.Generator().manual_seed(5).get_state()
    legacy_generator, migrated_generator = torch.Generator(), torch.Generator()
    legacy_generator.set_state(generator_state); migrated_generator.set_state(generator_state)
    legacy = legacy_step(legacy_model, batch, _args(), LegacyOmega(4, 4, 16, seed=3, edge_threshold=1.0), 1,
                         omega_generator=legacy_generator)
    migrated = compute_source_batch_losses(migrated_model, batch, _args(), OmegaSampler(4, 4, 16, seed=3, edge_threshold=1.0), 1,
                                           omega_generator=migrated_generator)
    for key in legacy["losses"]:
        torch.testing.assert_close(legacy["losses"][key], migrated["losses"][key], rtol=0, atol=0)
    for key in ("q", "q_valid_sum", "route_support"):
        torch.testing.assert_close(legacy[key], migrated[key], rtol=0, atol=0)
    legacy["losses"]["total"].backward(); migrated["losses"]["total"].backward()
    for (name, old), (_, new) in zip(legacy_model.named_parameters(), migrated_model.named_parameters()):
        if old.grad is not None:
            torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0, msg=name)
