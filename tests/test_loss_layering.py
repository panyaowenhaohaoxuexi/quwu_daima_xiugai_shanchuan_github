"""Regression coverage for canonical loss ownership and compatibility paths."""

import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch


def test_canonical_source_and_real_losses_match_fixed_pre_refactor_baseline():
    from loss.real.consistency import real_consistency_loss, stability_weights
    from loss.synthetic.counterfactual import compute_q
    from loss.synthetic.objective import compute_source_objective

    torch.manual_seed(31415)
    pred = torch.rand(1, 3, 4, 4, requires_grad=True)
    clear = torch.rand(1, 3, 4, 4)
    density = torch.rand(1, 1, 4, 4, requires_grad=True)
    density_gt = torch.rand(1, 1, 4, 4)
    route = torch.rand(1, 1, 4, 4, requires_grad=True)
    boundary, q, omega = torch.rand(1, 1, 4, 4), torch.rand(1, 1, 4, 4), torch.ones(1, 1, 4, 4)
    source = compute_source_objective(
        pred, clear, density, density_gt, route, boundary, q, omega,
        rec_gradient_weight=0.2, rec_ssim_weight=0.2, boundary_gradient_weight=0.5,
    )
    source["total"].backward()
    assert set(source) == {"total", "global", "fuse", "comp", "boundary", "density", "route", "binary"}
    torch.testing.assert_close(source["total"], torch.tensor(3.9984745979))
    torch.testing.assert_close(pred.grad.norm(), torch.tensor(0.9628067613))
    torch.testing.assert_close(density.grad.norm(), torch.tensor(0.2019856274))
    torch.testing.assert_close(route.grad.norm(), torch.tensor(21.4344787598))

    q_value, q_valid = compute_q(clear, clear + 0.2, clear - 0.1, omega, 0.1,
                                 window_size=3, min_valid_support=2, gradient_weight=0.5, ssim_weight=0.5)
    torch.testing.assert_close(q_value.sum(), torch.tensor(10.2569599152))
    assert q_valid.sum() == 16 and not q_value.requires_grad and not q_valid.requires_grad

    j = torch.rand(1, 3, 4, 4, requires_grad=True)
    jt = torch.rand(1, 3, 4, 4)
    m = torch.rand(1, 1, 4, 4, requires_grad=True)
    mt = torch.rand(1, 1, 4, 4)
    r = torch.rand(1, 1, 4, 4, requires_grad=True)
    rt = torch.rand(1, 1, 4, 4)
    weights = stability_weights(jt, jt + 0.1, mt, mt + 0.1, rt, rt + 0.1, 0.1, 0.1, 0.1, 0.05)
    real = real_consistency_loss(j, jt, m, mt, r, rt, *weights, lambda_j=1.1, lambda_m=0.9, lambda_r=0.8)
    real["L_real"].backward()
    assert set(real) == {"L_J", "L_M", "L_R", "L_real"}
    torch.testing.assert_close(real["L_real"], torch.tensor(1.3981907368))
    torch.testing.assert_close(j.grad.norm(), torch.tensor(0.1587713212))
    torch.testing.assert_close(m.grad.norm(), torch.tensor(0.2249999940))
    torch.testing.assert_close(r.grad.norm(), torch.tensor(4.1881737709))


def test_compatibility_modules_reexport_canonical_loss_functions_without_wrappers():
    from loss.common.masked import masked_mean
    from loss.fog_routed_source_loss import masked_mean as legacy_masked_mean
    from loss.real.consistency import real_consistency_loss
    from loss.synthetic.counterfactual import compute_q
    from loss.synthetic.objective import compute_source_objective
    from training.ema_core import real_consistency_loss as legacy_real_consistency_loss
    from training.source_counterfactual import compute_q as legacy_compute_q
    from training.source_objective import compute_source_objective as legacy_compute_source_objective

    assert legacy_masked_mean is masked_mean
    assert legacy_compute_q is compute_q
    assert legacy_compute_source_objective is compute_source_objective
    assert legacy_real_consistency_loss is real_consistency_loss
    for function in (masked_mean, compute_q, compute_source_objective, real_consistency_loss):
        assert "loss" in Path(inspect.getsourcefile(function)).parts


def test_loss_modules_have_no_forbidden_upstream_dependencies():
    loss_root = Path(__file__).parents[1] / "loss"
    forbidden = {"model", "training", "option", "Teacher", "EMA"}
    for path in loss_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not names.intersection(forbidden), path


def test_adaptation_objective_keeps_detached_ema_step_result_contract():
    from loss.real.objective import compute_adaptation_objective

    real = torch.tensor(2.0, requires_grad=True)
    source = torch.tensor(3.0, requires_grad=True)
    result = compute_adaptation_objective(real, source, lambda_anchor=0.5)

    assert set(result) == {"L_real", "L_src", "L_adapt"}
    assert result["L_adapt"] == 3.5
    assert result["L_adapt"].requires_grad


def test_source_and_ema_loss_chain_smoke_can_backpropagate_without_committing_teacher(monkeypatch):
    from model import FogRoutedRGBTIRDehazer
    from loss.real.consistency import real_consistency_loss, stability_weights
    from training.ema_step import run_ema_adaptation_step
    from training.omega_sampler import OmegaSampler
    from training.source_step import compute_source_batch_losses
    import training.ema_step as ema_step

    args = SimpleNamespace(
        route_tau_start=1.0, route_tau_end=0.2, route_hard_start_step=1,
        counterfactual_start_step=0, route_loss_start_step=0, route_loss_warmup_steps=0,
        binary_loss_start_step=0, binary_loss_warmup_steps=0, lambda_route=1.0,
        lambda_binary=1.0, q_temperature=0.1, counterfactual_chunk_size=2,
        density_smooth_l1_beta=0.1, lambda_global=1.0, lambda_fuse=1.0,
        lambda_comp=1.0, lambda_boundary=1.0, lambda_router=1.0,
        lambda_density=1.0, rec_l1_weight=1.0, rec_gradient_weight=0.0,
        rec_ssim_weight=0.0, boundary_l1_weight=1.0, boundary_gradient_weight=0.0,
        reconstruction_ssim_window=3, reconstruction_min_valid_support=2,
    )
    student = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    teacher = copy.deepcopy(student).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    source_batch = tuple(torch.rand(1, channels, 32, 32) for channels in (3, 3, 3, 1))
    real_hazy, real_tir = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    sampler = OmegaSampler(regions_per_image=4, min_area=4, max_area=16, seed=7, edge_threshold=1.0)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    anchor = {}

    def real_loss_fn():
        with torch.no_grad():
            target = teacher(real_hazy, real_tir, route_mode="hard")
        output = student(real_hazy, real_tir, route_mode="hard")
        weights = stability_weights(
            target["pred_clear"], target["pred_clear"], target["density_map"], target["density_map"],
            target["route_soft"], target["route_soft"], 0.1, 0.1, 0.1, 0.05,
        )
        return real_consistency_loss(
            output["pred_clear"], target["pred_clear"], output["density_map"], target["density_map"],
            output["route_soft"], target["route_soft"], *weights,
        )["L_real"]

    def anchor_loss_fn():
        anchor["result"] = compute_source_batch_losses(
            student, source_batch, args, sampler, global_step=1, force_anchor_mode=True,
        )
        return anchor["result"]["losses"]["total"]

    monkeypatch.setattr(ema_step, "perform_optimizer_step", lambda *_args, **_kwargs: False)
    result = run_ema_adaptation_step(
        student, teacher, optimizer, real_loss_fn, anchor_loss_fn, lambda_anchor=1.0, ema_decay=0.9,
    )

    assert result["step_succeeded"] is False
    assert set(result) == {"L_real", "L_src", "L_adapt", "step_succeeded"}
    assert all(torch.is_tensor(result[key]) and not result[key].requires_grad for key in ("L_real", "L_src", "L_adapt"))
    assert not anchor["result"]["q"].requires_grad
    assert all(torch.equal(value, teacher_before[name]) for name, value in teacher.state_dict().items())
