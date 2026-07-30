# M3FD UDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Starting from a V2 FLIR Source checkpoint, implement target-statistics-guided supervised fine-tuning followed by M3FD unsupervised EMA adaptation without adding data, pretrained style models, pseudo physical labels, or route-threshold overrides.

**Architecture:** Add a focused target-statistics transform that uses unlabeled M3FD RGB/TIR only as global photometric references while retaining FLIR `density_gt` and `completion_mask_gt`. Add a UDA entry point with two explicit stages: `source_style` saves a compatible Source checkpoint; `ema` initializes from it, keeps a Source physical-loss anchor, and ramps real-domain route consistency only after an explicit warm-up.

**Tech Stack:** Python, PyTorch, existing V2 `SynthMultiModalDataset`/`RealMultiModalDataset`, existing Source/EMA losses, pytest.

---

## Locked design decisions

- M3FD contributes only paired `(hazy, tir)` reference batches. No M3FD image is treated as `clear`, `M_GT`, or `mask_GT`.
- A style transform is a bounded per-image affine photometric mapping. It uses no learned generator, no image-to-image pixel copying, and no spatial warp.
- For RGB, the same mapping calculated from source `hazy` and target `hazy` statistics is applied to FLIR `hazy` and its paired `clear`. For TIR, a separate one-channel monotonic mapping is applied.
- `density_gt` and `completion_mask_gt` remain bit-identical after style augmentation. Existing synchronized geometry remains responsible for their resize/crop/flip/rotation.
- Stage A samples original and styled FLIR examples probabilistically, retains the existing physical Source loss, and emits normal `training_stage="source"` checkpoints so Stage B can reuse strict V2 checkpoint loading.
- Stage B uses existing image/density consistency and Source-anchor loss from the start. It disables real route consistency during `route_consistency_warmup_steps`, then linearly ramps it to the configured `lambda_ema_r`.
- M3FD participation must be recorded as UDA, not Source-only evaluation. Fixed M3FD probe images are diagnostic only; no target PSNR/SSIM claim is made.

## File map

| File | Change | Responsibility |
|---|---|---|
| `training/target_style.py` | Create | Bounded RGB/TIR target-statistics transform and validation helpers. |
| `training/real_adaptation.py` | Create | Shared geometry-equivariant real EMA loss with an explicit route-consistency multiplier. |
| `option/UDA.py` | Create | Two-stage UDA CLI, checkpoint config resolution, and validation. |
| `UDA.py` | Create | Stage A Source-style loop and Stage B EMA loop; checkpoint/log/probe orchestration. |
| `EMA.py` | Modify | Import the shared real adaptation helper; preserve current behavior by passing route multiplier `1.0`. |
| `tests/test_target_style.py` | Create | Unit tests for source/target statistics transforms. |
| `tests/test_real_adaptation.py` | Create | Unit tests for zero/ramped route consistency. |
| `tests/test_uda_options.py` | Create | CLI/config/checkpoint validation tests. |
| `tests/test_uda_smoke.py` | Create | Mocked two-stage integration test, probe output, and checkpoint compatibility. |
| `Codex/2026-07-30-m3fd-unsupervised-domain-adaptation-design.md` | Modify | Link the eventual executable and documented UDA protocol. |

### Task 1: Target-statistics transform and its contract

**Files:**
- Create: `training/target_style.py`
- Test: `tests/test_target_style.py`

- [ ] **Step 1: Write failing tests for identity, label preservation, and monotonic TIR behavior.**

```python
import torch
from training.target_style import apply_target_statistics


def test_beta_zero_is_strict_identity_and_leaves_labels_unchanged():
    hazy = torch.rand(2, 3, 8, 10)
    clear = torch.rand(2, 3, 8, 10)
    tir = torch.rand(2, 3, 8, 10)
    styled = apply_target_statistics(hazy, clear, tir, hazy.flip(-1), tir.flip(-2), beta=0.0)
    assert torch.equal(styled.hazy, hazy)
    assert torch.equal(styled.clear, clear)
    assert torch.equal(styled.tir, tir)


def test_rgb_mapping_is_shared_by_hazy_and_clear_and_tir_mapping_is_monotonic():
    hazy = torch.linspace(0.1, 0.4, 48).reshape(1, 3, 4, 4)
    clear = torch.linspace(0.3, 0.8, 48).reshape(1, 3, 4, 4)
    tir = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    target_rgb = torch.full((1, 3, 6, 7), 0.75)
    target_tir = torch.linspace(0.2, 0.8, 42).reshape(1, 1, 6, 7).repeat(1, 3, 1, 1)
    styled = apply_target_statistics(hazy, clear, tir, target_rgb, target_tir, beta=1.0)
    assert torch.allclose(styled.hazy, (styled.rgb_gain * hazy + styled.rgb_bias).clamp(0, 1))
    assert torch.allclose(styled.clear, (styled.rgb_gain * clear + styled.rgb_bias).clamp(0, 1))
    assert torch.all(styled.tir[..., 1:] >= styled.tir[..., :-1])
    assert styled.hazy.min() >= 0 and styled.hazy.max() <= 1
```

- [ ] **Step 2: Run the test and confirm import failure.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_target_style.py -q`  
Expected: FAIL because `training.target_style` does not exist.

- [ ] **Step 3: Implement a transparent, bounded transform API.**

```python
@dataclass(frozen=True)
class StyledSourceBatch:
    hazy: torch.Tensor
    clear: torch.Tensor
    tir: torch.Tensor
    rgb_gain: torch.Tensor
    rgb_bias: torch.Tensor
    tir_gain: torch.Tensor
    tir_bias: torch.Tensor


def apply_target_statistics(hazy, clear, tir, target_hazy, target_tir, *, beta,
                            min_gain=0.75, max_gain=1.35, max_abs_bias=0.20):
    """Return only transformed modalities; caller keeps density/mask unchanged."""
```

For each batch item, compute mean/std over spatial dimensions. Derive the raw gain as `target_std / source_std.clamp_min(1e-6)`, clamp it, and blend it with one using `beta`. Derive the bias from target/source means, clamp it, and apply the same RGB gain/bias to both source `hazy` and source `clear`. Compute TIR statistics on channel zero and repeat the transformed scalar TIR signal to three channels. Raise `ValueError` for mismatched batch/channel dimensions or `beta` outside `[0, 1]`.

- [ ] **Step 4: Add bounded random-beta coverage.**

```python
assert torch.allclose(styled.hazy, (styled.rgb_gain * hazy + styled.rgb_bias).clamp(0, 1))
assert torch.allclose(styled.clear, (styled.rgb_gain * clear + styled.rgb_bias).clamp(0, 1))
assert (styled.rgb_gain >= 0.75).all() and (styled.rgb_gain <= 1.35).all()
```

- [ ] **Step 5: Run the unit test.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_target_style.py -q`  
Expected: PASS.

- [ ] **Step 6: Commit the isolated transform.**

```bash
git add training/target_style.py tests/test_target_style.py
git commit -m "feat: add bounded target statistics transform"
```

### Task 2: Extract the route-ramped real adaptation loss without changing EMA behavior

**Files:**
- Create: `training/real_adaptation.py`
- Modify: `EMA.py:99-173`
- Test: `tests/test_real_adaptation.py`, `tests/test_ema_core.py`

- [ ] **Step 1: Write failing loss tests.**

```python
def test_route_multiplier_zero_excludes_only_route_consistency():
    losses = real_adaptation_loss(student, teacher, hazy, tir, generator, args, route_multiplier=0.0)
    assert losses["L_R"].item() == 0.0
    assert losses["L_J"].item() >= 0.0
    assert losses["L_M"].item() >= 0.0


def test_route_multiplier_scales_route_term_not_image_or_density_terms():
    low = real_adaptation_loss(student, teacher, hazy, tir, generator, args, route_multiplier=0.25)
    high = real_adaptation_loss(student, teacher, hazy, tir, generator, args, route_multiplier=1.0)
    assert torch.allclose(high["L_J"], low["L_J"])
    assert torch.allclose(high["L_M"], low["L_M"])
    assert high["L_R"] >= low["L_R"]
```

Use deterministic mocked teacher/student outputs and fixed geometries so `low` and `high` use exactly the same base predictions.

- [ ] **Step 2: Run the new tests and confirm failure.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_real_adaptation.py -q`  
Expected: FAIL because the module/function does not exist.

- [ ] **Step 3: Move `_Geometry`, `_sample_geometry`, and `_real_loss` from `EMA.py` into `training/real_adaptation.py`.**

Expose:

```python
def real_adaptation_loss(teacher, student, hazy, tir, generator, args, *,
                         route_multiplier=1.0, clip_criterion=None, text_features=None):
    if not 0.0 <= route_multiplier <= 1.0:
        raise ValueError("route_multiplier must be in [0, 1]")
    # Preserve current teacher/student geometry and stability-weight computation.
    # Call real_consistency_loss(..., lambda_r=args.lambda_ema_r * route_multiplier).
```

Keep `L_J`, `L_M`, `L_clip`, transform inversion, and `stability_weights` byte-for-byte equivalent in behavior. In `EMA.py`, import the helper and call it with `route_multiplier=1.0`; remove the duplicate private implementation.

- [ ] **Step 4: Run regression and new tests.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_real_adaptation.py tests/test_ema_core.py tests/test_ema_end_to_end.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit the refactor.**

```bash
git add training/real_adaptation.py EMA.py tests/test_real_adaptation.py tests/test_ema_core.py
git commit -m "refactor: share route-ramped real adaptation loss"
```

### Task 3: Define UDA configuration and checkpoint semantics

**Files:**
- Create: `option/UDA.py`
- Test: `tests/test_uda_options.py`

- [ ] **Step 1: Write parser/validation tests.**

```python
def test_uda_requires_a_v2_source_checkpoint_and_target_dirs(tmp_path):
    args = build_parser().parse_args([
        "--source_checkpoint", str(tmp_path / "source_best.pt"),
        "--real_data_dir", str(tmp_path / "m3fd"),
        "--real_tir_dir", str(tmp_path / "m3fd" / "ir"),
    ])
    assert args.style_probability == 0.5
    assert args.route_consistency_warmup_steps > 0


def test_uda_rejects_invalid_style_and_route_ramp_values():
    args = build_parser().parse_args(["--style_probability", "1.1"])
    with pytest.raises(ValueError, match="style_probability"):
        validate_config(args)
```

- [ ] **Step 2: Run and confirm failure.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_uda_options.py -q`  
Expected: FAIL because `option.UDA` does not exist.

- [ ] **Step 3: Implement `option/UDA.py`.**

Use `option/EMA.py` as the configuration authority for inherited V2 keys and checkpoint preflight. Add runtime-only arguments:

```text
--stage {source_style,ema}
--source_checkpoint
--resume_checkpoint
--real_data_dir --real_tir_dir
--source_anchor_data_dir --validation_data_dir
--style_probability (default 0.5)
--style_beta_min (default 0.0) --style_beta_max (default 0.6)
--style_min_gain (default 0.75) --style_max_gain (default 1.35)
--style_max_abs_bias (default 0.20)
--route_consistency_warmup_steps (default 1000)
--route_consistency_ramp_steps (default 1000)
--probe_hazy --probe_tir --probe_output_dir
```

Validate ranges, require exactly one initial/resume checkpoint according to stage, and persist all UDA semantic arguments in the UDA checkpoint config. Require V2 checkpoint preflight before any model allocation.

- [ ] **Step 4: Run option tests.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_uda_options.py tests/test_formal_options.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit configuration work.**

```bash
git add option/UDA.py tests/test_uda_options.py
git commit -m "feat: add two-stage UDA configuration"
```

### Task 4: Implement Stage A source-style fine-tuning

**Files:**
- Create: `UDA.py`
- Test: `tests/test_uda_smoke.py`

- [ ] **Step 1: Write a mocked Stage A failing test.**

```python
def test_source_style_stage_keeps_density_and_mask_supervision(monkeypatch, tmp_path):
    result = run_source_style_stage(fake_args, fake_source_loader, fake_target_loader, fake_model)
    assert result["checkpoint"]["training_stage"] == "source"
    assert fake_loss_call.density_gt.equal(original_density_gt)
    assert fake_loss_call.completion_mask_gt.equal(original_completion_mask_gt)
    assert result["styled_batch_seen"] is True
```

Mock VGG/CLIP constructors and use one 8×8 batch so the test checks routing/loss wiring rather than external resource downloads.

- [ ] **Step 2: Run and confirm failure.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_uda_smoke.py::test_source_style_stage_keeps_density_and_mask_supervision -q`  
Expected: FAIL because `UDA.py` has no stage runner.

- [ ] **Step 3: Implement `run_source_style_stage()` in `UDA.py`.**

The stage must:

1. Preflight/load a Source V2 checkpoint with `preflight_source_initialization_checkpoint` and `load_strict_v2_state_dict`.
2. Create `SynthMultiModalDataset` from FLIR and `RealMultiModalDataset` from M3FD.
3. On each Source batch, draw a target batch of the same batch size; sample `beta` per item and apply `apply_target_statistics` only where a Bernoulli `style_probability` mask is true.
4. Pass `(styled_hazy, styled_clear, styled_tir, original_density_gt, original_completion_mask_gt)` to existing `compute_physical_mask_batch_losses`.
5. Reuse Source learning-rate scheduling, validation, diagnostics, and `build_source_checkpoint`; save `source_style_last.pt` and PSNR-best `source_style_best.pt` with `training_stage="source"`.

The unstyled branch must call the same loss path. Do not mix real M3FD images into a supervised Source batch.

- [ ] **Step 4: Add a probe writer.**

```python
@torch.inference_mode()
def save_target_probe(model, hazy, tir, output_dir, *, route_temperature):
    output = model(hazy, tir, route_temperature=route_temperature, route_mode="hard")
    # save prediction, density_map, route_soft, route_hard, boundary_map
    return {"density_mean": output["density_map"].mean(),
            "route_hard_fraction": output["route_hard"].mean()}
```

Save probes before Stage A and after every validation epoch, and log the two scalar indicators. The writer must never require target clear/mask files.

- [ ] **Step 5: Run the Stage A smoke test.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_uda_smoke.py::test_source_style_stage_keeps_density_and_mask_supervision -q`  
Expected: PASS.

- [ ] **Step 6: Commit Stage A.**

```bash
git add UDA.py tests/test_uda_smoke.py
git commit -m "feat: add target-statistics source fine-tuning"
```

### Task 5: Implement Stage B EMA with delayed route consistency

**Files:**
- Modify: `UDA.py`
- Modify: `tests/test_uda_smoke.py`

- [ ] **Step 1: Write failing schedule and wiring tests.**

```python
def test_route_consistency_multiplier_has_warmup_then_linear_ramp():
    assert route_consistency_multiplier(0, warmup_steps=10, ramp_steps=20) == 0.0
    assert route_consistency_multiplier(10, warmup_steps=10, ramp_steps=20) == 0.0
    assert route_consistency_multiplier(20, warmup_steps=10, ramp_steps=20) == 0.5
    assert route_consistency_multiplier(30, warmup_steps=10, ramp_steps=20) == 1.0


def test_ema_stage_uses_source_anchor_while_route_loss_is_warmed_up(monkeypatch):
    result = run_ema_stage(fake_args, fake_source_loader, fake_real_loader, fake_model)
    assert result["first_real_route_multiplier"] == 0.0
    assert result["source_anchor_calls"] > 0
```

- [ ] **Step 2: Run and confirm failure.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_uda_smoke.py -q`  
Expected: FAIL because the schedule/stage does not exist.

- [ ] **Step 3: Add Stage B to `UDA.py`.**

Implement:

```python
def route_consistency_multiplier(step, *, warmup_steps, ramp_steps):
    if step <= warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, (step - warmup_steps) / float(ramp_steps))
```

Load `source_style_best.pt` or a compatible Stage A Source checkpoint; initialize EMA teacher/student exactly as current `EMA.py` does. Each step computes:

```python
real = real_adaptation_loss(..., route_multiplier=route_consistency_multiplier(...))
source = compute_physical_mask_batch_losses(student, source_batch, args, source_global_step, ...)
loss = real["L_real"] + args.w_loss_Clip * real["L_clip"] + args.lambda_anchor * source["losses"]["total"]
```

Use `build_ema_checkpoint` for `ema_last.pt`/`ema_best.pt`; record current route multiplier and target probe scalars in logs. Keep current Source validation and EMA teacher update semantics.

- [ ] **Step 4: Run Stage B and existing EMA regressions.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_uda_smoke.py tests/test_real_adaptation.py tests/test_ema_end_to_end.py -q`  
Expected: PASS.

- [ ] **Step 5: Commit Stage B.**

```bash
git add UDA.py tests/test_uda_smoke.py
git commit -m "feat: add delayed-route EMA adaptation stage"
```

### Task 6: Documentation, end-to-end verification, and launch commands

**Files:**
- Modify: `Codex/2026-07-30-m3fd-unsupervised-domain-adaptation-design.md`
- Modify: `README.md` if it contains training-entry instructions
- Test: full suite

- [ ] **Step 1: Document exact stage commands using explicit paths.**

```powershell
D:\anaconda\envs\CoA\python.exe UDA.py --stage source_style `
  --source_checkpoint F:\1_paper_pan\1_Dehaze_Paper\1_pth_model_duibi_experiments\1_Ours\pth\v2\source_best.pt `
  --source_anchor_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\train `
  --validation_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\test `
  --real_data_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD `
  --real_tir_dir F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\ir
```

Document that Stage B consumes the Stage A best Source checkpoint and that results must be labelled M3FD UDA.

- [ ] **Step 2: Run focused test groups.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_target_style.py tests/test_real_adaptation.py tests/test_uda_options.py tests/test_uda_smoke.py -q`  
Expected: PASS.

- [ ] **Step 3: Run the complete suite.**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests -q`  
Expected: PASS with only pre-existing dependency deprecation warnings.

- [ ] **Step 4: Run a one-step Stage A and one-step Stage B smoke run on M3FD/FLIR.**

Use `epochs=1`, `iters_per_epoch=1`, `batch_size=1`, a disposable output directory, and the fixed M3FD `00343` probe. Verify both stage checkpoints load through strict V2 preflight and all five probe images are emitted.

- [ ] **Step 5: Commit documentation and final verification updates.**

```bash
git add Codex/2026-07-30-m3fd-unsupervised-domain-adaptation-design.md README.md
git commit -m "docs: add M3FD UDA training workflow"
```

## Plan self-review

- Spec coverage: Tasks 1 and 4 implement bounded target-statistics Source adaptation; Tasks 2 and 5 implement delayed route consistency and EMA; Task 6 covers probes, protocol documentation, tests, and executable verification.
- Boundary coverage: the plan never creates M3FD density/mask labels, never changes the 0.5 hard-route threshold, and never adds an external model or dataset.
- Checkpoint coverage: Stage A intentionally emits a normal V2 Source checkpoint; Stage B uses existing strict Source→EMA loading.
- Test coverage: unit, option, stage smoke, EMA regression, full suite, and real one-step execution are all explicit.
