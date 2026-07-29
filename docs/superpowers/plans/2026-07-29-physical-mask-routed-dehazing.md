# Physical-Mask-Routed Dehazing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the old density-only, counterfactual route-label pipeline with a HDE-feature router explicitly supervised by the physically generated completion mask.

**Architecture:** `SynthMultiModalDataset` returns `(hazy, clear, tir, density_gt, completion_mask_gt)`. HDE exposes its already-computed full-resolution visible, infrared, and difference features as a formal routing interface. A lightweight feature-guided CNN consumes those features plus predicted density and emits route logits. Source and EMA anchor training use soft teacher-gated decoding, `SmoothL1` density supervision, direct weighted BCE route supervision, and reconstruction loss; inference alone thresholds the predicted route.

**Tech Stack:** Python, PyTorch, pytest, Pillow.

---

### Task 1: Make the synthetic dataset require and return physical completion masks

**Files:**
- Modify: `data/data_loader.py:16-308`
- Modify: `tools/check_synth_loader.py:27-45`
- Modify: `tests/test_fog_routed_data.py:64-280`

- [ ] **Step 1: Write failing dataset tests**

Add fixtures containing `hazy/1_mist`, `Transmission_Map_GT/1_mist`, and `mask_GT/1_mist`. Assert that the dataset returns five tensors, `mask` has shape `[1,H,W]`, only contains `0/1`, is spatially aligned with RGB/density after crop/flip/rotation, and a missing mask is reported as missing rather than silently used as a four-item sample.

```python
hazy, clear, tir, density, completion = dataset[0]
assert completion.shape == (1, 7, 9)
assert set(completion.unique().tolist()) <= {0.0, 1.0}
```

- [ ] **Step 2: Run the focused test file and verify RED**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_fog_routed_data.py -q`

Expected: failure because `SynthMultiModalDataset.__getitem__` returns four tensors and does not index `mask_GT`.

- [ ] **Step 3: Implement paired mask loading**

Set the default synthetic severity tuple to `("1_mist", "2_middle", "3_dense", "4_local_extreme")`. Index `root/mask_GT/<level>` by stem alongside density; add `mask_path` to every sample; validate mask dimensions with the other paired inputs; load it with `load_scalar_map_as_float_tensor`; binarize as `(mask > 0.5).float()`; and apply the same geometry with nearest-neighbour interpolation for the mask. Return `(hazy, clear, tir, density, mask)` and make the empty collate value a five-tuple.

- [ ] **Step 4: Update the loader diagnostic utility**

Unpack `completion_mask` in `tools/check_synth_loader.py` and print its shape and positive-pixel count so a command-line dataset check detects a misnamed or all-zero mask folder.

- [ ] **Step 5: Run dataset tests and the loader check**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_fog_routed_data.py -q`

Expected: PASS. Then run `D:\anaconda\envs\CoA\python.exe tools/check_synth_loader.py --help` and verify the command still imports successfully.

### Task 2: Expose existing HDE route features through its formal interface

**Files:**
- Modify: `model/hde.py:122-219`
- Modify: `tests/test_hde_formal_interface.py:1-28`
- Modify: `tests/test_hde_dual_stream.py:42-96`

- [ ] **Step 1: Write failing formal-interface tests**

Assert that standard `HDE(rgb, tir)` returns a `routing_features` mapping, without `return_debug=True`, with exact keys `fm_vis`, `fm_ir`, `struct_diff_gap`, and `struct_diff_gmp`; all maps must be finite and share the density map's `[B,*,H,W]` spatial size.

```python
output = HDE()(rgb, tir)
features = output["routing_features"]
assert set(features) == {"fm_vis", "fm_ir", "struct_diff_gap", "struct_diff_gmp"}
assert features["fm_vis"].shape == (1, 96, 32, 32)
```

- [ ] **Step 2: Run HDE tests and verify RED**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_hde_formal_interface.py tests/test_hde_dual_stream.py -q`

Expected: `KeyError: routing_features`.

- [ ] **Step 3: Add the formal routing feature mapping**

Build `routing_features` from HDE's already-computed `fm_vis`, `fm_ir`, `struct_diff_gap`, and `struct_diff_gmp`; include it in every dictionary-mode return path. Keep `return_feat=True` as the legacy inspection API, but do not require debug mode for routing features.

- [ ] **Step 4: Run HDE tests and verify GREEN**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_hde_formal_interface.py tests/test_hde_dual_stream.py -q`

Expected: PASS, with gradients from a routing feature reaching both RGB and TIR inputs.

### Task 3: Replace the density-only monotonic router with a feature-guided CNN router

**Files:**
- Create: `model/feature_guided_router.py`
- Modify: `model/Teacher.py:1-236`
- Modify: `model/__init__.py`
- Delete: `model/monotonic_router.py`
- Delete: `tests/test_monotonic_router.py`
- Create: `tests/test_feature_guided_router.py`
- Modify: `tests/test_fog_routed_dehazer.py`

- [ ] **Step 1: Write failing router tests**

Define the desired public API `FeatureGuidedRouter(hidden_channels)(density_map, routing_features, temperature)`. Verify its logits/soft/hard shapes, temperature validation, finite gradients, and local-context behavior: changing a neighbour in `fm_vis` changes the centre route because the router contains a `3x3` convolution. Also verify that changing `fm_vis` changes the route while density remains fixed, proving the router is no longer density-only.

- [ ] **Step 2: Run router tests and verify RED**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_feature_guided_router.py -q`

Expected: import failure because `FeatureGuidedRouter` does not exist.

- [ ] **Step 3: Implement the lightweight router**

Implement `FeatureGuidedRouter` with input channels `1 + 96 + 96 + 1 + 1`, a `1x1 Conv -> SiLU -> 3x3 Conv -> SiLU -> 1x1 Conv` head, and output keys `route_logits`, `route_soft`, `route_hard`. Compute `route_soft = sigmoid(route_logits / temperature)` and define `route_hard` only as a non-differentiable inference threshold. Do not enforce density monotonicity and do not use a straight-through hard route.

- [ ] **Step 4: Wire it into the model context**

Replace `MonotonicFogRouter` in `FogRoutedRGBTIRDehazer` with `FeatureGuidedRouter`; pass `hde_output["routing_features"]` to it; retain route logits and both route maps in `encode_context`. Update formal model tests to assert that the active model imports the new router and consumes HDE routing features.

- [ ] **Step 5: Run focused model/router tests**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_feature_guided_router.py tests/test_hde_formal_interface.py tests/test_fog_routed_dehazer.py -q`

Expected: PASS.

### Task 4: Use soft teacher-gated routing during training and hard routing only at inference

**Files:**
- Modify: `model/Teacher.py:275-430`
- Modify: `training/source.py:145-290`
- Modify: `option/Teacher.py:71-210`
- Modify: `option/EMA.py`
- Create: `tests/test_route_schedule.py` replacement cases
- Modify: `tests/test_source_step.py`

- [ ] **Step 1: Write failing schedule and model-gating tests**

Specify a schedule returning `route_temperature` and `teacher_gate_alpha`. At step zero, alpha must be `1.0`; after `route_teacher_anneal_steps`, it must be `0.0`. Specify a full-resolution training gate:

```python
gate = teacher_gate_alpha * completion_mask_gt + (1.0 - teacher_gate_alpha) * route_soft
```

Assert that decoder training uses this continuous gate and that the inference `forward(..., route_mode="hard")` path still uses only `route_hard`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_route_schedule.py tests/test_source_step.py -q`

Expected: failure because the current schedule contains counterfactual/hard-start fields and the decoder cannot accept a full training gate.

- [ ] **Step 3: Simplify the route schedule and decoder interface**

Remove counterfactual, Omega, hard-start, binary-loss, and route-loss-start controls. Keep `route_tau_start`, `route_tau_end`, and add `route_temperature_anneal_steps` plus `route_teacher_anneal_steps`. Make `source_route_schedule` linearly anneal temperature and alpha. Add a full-route override to `decode_with_route` for the continuous teacher gate, preserving the existing hard route only for inference. Remove the obsolete counterfactual override/memory-exclusion code paths that existed solely to manufacture `q` labels.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_route_schedule.py tests/test_source_step.py -q`

Expected: PASS.

### Task 5: Replace the old source objective with direct density and mask supervision

**Files:**
- Modify: `loss/source.py`
- Modify: `training/source.py`
- Modify: `tests/test_source_objective.py`
- Delete: `tests/test_auto_route_gt.py`
- Delete: `tools/generate_auto_route_gt.py`
- Delete: `tools/visualize_auto_route_gt.py`

- [ ] **Step 1: Write failing direct-supervision loss tests**

Test that route loss receives every mask pixel rather than an Omega support; that a wrong positive route logit raises loss; that `route_logits` receive finite gradients; and that a batch with sparse positives obtains a finite positive-class weight. Test the returned loss keys `reconstruction`, `density`, `route`, and `total`.

- [ ] **Step 2: Run the loss tests and verify RED**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_source_objective.py -q`

Expected: failure because the objective requires `q`, `omega_support`, boundary loss, and binary loss.

- [ ] **Step 3: Implement the three-term source objective**

Implement:

```python
L_rec = charbonnier(pred_clear, clear_rgb)
L_density = smooth_l1(density_map, density_gt)
L_route = binary_cross_entropy_with_logits(route_logits, completion_mask_gt, pos_weight=dynamic_pos_weight)
L_total = L_rec + lambda_density * L_density + lambda_route * L_route
```

Compute `dynamic_pos_weight` from the current batch and clamp its denominator so all-zero or all-one masks remain finite. `compute_source_batch_losses` must unpack the five-item batch, build the soft teacher gate, decode once, and use `route_logits` for BCE. Remove `OmegaSampler`, `compute_q`, counterfactual predictions, `q`, boundary, and binary values from the returned result.

- [ ] **Step 4: Run source-objective and source-step tests**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_source_objective.py tests/test_source_step.py -q`

Expected: PASS.

### Task 6: Propagate the five-item batch and new observability through Source, EMA, validation, and diagnostics

**Files:**
- Modify: `Teacher.py:79-190`
- Modify: `EMA.py:20-318`
- Modify: `training/validation.py:10-28`
- Modify: `utils/visualize_fog_routed.py`
- Modify: `option/Teacher.py`
- Modify: `option/EMA.py`
- Modify: `tests/test_teacher_end_to_end.py`
- Modify: `tests/test_ema_end_to_end.py`
- Modify: `tests/test_ema_startup_and_cuda.py`
- Modify: `tests/test_fog_routed_visualization.py`

- [ ] **Step 1: Write failing entry-point tests**

Update source and EMA smoke fixtures to create `mask_GT/<level>/sample.png`. Assert that source and EMA consume five-item batches, validation deliberately ignores the fifth item, and diagnostic panels render `mask_GT` next to predicted `route_soft`/`route_hard` rather than `q` or Omega support.

- [ ] **Step 2: Run entry-point tests and verify RED**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_teacher_end_to_end.py tests/test_ema_end_to_end.py tests/test_fog_routed_visualization.py -q`

Expected: fixture/indexing or unpacking failures because current callers expect four items and old route-supervision diagnostics.

- [ ] **Step 3: Update entry points and configuration**

Unpack and move `completion_mask` in Source; remove Omega construction, empty-Omega protection, `q` logging, and counterfactual diagnostics. Make EMA source-anchor training pass the same five-item batch to the simplified source objective. Make validation unpack and ignore mask. Replace obsolete parser/config keys with `route_temperature_anneal_steps`, `route_teacher_anneal_steps`, `lambda_density`, and `lambda_route`; update checkpoint preflight so old checkpoints fail clearly as incompatible rather than silently loading under a changed architecture. Render mask GT and predicted route maps in diagnostics.

- [ ] **Step 4: Run focused entry-point tests and verify GREEN**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests/test_teacher_end_to_end.py tests/test_ema_end_to_end.py tests/test_ema_startup_and_cuda.py tests/test_fog_routed_visualization.py -q`

Expected: PASS.

### Task 7: Run regression validation and perform a real numbered-directory smoke check

**Files:**
- Modify only if failures reveal a direct incompatibility in the tasks above.

- [ ] **Step 1: Run all unit tests**

Run: `D:\anaconda\envs\CoA\python.exe -m pytest tests -q`

Expected: PASS; update only tests whose assertions encode the removed counterfactual route-label design.

- [ ] **Step 2: Run import and parser checks**

Run: `D:\anaconda\envs\CoA\python.exe Teacher.py --help` and `D:\anaconda\envs\CoA\python.exe EMA.py --help`

Expected: both expose the new annealing controls and no counterfactual/Omega/old binary-route controls.

- [ ] **Step 3: Smoke-test the generated FLIR folder structure without training**

Run `tools/check_synth_loader.py` against the generated train directory once all four numbered severities and mask folders are available. Confirm a non-empty five-tensor batch, binary completion mask, and matching spatial shapes.

- [ ] **Step 4: Record compatibility and verification results**

Document in the final handoff that old checkpoints are intentionally incompatible because the router parameters, data-batch contract, and objective changed; report the exact test commands and results.
