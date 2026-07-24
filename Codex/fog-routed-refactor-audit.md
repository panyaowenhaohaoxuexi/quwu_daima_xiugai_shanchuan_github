# Fog-routed RGB--TIR refactor audit

## 2026-07-24 — phase 1: formal data, router, HDE, model shell

- Replaced the synthetic five-tuple path with the formal four-tuple
  `(hazy_rgb, clear_rgb, tir, density_gt)`; `IR_Completion_Mask_GT` is no
  longer indexed by `SynthMultiModalDataset`.
- Added lossless scalar/TIR loaders. 16-bit PNG values are recognized even
  when Pillow exposes them as `I/int32`, but unknown `I` ranges require an
  explicit fixed or calibrated range.
- Density semantics are centralized in `convert_density_semantics`; default
  is transmission-to-density inversion. Inspection and training share this
  conversion.
- Formal HDE output now contains `density_map` and a single-source
  `h2/h4/h8/h16` TIR structure pyramid.
- Added a strictly pointwise monotonic router. Its hard route follows the
  required straight-through form and has no neighbor input.
- Added `FogRoutedRGBTIRDehazer` with one shared encoding call and route-only
  decoding. Tiny CPU tests exercise a 31x47 input, soft/hard decode, padding
  crop, sigmoid RGB output, and backward.
- Appearance memory reports h2 per-sample mass/ratio. Reliability changes
  attention scores but not RGB values; h2 confidence is bilinearly restored
  while fallback masks use nearest-neighbor restoration.
- Counterfactual context reuse, complete Omega/q training, full EMA training,
  strict checkpoint schemas, and production training loops remain pending.

## 2026-07-24 — phase 2: deterministic sampling primitives

- Added `StatefulRandomSampler` for single-process recovery. It persists the
  epoch permutation and a committed cursor; the cursor advances only after a
  successful optimizer step.
- Added a detached `OmegaSampler` with 4--8 requested local regions, density
  quantile rotation, strong-edge blocking, non-overlap, soft affinity weights,
  per-sample ownership and explicit fallback status. It is ready for the
  counterfactual training loop but is not yet connected to it.

## 2026-07-24 — phase 3: counterfactual and EMA math

- Added a shared-context counterfactual helper that performs the two route
  decodes under `torch.no_grad()` and produces detached q with the required
  direction (one means completion has lower error).
- Added EMA stability weights and weighted J/M/R consistency primitives. J/M
  use weighted L1; R uses weighted BCE; weights and targets are detached.
- Added batch-level paired geometry descriptions for EMA. A/B/S each use one
  independently sampled identity/hflip/rot90-combination for the whole batch;
  RGB/TIR are paired and J/M/R are inverse-transformed before comparison.
- Added strict format/stage/model/density checkpoint metadata validation and
  capture/restore helpers for Python, NumPy, Torch CPU/CUDA, Omega, geometry,
  and named DataLoader generators.
- Added source and EMA schema builders. Source checkpoints contain only
  `model`; EMA checkpoints contain only `student`/`teacher` plus separate
  source and EMA step counters.
- Replaced HDE's direct torchvision deformable convolution dependency with a
  state-dict-compatible pure-PyTorch modulated sampler. The CPU fallback now
  performs differentiable `grid_sample` for every 3x3 kernel point and honors
  modulation masks.

## Tests observed

- Initial baseline: `67 passed` in CoA.
- After phase 1: `71 passed` in CoA.
- Primary command: `D:\anaconda\envs\CoA\python.exe -m pytest -q`.

## Call and memory accounting

- Current formal `forward`: one shared HDE/RGB/TIR encoding call and one
  route decode call.
- Counterfactual batching is not yet wired into the training entry point; no
  claim is made that the current entry point performs source q supervision.
- No construct-time disk/network/directory or backend probing was added.

## Risks and next phase

- HDE currently retains torchvision DeformConv for normal execution; the
  pure-PyTorch HDE fallback and cross-backend parity test are pending.
- The current main model implements the formal output/memory boundaries but
  still needs full multi-scale training-loss, Omega and EMA integration.
# 2026-07-24 — source-step scheduling and successful-step commit

- Added `training/schedules.py`: source route temperature, hard-route mode,
  counterfactual gate, route-loss warm-up and binary-loss warm-up are computed
  in one pure function.  A non-zero route loss cannot occur before the
  counterfactual gate.
- Added `training/step_control.py`: non-finite gradients skip the optimizer
  update.  Callers can now make all state commits conditional on the returned
  success flag; AMP support follows the same contract.
- `Teacher.py` now executes epoch/batch training rather than stopping after a
  temporary first batch.  It encodes once, gathers detached encoded context by
  Omega owner index for the two counterfactual decode groups, aggregates q back
  to source samples, and advances the source step/empty-Omega streak only after
  a successful optimizer update.
- Shared-context counterfactual gather was tested with mixed owner indices;
  it selects only the current candidate chunk and does not rerun HDE/RGB/TIR
  encoders.  `counterfactual_chunk_size` now bounds actual routed-decoder
  batches; chunked and single gathered decoding agree within test tolerance.
- Tests added/updated: `test_successful_step.py`, `test_route_schedule.py`,
  `test_source_counterfactual.py`, `test_source_objective.py`.

# 2026-07-24 — support-aware reconstruction losses

- Added support-aware local SSIM to `loss/fog_routed_source_loss.py`.  Its
  moments are normalized by only the active local support and invalid centers
  are excluded; it does not compute full-image SSIM then mask center pixels.
- `training/source_objective.py` now separates ordinary reconstruction
  (L1/gradient/SSIM) from the boundary objective (L1/gradient only), while all
  masks used for regional scoring remain detached.  The outer router weight
  now applies only to density/route/binary terms as specified.
- Configuration now exposes reconstruction and boundary component weights plus
  the support-aware SSIM window/support threshold.

# 2026-07-24 — EMA buffer classification

- Added explicit EMA buffer helpers.  Parameter tensors use EMA; all ordinary
  buffers are copied exactly after a successful student update; only an
  explicitly named floating `ema_state` buffer may use EMA.  This prevents
  HDE's fixed differential kernels and frozen BatchNorm statistics from being
  smoothed merely because their dtype is floating point.

## 2026-07-24 — checkpoint-driven evaluation construction

- Eval/Eval_EMA now construct `FogRoutedRGBTIRDehazer` from all persisted
  architecture semantics rather than only `base_channels`. The shared TIR
  normalization/tolerance mapping is reconstructed from checkpoint metadata.
  Missing semantic keys are a hard error before `strict=True` state loading.

## 2026-07-24 — formal real-domain data path

- Added `RealMultiModalDataset`, returning only `(hazy_rgb, tir,
  sample_metadata)`. It uses the formal TIR loader/alignment policy and never
  carries real clear targets, density GT, completion masks, sky masks or CLIP
  features. EMA now starts student/teacher from a strict source checkpoint and
  constructs independent real/source-anchor loaders. Per-step adaptation is
  the remaining EMA integration item.

## 2026-07-24 — EMA successful-step transaction

- Added a testable EMA step transaction: it evaluates real consistency and one
  source-anchor loss exactly once, backpropagates their weighted sum, and only
  after a successful optimizer step advances scheduler/teacher EMA/buffer
  synchronization. Non-finite updates leave teacher state untouched.

## 2026-07-24 — continued integration and safeguards

- Source training and EMA anchors now call the same full shared-context source
  step: one HDE/RGB/TIR encode, one gradient-bearing routed decode, and two
  no-grad owner-index-chunked routed counterfactual decode groups. Counterfactual
  candidates reuse detached context and do not re-encode either modality.
- Omega sampling now uses connected local growth under density similarity and
  TIR-edge barriers. It records requested/actual counts and fallback causes;
  a feasible 6-region sample covers low/middle/high density bands without
  overlap. The implementation returns no fabricated route target when no valid
  region can be formed.
- Completion counterfactual memory exclusion is conservatively dilated at
  every memory scale. The dilation radius includes deformable offset, encoder/
  fusion receptive-field allowance and `memory_exclusion_extra_margin`; an
  exhausted memory correctly falls back to the TIR-conditioned prior.
- Formal model construction in source training now uses every persisted model
  semantic key. Checkpoint restore rejects changed source/real manifests before
  a new iterator is made.
- Empty-Omega streak state is advanced only after a committed update with an
  effective route-loss weight and no valid Omega. Source and EMA-anchor state
  remain separate. A skipped/non-finite update stops the current iterator so
  its uncommitted batch is retried rather than advancing a persisted cursor.
- Source and EMA data loaders now receive the same persisted TIR normalization,
  percentile scope and channel-tolerance configuration. Source augmentation
  descriptions are synchronized with the stateful sampler epoch.
- The pure-PyTorch HDE deformable sampler is now numerically aligned with the
  torchvision backend, including zero padding and `(dy, dx)` offset order; CPU
  fallback remains the default formal HDE path.
- Evaluation validates the complete formal checkpoint schema and applies the
  checkpoint alignment policy before model invocation. EMA resume locks model
  and preprocessing semantics, restoring adaptation configuration unless an
  explicit override is requested and recorded.
- Source/EMA startup now logs the pre-worker density inspection and the first
  batch's converted density statistics separately, preserving the four-tensor
  supervised batch interface.
- Eval/Eval_EMA now use the declared output-format argument through their shared
  inference helper; strict alignment and checkpoint validation remain active.
- The unique decoder now consumes all routed h16/h8/h4/h2 structure scales via
  local cross attention. Its query is structure-only and Key/Value are solely
  the assembled RGB appearance tokens; the implementation never materializes a
  global spatial attention matrix.
- Full-completion counterfactuals now have an automated RGB-feature gradient
  isolation check; the public model signature is also tested to accept only
  hazy RGB and TIR, never clear GT or density GT.
- Per-scale memory diagnostics now expose actual confidence/fallback/mass/ratio
  only under `output["debug"]`; formal h2 reporting fields remain unchanged.
- Counterfactual decode has an explicit all-buffer before/after regression test;
  it preserves HDE BatchNorm buffers while reusing detached shared context.
- Default formal model parameter baseline: 1,563,517 total parameters;
  695,178 in shared HDE/router/RGB/TIR encoding and 868,339 in route-dependent
  three-stream, memory and decoder components. A normal source step therefore
  executes one shared encode plus one gradient decode and two no-grad routed
  decode groups; it does not repeat the shared encoder per Omega.
- The pure-PyTorch/torchvision deform sampler parity test now covers forward
  values and gradients of input, convolution weight, offsets and modulation
  masks under the documented numeric tolerance.
- Added a true source-entrypoint smoke test: a 32×32 four-tensor batch performs
  an optimizer update, writes a strict checkpoint, restores at an epoch
  boundary and performs the next deterministic update. Source `--train_size`
  is now explicit and shared with the EMA anchor dataset.
- Added an EMA-entrypoint smoke test: strict source checkpoint initialization,
  real A/B/S consistency, complete source anchor/q route objective, successful
  EMA update and strict EMA checkpoint schema are all exercised on CPU.
- `train_size` is treated as persisted preprocessing semantics during source
  and EMA restoration, preventing an implicit return to the parser default.
- EMA resume now keeps invocation-scoped data roots and the requested epoch
  stop boundary while still locking model/preprocessing semantics and restoring
  adaptation settings.  If an EMA checkpoint was written with either the real
  or source sampler exhausted, each independent sampler advances exactly once
  to its next saved-seed permutation before the resumed iterator is created.
  The strict CPU EMA entrypoint test exercises that epoch-boundary resume path.
- Exact sampler-resume support is now explicitly single-process: both formal
  source and EMA entrypoints reject `WORLD_SIZE != 1` before any directory,
  dataset or model side effect.  This prevents accidentally claiming DDP
  mid-epoch determinism that the first implementation does not provide.
- Source and EMA checkpoint `rng_state` now carries the live Omega generator;
  EMA also persists the live paired-geometry generator.  They are restored
  after model/optimizer/sampler state and before the next iterator or Omega/
  geometry sample is made.  End-to-end source and EMA checkpoint tests assert
  these generator states are present.
- Source resume now validates the full persisted semantic set, including TIR
  normalization, density conversion/normalization, alignment policy and
  training spatial size; a same-shaped checkpoint cannot silently resume with
  changed preprocessing meaning.

## 2026-07-24 — real-data runnable smoke validation

- Formal source loading was run against
  `F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\train`:
  8,445 strict RGB/clear/TIR/transmission pairs were discovered and a
  converted four-tensor batch passed the loader check.
- A copied, unmodified FLIR pair completed one CPU source optimization step
  with hard-route main decoding plus both counterfactual decode groups; the
  strict source checkpoint recorded `global_step=1` and the live Omega RNG.
- One original-resolution M3FD pair (`1024x768`) completed a CPU EMA step
  using the FLIR checkpoint as source initialization.  Formal teacher Eval_EMA
  then wrote prediction, density, soft/hard-route and boundary outputs, all at
  exactly `1024x768`.  These smoke runs did not perform any full training.
- Latest full CoA regression: `133 passed` (`D:\anaconda\envs\CoA\python.exe -m pytest -q`).

## 2026-07-24 -- completion fixes: bounded memory, valid q, transactional epochs and geometry

- Baseline HEAD was `363a74d27114f56062b1816098c4eafa203c2b93`; the initial full CoA
  regression was `133 passed, 5 warnings`. Existing user untracked resources
  were retained throughout.
- `memory_query_chunk_size` is now a strict model construction/checkpoint
  semantic. Every per-sample score matrix is constructed with at most that
  many query rows; instrumentation regression exercised seven-row chunks and
  an unchunked reference with identical retrieved output, confidence,
  fallback, reliable mass and ratio.
- TIR percentile loading now validates channels in raw code-value space, then
  averages to one channel before normalizing. Per-image percentiles return a
  finite zero tensor for a degenerate range; dataset scope uses the persisted
  calibrated low/high values and rejects missing or inverted values.
- `compute_q` returns detached `(q, q_valid_mask)`. L1, gradient and local
  SSIM are normalized over the components actually valid for each candidate;
  route BCE receives only pixels with a valid q comparison. Training prints
  the q/reconstruction/boundary component weights and explicitly warns about
  L1-only defaults.
- Failed source and EMA optimizer steps now rebuild the iterator without
  committing sampler position, global/EMA step, teacher, scheduler, streak or
  epoch. Epoch checkpoints are written only after the sampler has truly
  completed the epoch.
- Source training uses a deterministic geometry description containing resize,
  crop, flip and rot90 settings. It scales only the short edge to `train_size`,
  crops all four float tensors together, clamps resized density, and leaves
  evaluation at original resolution.
- Memory exclusion radii are derived from the stem/downsample/residual
  recurrence plus local-offset, renderer, fusion-residual and configured
  margin. With `deform_max_offset=2` and margin `1`, h2/h4/h8/h16 token-grid
  radii are `9/10/11/11`.
- Final verification: `D:\anaconda\envs\CoA\python.exe -m pytest -q` ->
  `141 passed, 5 warnings in 17.75s`. Warnings are four upstream torchvision
  Pillow deprecations and the existing `TestDataset` collection warning.
