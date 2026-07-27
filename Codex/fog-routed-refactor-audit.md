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

## 2026-07-24 -- failed-step rollback and valid-q route supervision

- Baseline HEAD: `9d03e61a0e99d6c8397a5dd0fe4cb933cd49b06a`; baseline regression:
  `141 passed, 5 warnings`. Existing untracked user resources were preserved.
- Added buffer-only step transactions. Failed source attempts restore every
  source-model named buffer and failed EMA attempts restore every student and
  teacher named buffer using in-place `copy_()`. Parameters and optimizer state
  are not copied or restored because a failed optimizer step commits neither.
- Transactions restore Python/NumPy/Torch CPU/CUDA, Omega, geometry and named
  DataLoader-generator states. GradScaler state is intentionally excluded: an
  AMP overflow's reduced scale survives and is used by the retry.
- Source/real/source-anchor DataLoaders use independent seeds `+301/+302/+303`.
  Iterator pre-creation generator state is retained and restored before retry,
  preventing a failed iterator rebuild from consuming an extra worker base seed.
  Checkpoint capture/restore now carries the corresponding named generators.
- Source and EMA use independent runtime-only failed-step streaks (limit 20).
  These are not persisted because checkpoints are emitted only after completed
  successful epochs.
- Route supervision now records sampled Omega regions, valid-q regions and
  valid route pixels. Source and EMA empty-Omega protection uses valid-q
  region count rather than sampled region count.
- Regression coverage includes buffer/RNG transaction rollback, all-invalid and
  mixed valid-q accounting, source failed-once retry and EMA failed-once retry.
  Both retry tests match no-failure reference Omega/DataLoader generator states.
- Final command: `D:\anaconda\envs\CoA\python.exe -m pytest -q -p no:cacheprovider`
  with `PYTHONDONTWRITEBYTECODE=1` -> `145 passed, 5 warnings in 36.85s`.

## 2026-07-24 -- strict DataLoader RNG restore and full EMA retry injection

- Baseline HEAD: `2c5f29203d081813173e06f3bedd03787552543e`. The pre-change targeted
  suite (`test_checkpointing`, `test_step_transaction`, `test_ema_step`, and
  `test_ema_end_to_end`) reported `16 passed, 4 warnings`.
- Modified files: `training/checkpointing.py`, `tests/test_checkpointing.py`,
  `tests/test_ema_end_to_end.py`, and this audit record. Checkpoint
  `format_version` and top-level schema are unchanged.
- Source and EMA formal restore paths now preflight their named DataLoader RNG
  mapping before any model, optimizer, sampler, or RNG mutation. Missing
  mappings are rejected as, for example, `source checkpoint lacks DataLoader
  generator states: required=['source']`; a missing EMA `real` or
  `source_anchor` key and non-tensor stored state are also explicitly rejected.
  Extra stored generator keys remain accepted and calls without a requested
  mapping preserve generic helper compatibility.
- Regression tests prove failed preflight leaves source model and EMA
  student/teacher parameters and buffers, optimizer state, sampler cursors,
  and each requested generator unchanged. Complete mappings restore the named
  Source `source` and EMA `real`/`source_anchor` generator states exactly.
- The EMA retry integration test no longer replaces
  `EMA.run_ema_adaptation_step`. It injects one failure in
  `training.ema_step.perform_optimizer_step` after real A/B/S forward, source
  anchor/Omega/q/route support construction, and `backward()` have executed;
  the retry calls the original optimizer step. Teacher update was observed
  exactly once, only on the successful retry.
- The two attempts have byte-identical geometry-generator before/after states,
  student/teacher buffers at real-forward entry, and anchor Omega generator
  before/after states, q, valid-q sums, route support, Omega support/weight,
  owner indices, and route-supervision statistics. The successful retry
  checkpoint exactly matches the no-failure reference for student, teacher,
  optimizer, Omega/geometry/DataLoader RNG, sampler states, empty-Omega
  streaks, and committed steps.
- Verification: targeted suite with `PYTHONDONTWRITEBYTECODE=1` and
  `-p no:cacheprovider` -> `23 passed, 4 warnings in 9.33s`; full suite with
  the same cache controls -> `152 passed, 5 warnings in 24.16s`.
- Remaining risks: strict named-generator validation intentionally rejects
  legacy or damaged formal resume checkpoints missing these states; the five
  observed warnings remain upstream torchvision Pillow deprecations plus the
  existing `TestDataset` pytest collection warning.

## 2026-07-25 -- formal composite losses and global-TIR fallback prior

- Baseline HEAD: `e1710e3a331416a07a379e35d2c0cce90be6cf47`. The initial
  focused suite (`test_fog_routed_dehazer`, `test_source_counterfactual`,
  `test_formal_imports`, `test_formal_options`, and `test_checkpointing`)
  reported `41 passed, 4 warnings`.
- Design inspection confirmed that counterfactual utility is L1 + gradient +
  local SSIM, final reconstruction is also L1 + local SSIM + gradient, and
  the memory fallback prior must combine local TIR structure with global TIR
  scene context. Existing smoke defaults were q=`1/0/0`, reconstruction=`1/0/0`,
  and boundary=`1/0` for L1/gradient/SSIM (or L1/gradient).
- The then-shared configuration module owned `--formal_training`, its preliminary
  (ablation-tunable) formal preset q=`1/.5/.5`, reconstruction=`1/.2/.2`,
  boundary=`1/.5`, and shared validation. Explicit CLI loss-weight arguments
  are tracked so an explicit zero is rejected rather than silently replaced;
  any negative one of the eight weights is rejected in all modes.
- Source persists the validated `vars(args)` configuration. EMA previously
  retained source-config loss weights in its checkpoint config; its new
  `build_ema_checkpoint_config()` overwrites `formal_training` and all eight
  loss weights from the active EMA args before checkpoint creation.
- `MemoryRetriever` still uses its existing validity-weighted
  `[B, structure_channels]` global context for key/query. Its fallback prior
  now receives local structure concatenated with that context expanded over
  spatial dimensions. Default first prior convolution inputs changed h2
  `16->32`, h4 `32->64`, h8 `48->96`, h16 `64->128`; the retrieval return
  interface remains six values.
- New regression coverage checks both entry parsers, smoke defaults, formal
  values, explicit zero/negative validation, source/EMA checkpoint weights,
  and a deterministic empty-memory fallback where values cannot affect the
  appearance but remote structure changes its global-context-conditioned
  target pixel. Focused verification after the changes: `63 passed, 4 warnings`.
- Complete verification with `PYTHONDONTWRITEBYTECODE=1` and
  `-p no:cacheprovider`: `152 passed, 5 warnings in 12.97s`. Warnings remain
  the four upstream torchvision Pillow deprecations plus pytest's existing
  non-collectable `TestDataset` warning.
- Compatibility break: old source and EMA checkpoints contain same-named
  fallback-prior parameters with the old input-channel shape. They cannot be
  loaded into this revision, including through `strict=False`; no migration is
  supplied in this change.

## 2026-07-25 -- source resume objective locking and persistence hygiene

- Baseline inspection on branch `v4`, HEAD `d3a638e1acb5bb31d390d4dce54a0697b901130f`,
  found no tracked worktree changes. `_SEMANTIC_KEYS` contains only
  architecture/preprocessing values; neither `formal_training` nor the eight
  loss weights were in source resume recovery or mismatch validation.
- Before this change `Teacher.py` parsed and fully validated raw CLI args,
  logged loss components, saved config, then loaded the resume checkpoint.
  Consequently an omitted `--formal_training` could commit the L1-only parser
  defaults before the source model/optimizer state was restored. The shared
  validation also turns an unmarked formal run's unset weights into the formal
  preset and rejects negative weights, so it could not safely run before the
  checkpoint objective was resolved.
- Added the independent nine-field training-objective set and strict
  `apply_source_resume_config(...)`. Source resume now rejects a checkpoint
  missing any objective key with `source checkpoint training objective
  configuration is incomplete; missing: ...`; it never falls back to parser
  L1-only defaults. By default all nine values come from the checkpoint.
  `--allow_source_training_override` applies only fields explicitly supplied
  on this invocation and prints only actual value differences. It cannot
  disable formal training because no negative formal flag exists.
- Parser actions now place explicit `--formal_training` and loss-weight flags
  in one private `_explicit_training_objective_keys` collection. The resume
  merge retains restored weight markings until the single final validation, so
  checkpoint custom formal weights are not replaced by the formal preset.
- Teacher now resolves checkpoint objectives before its only `validate_config`
  call, then validates architecture/preprocessing semantics, logs resolved
  components, writes resolved config, and only then builds data/model/optimizer.
  Existing source semantic mismatch rejection remains unconditional.
- `persisted_config_from_args()` removes all underscore-prefixed parser state.
  Source/EMA config JSON, source checkpoints, and EMA checkpoint config now
  use it; regression tests verify no explicit/private marker reaches persisted
  config. Source persistence uses eight distinct `1.1`--`1.8` values. EMA
  starts from distinct source sentinels `9.1`--`9.8` and verifies every saved
  weight comes from current EMA args.
- Test count reconciliation: this task started from a fresh current-HEAD
  collect-only result of `152 tests`. After adding ten collected cases,
  `PYTHONDONTWRITEBYTECODE=1 ... pytest -p no:cacheprovider --collect-only -q`
  reported `162 tests collected in 3.52s`, and the full run reported
  `162 passed, 5 warnings in 15.74s`. The earlier repeated `152` records are
  historical/pre-this-task counts and must not be read as a post-change count
  for this source-resume work. The five warnings remain the four upstream
  torchvision Pillow deprecations and the existing non-collectable
  `TestDataset` warning.

## 2026-07-25 -- canonical loss layering

- Baseline HEAD: `7274408285dc0ebf86b6affe3dfb79cb75ca7558` on `v4`; only the
  existing user untracked resources were present. The prescribed loss/flow
  baseline reported `22 passed, 4 warnings`.
- Pre-move deterministic reference (`torch.manual_seed(31415)`): Source total
  `3.9984745979`, q sum `10.2569599152`, real `L_real` `1.3981907368`; Source
  prediction/density/route gradient norms were `0.9628067613`, `0.2019856274`,
  `21.4344787598`, and real J/M/R norms were `0.1587713212`,
  `0.2249999940`, `4.1881737709`. Canonical regression coverage preserves
  these values, detached q, return fields and gradient connectivity.
- Moved canonical implementations: old `loss/fog_routed_source_loss.py` ->
  `loss/common/masked.py`, `loss/common/structural.py`, and
  `loss/synthetic/routing.py`; `training/source_objective.py` ->
  `loss/synthetic/objective.py`; local counterfactual error/q code ->
  `loss/synthetic/counterfactual.py`; `training/ema_core.py` loss math ->
  `loss/real/consistency.py`; and EMA adaptation composition ->
  `loss/real/objective.py`.
- Compatibility files retained: `loss/fog_routed_source_loss.py` and
  `training/source_objective.py` are explicit re-exports; `training/ema_core.py`
  retains only EMA state updates and imports legacy loss names directly from
  `loss.real.consistency`. No legacy module contains a second loss formula.
- `KL.py`, `Dice.py`, and `mssim.py` remain untouched as old independently
  importable loss modules. They have no in-repository consumer, but removing
  them is intentionally outside this task. `loss/__init__.py` no longer uses
  wildcard imports and exposes only explicit formal APIs.
- Source loss defaults, formal preset, explicit CLI tracking and validation then
  lived in `option/Teacher.py`; EMA reused the Source-anchor registration and
  validation functions while keeping EMA-real weights local.
- Static import coverage verifies that `loss/` imports no model, training,
  option or entrypoint package. The requested formula scan found canonical
  definitions only under `loss/` and no q/BCE/binary formula matches in
  training, model or entrypoint modules.
- Added a no-commit Source/EMA chain smoke: real consistency, Source anchor,
  adaptation objective and `run_ema_adaptation_step` all backpropagate; failed
  optimizer submission leaves teacher state unchanged and emits detached
  `L_real`, `L_src`, `L_adapt`.
- Verification: focused legacy/formal loss flow baseline after migration,
  compatibility and import tests passed; complete pytest with cache/bytecode
  writes disabled reported `168 passed, 5 warnings in 12.13s`. Warnings remain
  four upstream torchvision Pillow deprecations and the existing non-collected
  `TestDataset` class warning.

## 2026-07-27 -- stage-local configuration ownership

- Source configuration is now self-contained in `option/Teacher.py`; real-domain
  EMA configuration is independently implemented in `option/EMA.py`. Importing
  either module remains side-effect free.
- EMA inherits model, preprocessing, route, Omega, Source-anchor loss and
  train-size semantics from the selected checkpoint before full validation;
  its parser exposes only EMA runtime/training controls and
  `source_anchor_data_dir`.
- New EMA checkpoints store `source_anchor_data_dir` and omit the historical
  source-path key. When loading an old checkpoint, its historical source path
  is mapped to the new anchor-path field; an explicitly supplied new path wins.
- Legacy Source checkpoints missing training-objective fields receive a warned
  historical L1 compatibility profile, while all present checkpoint weights
  remain unchanged.
