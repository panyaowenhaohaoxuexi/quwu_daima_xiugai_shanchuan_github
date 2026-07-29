# V2 Physical-Mask Routing Design

## Goal

Replace the mixed legacy counterfactual/Omega route-supervision path with one V2 physical-mask routing path for synthetic Source training and EMA source anchors.

## Contracts

`SynthMultiModalDataset` requires a stem-matched `mask_GT/<haze_level>/<stem>.png` and always returns `(hazy, clear, tir, density_gt, completion_mask_gt)`.  Density defaults to the stored `M=1-T` semantics. RGB, TIR, and density use bilinear geometry; the binary completion mask uses nearest-neighbour geometry.

HDE formally returns density, its structure pyramid, routing features (`fm_vis`, `fm_ir`, `struct_diff_gap`, `struct_diff_gmp`), and `debug=None`.  The router consumes only that feature set plus density and produces route logits, sigmoid soft routes, and thresholded hard routes.

## Training and inference

At Source and EMA-anchor training time, temperature is linearly annealed and decoding receives the full continuous gate `alpha * completion_mask_gt + (1-alpha) * route_soft`.  `alpha` decreases linearly from one to zero over `route_teacher_anneal_steps`.  Inference has no ground-truth mask and decodes with the router hard route.

The only Source objective is reconstruction Charbonnier loss plus weighted SmoothL1 density loss and class-balanced BCE-with-logits mask loss.  The BCE positive weight is dynamically computed from the current batch and remains finite for all-zero and all-one masks.

## Compatibility and observability

Counterfactual, Omega, q, binary-route, boundary-route and monotonic-router configuration/implementation paths are removed.  Checkpoint preflight rejects legacy router state or legacy route configuration with a clear V2-incompatibility error.  Validation ignores the fifth batch element; diagnostics show hazy, TIR, clear, prediction, M_GT, M_pred, mask_GT, route_soft, and route_hard.

## Acceptance

Focused data, HDE, router, loss, gate-annealing, entry-point, visualization, and checkpoint tests pass, followed by the full test suite.  No active package import, parser option, or training path retains the removed legacy route chain.
