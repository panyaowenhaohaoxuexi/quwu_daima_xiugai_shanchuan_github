# CoA-Style Loss Module Layout

## Goal

Split V2 Source reconstruction losses into focused modules following the CoA layout while keeping physical-mask density and route supervision unchanged.

## Layout

- `loss/ssim.py` provides CoA-equivalent Gaussian-window `SSIM`.
- `loss/cr.py` provides the CoA VGG19 contrast/CR formula, without hard-coded CUDA allocation.
- `loss/regional.py` provides mask-aware regional L1, gradient, and local-SSIM reconstruction terms.
- `loss/source.py` owns only V2 Source objective composition and criterion construction.
- `loss/__init__.py` re-exports public loss APIs.

## Compatibility

The global and regional coefficients remain unchanged. `loss/source.py` keeps the public Source loss entry point. Source and EMA create the criteria once and move them to the selected device. No Omega, q, counterfactual, or old route-supervision code is reintroduced.
