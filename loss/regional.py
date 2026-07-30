"""Mask-aware reconstruction losses for fusion, completion, and boundary regions."""

from .common import masked_gradient_error, masked_local_ssim_error, masked_mean


def regional_reconstruction_error(prediction, target, support, *, l1_weight=1.0,
                                  gradient_weight=0.2, ssim_weight=0.2,
                                  ssim_window=7, min_valid_support=4):
    """Original regional CoA weighting: L1 + gradient + local SSIM."""
    return (float(l1_weight) * masked_mean((prediction - target).abs(), support) +
            float(gradient_weight) * masked_gradient_error(prediction, target, support) +
            float(ssim_weight) * masked_local_ssim_error(
                prediction, target, support, window_size=ssim_window,
                min_valid_support=min_valid_support,
            ))

