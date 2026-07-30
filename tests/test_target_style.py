import torch

from training.target_style import apply_target_statistics


def test_target_statistics_beta_zero_is_strict_identity():
    hazy = torch.rand(2, 3, 8, 10)
    clear = torch.rand(2, 3, 8, 10)
    tir = torch.rand(2, 3, 8, 10)

    styled = apply_target_statistics(
        hazy, clear, tir, hazy.flip(-1), tir.flip(-2), beta=0.0
    )

    assert torch.equal(styled.hazy, hazy)
    assert torch.equal(styled.clear, clear)
    assert torch.equal(styled.tir, tir)
    assert torch.equal(styled.rgb_gain, torch.ones_like(styled.rgb_gain))
    assert torch.equal(styled.rgb_bias, torch.zeros_like(styled.rgb_bias))
    assert torch.equal(styled.tir_gain, torch.ones_like(styled.tir_gain))
    assert torch.equal(styled.tir_bias, torch.zeros_like(styled.tir_bias))


def test_target_statistics_uses_the_same_rgb_affine_map_for_hazy_and_clear():
    hazy = torch.linspace(0.1, 0.4, 48).reshape(1, 3, 4, 4)
    clear = torch.linspace(0.3, 0.8, 48).reshape(1, 3, 4, 4)
    tir = torch.linspace(0.0, 1.0, 16).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
    target_hazy = torch.linspace(0.45, 0.85, 126).reshape(1, 3, 6, 7)
    target_tir = torch.linspace(0.2, 0.8, 42).reshape(1, 1, 6, 7).repeat(1, 3, 1, 1)

    styled = apply_target_statistics(hazy, clear, tir, target_hazy, target_tir, beta=1.0)

    assert torch.allclose(
        styled.hazy, (styled.rgb_gain * hazy + styled.rgb_bias).clamp(0.0, 1.0)
    )
    assert torch.allclose(
        styled.clear, (styled.rgb_gain * clear + styled.rgb_bias).clamp(0.0, 1.0)
    )
    assert torch.allclose(
        styled.tir, (styled.tir_gain * tir + styled.tir_bias).clamp(0.0, 1.0)
    )
    assert torch.all(styled.tir[..., 1:] >= styled.tir[..., :-1])
    assert (styled.rgb_gain >= 0.75).all() and (styled.rgb_gain <= 1.35).all()
