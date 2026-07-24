import torch
import pytest

from model.deform_sampler import ModulatedDeformSampler


def test_pure_pytorch_modulated_deform_sampler_runs_cpu_forward_and_backward():
    sampler = ModulatedDeformSampler(2, 3, kernel_size=3, padding=1, backend="pure_pytorch")
    value = torch.rand(1, 2, 7, 9, requires_grad=True)
    offset = torch.zeros(1, 18, 7, 9, requires_grad=True)
    mask = torch.ones(1, 9, 7, 9, requires_grad=True)

    output = sampler(value, offset, mask)
    output.mean().backward()

    assert output.shape == (1, 3, 7, 9)
    assert value.grad is not None and offset.grad is not None and mask.grad is not None


def test_torchvision_backend_matches_pure_backend_when_available():
    try:
        from torchvision.ops import deform_conv2d  # noqa: F401
    except (ImportError, RuntimeError):
        pytest.skip("torchvision deformable convolution backend is unavailable")

    pure = ModulatedDeformSampler(2, 3, kernel_size=3, padding=1, backend="pure_pytorch")
    torchvision = ModulatedDeformSampler(2, 3, kernel_size=3, padding=1, backend="torchvision")
    torchvision.load_state_dict(pure.state_dict(), strict=True)
    value = torch.rand(1, 2, 7, 9, requires_grad=True)
    offset = (torch.randn(1, 18, 7, 9) * 0.1).requires_grad_()
    mask = torch.sigmoid(torch.randn(1, 9, 7, 9)).requires_grad_()
    value_t = value.detach().clone().requires_grad_(True)
    offset_t = offset.detach().clone().requires_grad_(True)
    mask_t = mask.detach().clone().requires_grad_(True)

    try:
        actual = torchvision(value_t, offset_t, mask_t)
    except RuntimeError as error:
        pytest.skip(f"torchvision deformable convolution backend unavailable: {error}")
    expected = pure(value, offset, mask)
    assert torch.allclose(actual, expected, rtol=2e-4, atol=2e-5)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for left, right in (
        (value_t.grad, value.grad), (offset_t.grad, offset.grad), (mask_t.grad, mask.grad),
        (torchvision.weight.grad, pure.weight.grad),
    ):
        assert torch.allclose(left, right, rtol=3e-4, atol=3e-5)
