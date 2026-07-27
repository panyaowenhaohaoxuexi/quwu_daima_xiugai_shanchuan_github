import pytest
import torch

from model.structure_appearance_transformer import StructureAppearanceTransformerBlock


def test_transformer_masks_padding_and_all_invalid_windows_without_nonfinite_values():
    torch.manual_seed(17)
    block = StructureAppearanceTransformerBlock(
        channels=8, num_heads=4, window_size=3, window_chunk_size=2,
    ).eval()
    structure = torch.randn(2, 8, 5, 7, requires_grad=True)
    appearance = torch.randn(2, 8, 5, 7, requires_grad=True)
    validity = torch.ones(2, 1, 5, 7)
    validity[0, :, :, 6:] = 0
    validity[1] = 0

    output = block(structure, appearance, validity)

    assert output.shape == structure.shape
    assert torch.isfinite(output).all()
    assert torch.equal(output[1], torch.zeros_like(output[1]))
    assert torch.equal(output[0, :, :, 6:], torch.zeros_like(output[0, :, :, 6:]))
    output.sum().backward()
    assert torch.isfinite(structure.grad).all()
    assert torch.isfinite(appearance.grad).all()


def test_transformer_window_chunking_has_matching_values_and_gradients():
    torch.manual_seed(23)
    reference = StructureAppearanceTransformerBlock(8, 4, 3, 100).eval()
    chunked = StructureAppearanceTransformerBlock(8, 4, 3, 2).eval()
    chunked.load_state_dict(reference.state_dict())
    structure = torch.randn(2, 8, 5, 7)
    appearance = torch.randn(2, 8, 5, 7)
    validity = torch.ones(2, 1, 5, 7)

    def run(module):
        q, kv = structure.clone().requires_grad_(True), appearance.clone().requires_grad_(True)
        output = module(q, kv, validity)
        output.square().mean().backward()
        return output.detach(), q.grad.detach(), kv.grad.detach()

    actual = run(chunked)
    expected = run(reference)
    for value, target in zip(actual, expected):
        torch.testing.assert_close(value, target, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kwargs", [
    {"channels": 0}, {"num_heads": 0}, {"channels": 6, "num_heads": 4},
    {"window_size": 0}, {"window_chunk_size": 0}, {"mlp_ratio": 0},
    {"attention_dropout": 1.0}, {"projection_dropout": -0.1}, {"ffn_dropout": 1.0},
])
def test_transformer_rejects_invalid_direct_construction(kwargs):
    values = {"channels": 8, "num_heads": 4, "window_size": 3, "window_chunk_size": 2}
    values.update(kwargs)
    with pytest.raises(ValueError):
        StructureAppearanceTransformerBlock(**values)
