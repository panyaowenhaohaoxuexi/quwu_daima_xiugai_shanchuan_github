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
        parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if name in {
                "q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight",
                "relative_position_bias", "ffn.0.weight", "ffn.3.weight",
            }
        }
        return output.detach(), q.grad.detach(), kv.grad.detach(), parameter_grads

    actual = run(chunked)
    expected = run(reference)
    for value, target in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(value, target, atol=1e-6, rtol=1e-5)
    for name in expected[3]:
        torch.testing.assert_close(actual[3][name], expected[3][name], atol=1e-6, rtol=1e-5)
        assert torch.isfinite(actual[3][name]).all()
    assert actual[3]["relative_position_bias"].abs().sum() > 0


def test_transformer_never_projects_more_windows_than_the_configured_chunk_size():
    block = StructureAppearanceTransformerBlock(8, 4, 3, 2).eval()
    seen = []
    handle = block.q_proj.register_forward_pre_hook(lambda _m, values: seen.append(values[0].shape[0]))
    try:
        block(torch.randn(2, 8, 7, 8), torch.randn(2, 8, 7, 8), torch.ones(2, 1, 7, 8))
    finally:
        handle.remove()
    assert seen
    assert max(seen) <= block.window_chunk_size


@pytest.mark.parametrize("extra_height,extra_width", [(0, 2), (1, 0), (1, 2)])
def test_transformer_right_bottom_padding_preserves_existing_valid_region(extra_height, extra_width):
    torch.manual_seed(31)
    block = StructureAppearanceTransformerBlock(8, 4, 3, 2).eval()
    structure, appearance = torch.randn(1, 8, 5, 7), torch.randn(1, 8, 5, 7)
    reference = block(structure, appearance, torch.ones(1, 1, 5, 7))
    padded_structure = torch.nn.functional.pad(structure, (0, extra_width, 0, extra_height))
    padded_appearance = torch.nn.functional.pad(appearance, (0, extra_width, 0, extra_height))
    validity = torch.nn.functional.pad(torch.ones(1, 1, 5, 7), (0, extra_width, 0, extra_height))
    padded = block(padded_structure, padded_appearance, validity)
    torch.testing.assert_close(padded[..., :5, :7], reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("kwargs", [
    {"channels": 0}, {"num_heads": 0}, {"channels": 6, "num_heads": 4},
    {"window_size": 0}, {"window_chunk_size": 0}, {"mlp_ratio": 0},
    {"window_size": 0.5}, {"window_chunk_size": 0.5}, {"window_size": True},
    {"window_chunk_size": False}, {"window_size": float("nan")},
    {"window_chunk_size": float("inf")}, {"window_size": "7"},
    {"attention_dropout": 1.0}, {"projection_dropout": -0.1}, {"ffn_dropout": 1.0},
])
def test_transformer_rejects_invalid_direct_construction(kwargs):
    values = {"channels": 8, "num_heads": 4, "window_size": 3, "window_chunk_size": 2}
    values.update(kwargs)
    with pytest.raises(ValueError):
        StructureAppearanceTransformerBlock(**values)
