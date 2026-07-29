import torch

from model import FogRoutedRGBTIRDehazer


def test_inference_decodes_with_predicted_hard_route_only():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), route_mode="hard")
    assert output["pred_clear"].shape == (1, 3, 32, 32)
    assert torch.equal(output["route_hard"], (output["route_soft"] >= 0.5).float())


def test_training_gate_is_a_full_differentiable_route_override():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    context = model.encode_context(torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32))
    gate = context["route_soft"]
    output = model.decode_with_route(context, route_mode="soft", route_override_value=gate)
    output["pred_clear"].mean().backward()
    assert gate.grad_fn is not None
