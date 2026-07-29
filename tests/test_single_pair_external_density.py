from pathlib import Path

import pytest
import torch
from PIL import Image

from test_single_pair_external_hde import (
    build_parser,
    inject_external_density,
    load_external_transmission_map,
)


class _Router:
    def __call__(self, density_map, temperature):
        return {
            "route_logits": density_map + float(temperature),
            "route_soft": density_map * 0.5,
            "route_hard": (density_map > 0.5).to(density_map.dtype),
        }


def test_external_transmission_is_inverted_and_requires_exact_size(tmp_path):
    path = Path(tmp_path) / "transmission.png"
    Image.new("L", (5, 3), color=64).save(path)

    transmission, density = load_external_transmission_map(path, expected_size=(3, 5))

    assert transmission.shape == density.shape == (1, 1, 3, 5)
    assert torch.allclose(density, 1.0 - transmission)
    with pytest.raises(ValueError, match="size mismatch"):
        load_external_transmission_map(path, expected_size=(4, 5))


def test_injection_replaces_only_density_and_recomputes_route():
    original = torch.zeros(1, 1, 2, 2)
    context = {
        "density_map": original,
        "route_logits": original,
        "route_soft": original,
        "route_hard": original,
        "tir_structure_pyramid": {"h2": torch.ones(1, 32, 1, 1)},
    }
    density = torch.tensor([[[[0.0, 0.25], [0.75, 1.0]]]])

    injected = inject_external_density(context, density, _Router(), route_temperature=0.2)

    assert injected is not context
    assert injected["tir_structure_pyramid"] is context["tir_structure_pyramid"]
    assert torch.equal(context["density_map"], original)
    assert torch.equal(injected["density_map"], density)
    assert torch.equal(injected["route_soft"], density * 0.5)
    assert torch.equal(injected["route_hard"], (density > 0.5).float())


def test_external_transmission_argument_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "--checkpoint", "checkpoint.pt", "--hazy", "hazy.png", "--tir", "tir.png", "--output_dir", "out"
        ])
