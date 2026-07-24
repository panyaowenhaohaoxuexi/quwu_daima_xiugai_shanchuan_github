import torch

from model.hde import HDE


def test_hde_returns_shared_ir_structure_pyramid_with_gradients():
    hde = HDE()
    rgb = torch.rand(1, 3, 32, 32)
    tir = torch.rand(1, 3, 32, 32, requires_grad=True)

    output = hde(rgb, tir)

    assert set(output) == {"density_map", "tir_structure_pyramid", "debug"}
    pyramid = output["tir_structure_pyramid"]
    assert tuple(pyramid) == ("h2", "h4", "h8", "h16")
    assert pyramid["h2"].shape[-2:] == (16, 16)
    assert pyramid["h4"].shape[-2:] == (8, 8)
    assert pyramid["h8"].shape[-2:] == (4, 4)
    assert pyramid["h16"].shape[-2:] == (2, 2)
    assert output["debug"] is None

    (output["density_map"].mean() + pyramid["h2"].mean()).backward()
    assert tir.grad is not None
    assert torch.isfinite(tir.grad).all()
    assert sum(1 for module in hde.modules() if module.__class__.__name__ == "IRDifferenceStructureEncoder") == 1
