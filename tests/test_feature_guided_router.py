import torch

from model.feature_guided_router import FeatureGuidedRouter


def _features(height=5, width=5):
    return {
        "fm_vis": torch.zeros(1, 96, height, width),
        "fm_ir": torch.zeros(1, 96, height, width),
        "struct_diff_gap": torch.zeros(1, 1, height, width),
        "struct_diff_gmp": torch.zeros(1, 1, height, width),
    }


def test_feature_guided_router_outputs_a_route_from_hde_features():
    torch.manual_seed(1)
    router = FeatureGuidedRouter(hidden_channels=8)
    density = torch.zeros(1, 1, 5, 5, requires_grad=True)
    features = _features()
    features["fm_vis"].requires_grad_()

    output = router(density, features, temperature=0.7)

    assert set(output) == {"route_logits", "route_soft", "route_hard"}
    assert output["route_soft"].shape == density.shape
    output["route_soft"].sum().backward()
    assert features["fm_vis"].grad is not None


@torch.no_grad()
def test_feature_guided_router_uses_hde_features_and_local_context():
    router = FeatureGuidedRouter(hidden_channels=1)
    for parameter in router.parameters():
        parameter.zero_()
    router.head[0].weight[0, 1] = 1.0  # fm_vis channel zero
    router.head[2].weight[0, 0].fill_(1.0)
    router.head[4].weight[0, 0, 0, 0] = 1.0
    density = torch.zeros(1, 1, 5, 5)
    features = _features()
    baseline = router(density, features)["route_logits"]
    features["fm_vis"][0, 0, 2, 3] = 1.0
    changed = router(density, features)["route_logits"]
    assert changed[0, 0, 2, 2] > baseline[0, 0, 2, 2]


def test_feature_guided_router_temperature_and_hard_route_are_non_straight_through():
    router = FeatureGuidedRouter(hidden_channels=2)
    output = router(torch.zeros(1, 1, 3, 3), _features(3, 3), temperature=0.5)
    assert torch.equal(output["route_hard"], (output["route_soft"] >= 0.5).float())
    assert output["route_hard"].requires_grad is False
