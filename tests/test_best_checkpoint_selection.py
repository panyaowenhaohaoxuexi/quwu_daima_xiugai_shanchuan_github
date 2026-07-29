import torch


def test_validation_uses_paired_clear_targets_and_returns_mean_psnr_ssim():
    from training.validation import evaluate_paired_validation

    class IdentityModel(torch.nn.Module):
        def forward(self, hazy, _tir, **_kwargs):
            return {"pred_clear": hazy}

    batch = (torch.full((1, 3, 8, 8), 0.6), torch.full((1, 3, 8, 8), 0.6),
             torch.zeros((1, 3, 8, 8)), torch.zeros((1, 1, 8, 8)), torch.zeros((1, 1, 8, 8)))
    metrics = evaluate_paired_validation(IdentityModel(), [batch], torch.device("cpu"), route_temperature=0.2)

    assert metrics["psnr"] > 100
    assert metrics["ssim"] == 1.0


def test_best_checkpoint_is_replaced_only_when_psnr_improves(tmp_path):
    from training.validation import save_best_if_improved

    path = tmp_path / "best.pt"
    best = save_best_if_improved(10.0, float("-inf"), {"tag": "first"}, path)
    assert best == 10.0
    assert torch.load(path, map_location="cpu")["tag"] == "first"

    best = save_best_if_improved(9.0, best, {"tag": "worse"}, path)
    assert best == 10.0
    assert torch.load(path, map_location="cpu")["tag"] == "first"


def test_formal_checkpoints_persist_best_validation_psnr_for_resume():
    from option.Teacher import build_parser, persisted_config_from_args, validate_config
    from utils.checkpoint import build_source_checkpoint

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = build_source_checkpoint(model, optimizer, epoch=1, global_step=2,
                                         config=persisted_config_from_args(validate_config(build_parser().parse_args([]))),
                                         best_psnr=23.5)

    assert checkpoint["best_psnr"] == 23.5
