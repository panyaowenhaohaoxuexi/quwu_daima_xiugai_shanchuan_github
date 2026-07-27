import torch


def test_simple_source_checkpoint_has_only_epoch_resume_state():
    from utils.checkpoint import build_source_checkpoint, load_source_checkpoint
    from model.Teacher import FogRoutedRGBTIRDehazer

    model = FogRoutedRGBTIRDehazer(base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = build_source_checkpoint(model, optimizer, epoch=3, global_step=17, config={"base_channels": 8})

    assert set(checkpoint) == {"training_stage", "model", "optimizer", "epoch", "global_step", "config"}
    restored_model = FogRoutedRGBTIRDehazer(base_channels=8)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    state = load_source_checkpoint(checkpoint, restored_model, restored_optimizer)
    assert state == {"epoch": 3, "global_step": 17}
