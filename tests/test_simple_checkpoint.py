import torch


def _model_config():
    return {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 256,
        "memory_topk": 8, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "decoder_num_heads": 4, "decoder_depth": 1,
        "decoder_window_size": 7, "decoder_window_chunk_size": 128, "decoder_mlp_ratio": 4.0,
        "decoder_attention_dropout": 0.0, "decoder_projection_dropout": 0.0,
        "decoder_ffn_dropout": 0.0,
    }


def test_simple_source_checkpoint_has_only_epoch_resume_state():
    from utils.checkpoint import build_source_checkpoint, load_source_checkpoint
    from model.Teacher import FogRoutedRGBTIRDehazer

    model = FogRoutedRGBTIRDehazer(base_channels=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = build_source_checkpoint(model, optimizer, epoch=3, global_step=17, config=_model_config())

    assert set(checkpoint) == {"format_version", "training_stage", "model", "optimizer", "epoch", "global_step", "config"}
    restored_model = FogRoutedRGBTIRDehazer(base_channels=8)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-4)
    state = load_source_checkpoint(checkpoint, restored_model, restored_optimizer)
    assert state == {"epoch": 3, "global_step": 17}
