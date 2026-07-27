def test_refactored_source_module_exposes_the_complete_batch_api():
    from training.source import OmegaSampler, compute_source_batch_losses, source_route_schedule

    assert callable(compute_source_batch_losses)
    assert callable(source_route_schedule)
    assert OmegaSampler.__name__ == "OmegaSampler"
