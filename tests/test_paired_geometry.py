import torch

from training.paired_geometry import GeometryTransform


def test_batch_geometry_round_trip_handles_non_square_rotations_and_flip():
    source = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7)
    transform = GeometryTransform(rot90_k=1, horizontal_flip=True)

    restored = transform.inverse(transform.apply(source))

    assert torch.equal(restored, source)
    assert transform.apply(source).shape[-2:] == (7, 5)
