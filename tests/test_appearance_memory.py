import torch

from model.appearance_memory import MemoryRetriever, TIRConditionedAppearancePrior


def _memory():
    return MemoryRetriever(2, 2, max_tokens=8, topk=4, attention_temperature=0.1,
                           reliability_epsilon=1e-6, ratio_threshold=0.1, confidence_threshold=0.1)


def test_memory_gate_is_batch_safe_continuous_and_coverage_limited():
    memory = _memory()
    confidence = torch.ones(2, 1, 3, 5)
    candidate_count = torch.full((2, 1, 3, 5), 4, dtype=torch.long)
    validity = torch.ones_like(confidence)
    ratio = torch.tensor([0.01, 1.0])
    gate = memory.retrieval_gate(confidence, ratio, candidate_count, validity)
    assert gate.shape == (2, 1, 3, 5)
    assert torch.all(gate[0] < gate[1])
    single = memory.retrieval_gate(confidence, ratio, torch.ones_like(candidate_count), validity)
    assert torch.all(single < gate)
    empty = memory.retrieval_gate(confidence, ratio, torch.zeros_like(candidate_count), validity)
    assert torch.equal(empty, torch.zeros_like(empty))
    near = memory.retrieval_gate(confidence * 0.51, ratio, candidate_count, validity)
    before = memory.retrieval_gate(confidence * 0.50, ratio, candidate_count, validity)
    assert torch.all(near > before)
    assert torch.isfinite(gate).all()


def test_memory_retriever_uses_attention_entropy_for_deterministic_gate_ordering():
    memory = MemoryRetriever(2, 2, max_tokens=4, topk=4, attention_temperature=0.1,
                             reliability_epsilon=1e-6, ratio_threshold=0.01, confidence_threshold=0.1).eval()
    with torch.no_grad():
        memory.key.weight.copy_(torch.eye(2).view(2, 2, 1, 1)); memory.key.bias.zero_()
        memory.query.weight.copy_(torch.eye(2).view(2, 2, 1, 1)); memory.query.bias.zero_()
        memory.context_key.weight.zero_(); memory.context_query.weight.zero_()
    # The first query has one strongly matching key; later queries have three equal matches.
    structure = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 1.0, 1.0]]]])
    values = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)
    validity = reliability = torch.ones(1, 1, 1, 4)
    _, confidence, _, _, _, count, gate = memory(structure, values, reliability, validity)
    assert int(count[0, 0, 0, 0]) == 4
    assert confidence[0, 0, 0, 0] > confidence[0, 0, 0, 1]
    assert gate[0, 0, 0, 0] > gate[0, 0, 0, 1]


def test_memory_and_prior_zero_invalid_padding_and_preserve_valid_region_when_padded():
    torch.manual_seed(31)
    memory = _memory().eval()
    prior = TIRConditionedAppearancePrior(2, 2).eval()
    structure = torch.randn(2, 2, 3, 5, requires_grad=True)
    values = torch.randn(2, 2, 3, 5, requires_grad=True)
    reliability = torch.ones(2, 1, 3, 5)
    validity = torch.ones(2, 1, 3, 5)
    reference = memory(structure, values, reliability, validity)
    prior_reference = prior(structure, validity)
    padded_structure = torch.nn.functional.pad(structure, (0, 2, 0, 1))
    padded_values = torch.nn.functional.pad(values, (0, 2, 0, 1))
    padded_reliability = torch.nn.functional.pad(reliability, (0, 2, 0, 1))
    padded_validity = torch.nn.functional.pad(validity, (0, 2, 0, 1))
    actual = memory(padded_structure, padded_values, padded_reliability, padded_validity)
    prior_actual = prior(padded_structure, padded_validity)
    for item, expected in zip((actual[0], actual[1], actual[5], actual[6], prior_actual),
                              (reference[0], reference[1], reference[5], reference[6], prior_reference)):
        torch.testing.assert_close(item[..., :3, :5], expected, atol=1e-6, rtol=1e-5)
    for item in (actual[0], actual[1], actual[6], prior_actual):
        assert torch.equal(item[..., 3:, :], torch.zeros_like(item[..., 3:, :]))
        assert torch.equal(item[..., :, 5:], torch.zeros_like(item[..., :, 5:]))
    assert torch.equal(actual[5][..., 3:, :], torch.zeros_like(actual[5][..., 3:, :]))
    assert torch.equal(actual[5][..., :, 5:], torch.zeros_like(actual[5][..., :, 5:]))
    actual[0].sum().backward()
    assert torch.isfinite(structure.grad).all()
    assert torch.isfinite(values.grad).all()
