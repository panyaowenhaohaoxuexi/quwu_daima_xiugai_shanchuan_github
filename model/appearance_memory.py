"""RGB appearance retrieval and TIR-conditioned appearance priors."""

import torch
from torch import nn
from torch.nn import functional as F


class TIRConditionedAppearancePrior(nn.Module):
    """Generate appearance only from local and global TIR structure."""

    def __init__(self, structure_channels, appearance_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(2 * structure_channels, appearance_channels, 1), nn.SiLU(inplace=True),
            nn.Conv2d(appearance_channels, appearance_channels, 1),
        )

    def forward(self, structure, validity):
        valid_mass = validity.sum(dim=(1, 2, 3)).clamp_min(1.0)
        global_context = (structure * validity).sum(dim=(2, 3)) / valid_mass[:, None]
        context = global_context[:, :, None, None].expand(-1, -1, *structure.shape[-2:])
        return self.layers(torch.cat((structure, context), dim=1)) * validity.to(dtype=structure.dtype)


class MemoryRetriever(nn.Module):
    """Retrieve reliable RGB appearance values and blend them with a TIR prior."""

    def __init__(self, channels, structure_channels, max_tokens, topk, attention_temperature,
                 reliability_epsilon, ratio_threshold, confidence_threshold, query_chunk_size=1024):
        super().__init__()
        checks = (("max_tokens", max_tokens, lambda value: int(value) > 0),
                  ("topk", topk, lambda value: 2 <= int(value) <= int(max_tokens)),
                  ("query_chunk_size", query_chunk_size, lambda value: int(value) > 0),
                  ("attention_temperature", attention_temperature, lambda value: float(value) > 0),
                  ("reliability_epsilon", reliability_epsilon, lambda value: float(value) > 0),
                  ("ratio_threshold", ratio_threshold, lambda value: 0 <= float(value) <= 1),
                  ("confidence_threshold", confidence_threshold, lambda value: 0 <= float(value) <= 1))
        for name, value, valid in checks:
            if not valid(value):
                raise ValueError(f"invalid {name}={value}")
        self.max_tokens, self.topk, self.query_chunk_size = int(max_tokens), int(topk), int(query_chunk_size)
        self.attention_temperature, self.reliability_epsilon = float(attention_temperature), float(reliability_epsilon)
        self.ratio_threshold, self.confidence_threshold = float(ratio_threshold), float(confidence_threshold)
        self.key = nn.Conv2d(structure_channels, channels, 1)
        self.query = nn.Conv2d(structure_channels, channels, 1)
        self.context_key = nn.Linear(structure_channels, channels, bias=False)
        self.context_query = nn.Linear(structure_channels, channels, bias=False)
        self.prior = TIRConditionedAppearancePrior(structure_channels, channels)

    def retrieval_gate(self, confidence, reliable_ratio, candidate_count, validity):
        """Continuous retrieval confidence with explicit per-sample ratio broadcasting."""
        eps = torch.finfo(confidence.dtype).eps
        ratio_scale = max(float(self.ratio_threshold), eps)
        ratio_gate = (reliable_ratio / (reliable_ratio + ratio_scale)).view(-1, 1, 1, 1)
        has_memory = (candidate_count > 0).to(confidence.dtype)
        coverage_gate = (candidate_count.to(confidence.dtype) / float(self.topk)).clamp(0.0, 1.0)
        return (has_memory * confidence * ratio_gate * coverage_gate * validity).clamp(0.0, 1.0)

    def forward(self, structure, value, reliability, validity):
        batch, channels, height, width = value.shape
        valid_mass = validity.sum(dim=(1, 2, 3))
        reliable_mass = reliability.sum(dim=(1, 2, 3))
        reliable_ratio = reliable_mass / valid_mass.clamp_min(1.0)
        global_context = (structure * validity).sum(dim=(2, 3)) / valid_mass[:, None].clamp_min(1.0)
        key = self.key(structure) + self.context_key(global_context).view(batch, channels, 1, 1)
        query = self.query(structure) + self.context_query(global_context).view(batch, channels, 1, 1)
        key = F.normalize(key.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        query = F.normalize(query.flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        values = value.flatten(2).transpose(1, 2)
        reliability_flat = reliability.flatten(1)
        retrieved = torch.zeros_like(values)
        confidence = torch.zeros(batch, height * width, device=value.device, dtype=value.dtype)
        candidate_count = torch.zeros(batch, height * width, device=value.device, dtype=torch.long)
        for item in range(batch):
            valid_indices = torch.where(reliability_flat[item] > self.reliability_epsilon)[0]
            if valid_indices.numel() == 0:
                continue
            if valid_indices.numel() > self.max_tokens:
                positions = torch.linspace(0, valid_indices.numel() - 1, self.max_tokens, device=value.device).round().long()
                valid_indices = valid_indices[positions]
            item_key, item_value = key[item, valid_indices], values[item, valid_indices]
            reliability_bias = reliability_flat[item, valid_indices].clamp_min(self.reliability_epsilon).log().unsqueeze(0)
            top_count = min(self.topk, valid_indices.numel())
            for query_start in range(0, query.shape[1], self.query_chunk_size):
                query_end = min(query_start + self.query_chunk_size, query.shape[1])
                scores = query[item, query_start:query_end] @ item_key.t() / self.attention_temperature + reliability_bias
                top_scores, top_indices = torch.topk(scores, k=top_count, dim=-1)
                attention = torch.softmax(top_scores, dim=-1)
                retrieved[item, query_start:query_end] = (attention.unsqueeze(-1) * item_value[top_indices]).sum(dim=1)
                candidate_count[item, query_start:query_end].fill_(top_count)
                if top_count == 1:
                    confidence[item, query_start:query_end].fill_(1.0)
                else:
                    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
                    confidence[item, query_start:query_end] = (1.0 - entropy / torch.log(torch.tensor(float(top_count), device=value.device))).clamp(0, 1)
        retrieved = retrieved.transpose(1, 2).reshape(batch, channels, height, width)
        confidence = confidence.reshape(batch, 1, height, width) * validity
        candidate_count = candidate_count.reshape(batch, 1, height, width)
        candidate_count = torch.where(validity > 0, candidate_count, torch.zeros_like(candidate_count))
        fallback = ((reliable_ratio < self.ratio_threshold).view(batch, 1, 1, 1) |
                    (confidence < self.confidence_threshold)).to(value.dtype) * validity
        retrieval_gate = self.retrieval_gate(confidence, reliable_ratio, candidate_count, validity)
        prior = self.prior(structure, validity)
        appearance = (retrieval_gate * retrieved + (1.0 - retrieval_gate) * prior) * validity
        return appearance, confidence, fallback, reliable_mass, reliable_ratio, candidate_count, retrieval_gate
