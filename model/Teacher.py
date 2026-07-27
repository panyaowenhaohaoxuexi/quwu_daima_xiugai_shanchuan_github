"""Formal RGB--TIR fog-routed dehazer.

The module deliberately separates reusable encoding from route-dependent decoding
so counterfactual candidates never rerun HDE or either encoder.
"""

import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .hde import HDE
from .monotonic_router import MonotonicFogRouter


def _groups(channels):
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


def appearance_receptive_field_radius_by_scale(deform_max_offset, extra_margin=0):
    """Conservative RGB-value receptive-field radii in each token grid.

    The recurrence follows ``PyramidEncoder`` exactly: stem 3x3, one stride-2
    3x3 downsample, then the two 3x3 layers in each residual block.  A value
    can additionally travel through the local sampler, a 3x3 renderer and the
    two 3x3 fusion-residual layers.  The 1x1 appearance projection contributes
    no spatial radius.
    """
    radius_full, stride = 1, 1  # RGB encoder stem.
    result = {}
    for name in ("h2", "h4", "h8", "h16"):
        radius_full += stride  # stride-2 downsample 3x3
        stride *= 2
        radius_full += 2 * stride  # two residual 3x3 layers
        encoder_token_radius = int(math.ceil(radius_full / stride))
        result[name] = (
            encoder_token_radius + int(math.ceil(deform_max_offset)) + 1 + 2 + int(extra_margin)
        )
    return result


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )

    def forward(self, value):
        return F.silu(value + self.layers(value))


class PyramidEncoder(nn.Module):
    def __init__(self, base_channels):
        super().__init__()
        widths = (base_channels, base_channels * 2, base_channels * 3, base_channels * 4)
        self.widths = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(base_channels), base_channels),
            nn.SiLU(inplace=True),
        )
        self.down = nn.ModuleList()
        incoming = base_channels
        for width in widths:
            self.down.append(nn.Sequential(
                nn.Conv2d(incoming, width, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(_groups(width), width),
                nn.SiLU(inplace=True),
                ResidualBlock(width),
            ))
            incoming = width

    def forward(self, value):
        value = self.stem(value)
        result = {}
        for name, layer in zip(("h2", "h4", "h8", "h16"), self.down):
            value = layer(value)
            result[name] = value
        return result


class LocalDeformableAppearanceSampler(nn.Module):
    """Pure-PyTorch local sampler; all sampled values originate in RGB appearance."""
    def __init__(self, appearance_channels, structure_channels, samples, max_offset):
        super().__init__()
        self.samples = samples
        self.max_offset = float(max_offset)
        self.offset_head = nn.Conv2d(structure_channels, 2 * samples, 3, padding=1)
        self.weight_head = nn.Conv2d(structure_channels, samples, 3, padding=1)
        self.value_project = nn.Conv2d(appearance_channels, appearance_channels, 1)

    def forward(self, structure, appearance):
        batch, _, height, width = appearance.shape
        offsets = torch.tanh(self.offset_head(structure)) * self.max_offset
        weights = torch.softmax(self.weight_head(structure), dim=1)
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=appearance.device, dtype=appearance.dtype),
            torch.linspace(-1.0, 1.0, width, device=appearance.device, dtype=appearance.dtype),
            indexing="ij",
        )
        base = torch.stack((x, y), dim=-1).unsqueeze(0)
        values = self.value_project(appearance)
        sampled = []
        for index in range(self.samples):
            dx = offsets[:, 2 * index] * (2.0 / max(width - 1, 1))
            dy = offsets[:, 2 * index + 1] * (2.0 / max(height - 1, 1))
            grid = (base + torch.stack((dx, dy), dim=-1)).clamp(-1.0, 1.0)
            sampled.append(F.grid_sample(values, grid, mode="bilinear", padding_mode="border", align_corners=True))
        output = sum(weights[:, index:index + 1] * value for index, value in enumerate(sampled))
        return output, offsets, weights


class LocalCrossAttention(nn.Module):
    """Windowed cross attention: structure Query, RGB appearance Key/Value."""
    def __init__(self, channels, window_size=3):
        super().__init__()
        self.window_size = int(window_size)
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, structure, appearance, validity):
        batch, channels, height, width = structure.shape
        radius = self.window_size // 2
        query = self.query(structure).unsqueeze(2)
        key = F.unfold(self.key(appearance), self.window_size, padding=radius)
        key = key.reshape(batch, channels, self.window_size ** 2, height, width)
        value = F.unfold(self.value(appearance), self.window_size, padding=radius)
        value = value.reshape(batch, channels, self.window_size ** 2, height, width)
        valid = F.unfold(validity, self.window_size, padding=radius)
        valid = valid.reshape(batch, self.window_size ** 2, height, width) > 0.5
        logits = (query * key).sum(dim=1) / (channels ** 0.5)
        attention = torch.softmax(logits.masked_fill(~valid, -1e4), dim=1)
        attended = (attention.unsqueeze(1) * value).sum(dim=2)
        return structure + self.output(attended) * validity


class MemoryRetriever(nn.Module):
    def __init__(self, channels, structure_channels, max_tokens, topk, attention_temperature,
                 reliability_epsilon, ratio_threshold, confidence_threshold, query_chunk_size=1024):
        super().__init__()
        self.max_tokens = int(max_tokens)
        self.topk = int(topk)
        self.attention_temperature = float(attention_temperature)
        self.reliability_epsilon = float(reliability_epsilon)
        self.ratio_threshold = float(ratio_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.query_chunk_size = int(query_chunk_size)
        if self.query_chunk_size < 1:
            raise ValueError("query_chunk_size must be >= 1")
        self.key = nn.Conv2d(structure_channels, channels, 1)
        self.query = nn.Conv2d(structure_channels, channels, 1)
        self.context_key = nn.Linear(structure_channels, channels, bias=False)
        self.context_query = nn.Linear(structure_channels, channels, bias=False)
        self.prior = nn.Sequential(
            nn.Conv2d(2 * structure_channels, channels, 1), nn.SiLU(inplace=True), nn.Conv2d(channels, channels, 1)
        )

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
        value_flat = value.flatten(2).transpose(1, 2)
        reliability_flat = reliability.flatten(1)
        retrieved = torch.zeros_like(value_flat)
        confidence = torch.zeros(batch, height * width, device=value.device, dtype=value.dtype)
        candidate_count = torch.zeros(batch, height * width, device=value.device, dtype=torch.long)
        for item in range(batch):
            valid_indices = torch.where(reliability_flat[item] > self.reliability_epsilon)[0]
            if valid_indices.numel() == 0:
                continue
            if valid_indices.numel() > self.max_tokens:
                positions = torch.linspace(0, valid_indices.numel() - 1, self.max_tokens,
                                           device=value.device).round().long()
                valid_indices = valid_indices[positions]
            item_key = key[item, valid_indices]
            item_value = value_flat[item, valid_indices]
            item_reliability = reliability_flat[item, valid_indices]
            top_count = min(self.topk, valid_indices.numel())
            reliability_bias = item_reliability.clamp_min(self.reliability_epsilon).log().unsqueeze(0)
            for query_start in range(0, query.shape[1], self.query_chunk_size):
                query_end = min(query_start + self.query_chunk_size, query.shape[1])
                query_chunk = query[item, query_start:query_end]
                scores = query_chunk @ item_key.t() / self.attention_temperature
                scores = scores + reliability_bias
                top_scores, top_indices = torch.topk(scores, k=top_count, dim=-1)
                attention = torch.softmax(top_scores, dim=-1)
                retrieved[item, query_start:query_end] = (attention.unsqueeze(-1) * item_value[top_indices]).sum(dim=1)
                candidate_count[item, query_start:query_end].fill_(top_count)
                if top_count == 1:
                    confidence[item, query_start:query_end].fill_(1.0)
                else:
                    entropy = -(attention * attention.clamp_min(1e-8).log()).sum(dim=-1)
                    confidence[item, query_start:query_end] = (
                        1.0 - entropy / torch.log(torch.tensor(float(top_count), device=value.device))
                    ).clamp(0, 1)
        retrieved = retrieved.transpose(1, 2).reshape(batch, channels, height, width)
        confidence = confidence.reshape(batch, 1, height, width)
        candidate_count = candidate_count.reshape(batch, 1, height, width)
        sample_fallback = (reliable_ratio < self.ratio_threshold).view(batch, 1, 1, 1)
        fallback = (sample_fallback | (confidence < self.confidence_threshold)).to(value.dtype)
        global_context_map = global_context[:, :, None, None].expand(-1, -1, height, width)
        prior_input = torch.cat((structure, global_context_map), dim=1)
        appearance = fallback * self.prior(prior_input) + (1.0 - fallback) * retrieved
        return appearance, confidence, fallback, reliable_mass, reliable_ratio, candidate_count


class FogRoutedRGBTIRDehazer(nn.Module):
    def __init__(self, base_channels=16, router_hidden_channels=8, deform_num_samples=4,
                 deform_max_offset=2.0, num_structure_renderers=2, memory_max_tokens=256,
                 memory_topk=8, memory_attention_temperature=0.07,
                 memory_reliability_epsilon=1e-6, memory_reliable_ratio_threshold=0.01,
                 memory_confidence_threshold=0.1, memory_exclusion_extra_margin=0,
                 boundary_width=1, memory_query_chunk_size=1024):
        super().__init__()
        if memory_max_tokens < 1 or not (2 <= memory_topk <= memory_max_tokens):
            raise ValueError("require 2 <= memory_topk <= memory_max_tokens")
        if deform_num_samples < 1 or deform_max_offset < 0 or num_structure_renderers < 1:
            raise ValueError("invalid deform or renderer configuration")
        self.base_channels = int(base_channels)
        self.boundary_width = int(boundary_width)
        self.deform_max_offset = float(deform_max_offset)
        self.memory_exclusion_extra_margin = int(memory_exclusion_extra_margin)
        self.memory_query_chunk_size = int(memory_query_chunk_size)
        if self.memory_exclusion_extra_margin < 0 or self.memory_query_chunk_size < 1:
            raise ValueError("memory exclusion margin must be >= 0 and query chunk size must be >= 1")
        self.hde = HDE()
        self.router = MonotonicFogRouter(router_hidden_channels)
        self.rgb_encoder = PyramidEncoder(base_channels)
        self.tir_encoder = PyramidEncoder(base_channels)
        self.scale_names = ("h2", "h4", "h8", "h16")
        rgb_widths = self.rgb_encoder.widths
        structure_widths = (32, 48, 64, 96)
        self.appearance = nn.ModuleDict()
        self.structure = nn.ModuleDict()
        self.sampler = nn.ModuleDict()
        self.selector = nn.ModuleDict()
        self.renderers = nn.ModuleDict()
        self.magnitude = nn.ModuleDict()
        self.fusion_residual = nn.ModuleDict()
        self.completion = nn.ModuleDict()
        self.boundary = nn.ModuleDict()
        self.merge = nn.ModuleDict()
        self.memory = nn.ModuleDict()
        for name, rgb_width, structure_width in zip(self.scale_names, rgb_widths, structure_widths):
            self.appearance[name] = nn.Conv2d(rgb_width, rgb_width, 1)
            self.structure[name] = nn.Conv2d(structure_width, rgb_width, 1)
            self.sampler[name] = LocalDeformableAppearanceSampler(rgb_width, rgb_width, deform_num_samples, deform_max_offset)
            self.selector[name] = nn.Conv2d(rgb_width * 2, num_structure_renderers, 1)
            self.renderers[name] = nn.ModuleList([nn.Conv2d(rgb_width * 2, rgb_width, 3, padding=1) for _ in range(num_structure_renderers)])
            self.magnitude[name] = nn.Conv2d(1, 1, 1)
            self.fusion_residual[name] = ResidualBlock(rgb_width)
            self.completion[name] = nn.Sequential(nn.Conv2d(rgb_width, rgb_width, 1), ResidualBlock(rgb_width))
            self.boundary[name] = nn.Sequential(nn.Conv2d(rgb_width * 2 + 1, rgb_width, 1), ResidualBlock(rgb_width))
            self.merge[name] = nn.Conv2d(rgb_width * 3, rgb_width, 1)
            self.memory[name] = MemoryRetriever(
                rgb_width, rgb_width, memory_max_tokens, memory_topk, memory_attention_temperature,
                memory_reliability_epsilon, memory_reliable_ratio_threshold, memory_confidence_threshold,
                memory_query_chunk_size,
            )
        h2_channels = rgb_widths[0]
        self.decoder_cross = nn.ModuleDict({
            name: LocalCrossAttention(width) for name, width in zip(self.scale_names, rgb_widths)
        })
        self.decoder_up = nn.ModuleDict({
            "h16_to_h8": nn.Conv2d(rgb_widths[3], rgb_widths[2], 1),
            "h8_to_h4": nn.Conv2d(rgb_widths[2], rgb_widths[1], 1),
            "h4_to_h2": nn.Conv2d(rgb_widths[1], rgb_widths[0], 1),
        })
        self.rgb_output_head = nn.Conv2d(h2_channels, 3, 3, padding=1)

    @staticmethod
    def _pad_inputs(hazy_rgb, tir):
        if hazy_rgb.shape != tir.shape or hazy_rgb.ndim != 4 or hazy_rgb.shape[1] != 3:
            raise ValueError("hazy_rgb and tir must have matching [B,3,H,W] shape")
        height, width = hazy_rgb.shape[-2:]
        pad_h, pad_w = (-height) % 16, (-width) % 16
        padding = (0, pad_w, 0, pad_h)
        validity = torch.ones(hazy_rgb.shape[0], 1, height, width, device=hazy_rgb.device, dtype=hazy_rgb.dtype)
        return F.pad(hazy_rgb, padding), F.pad(tir, padding), F.pad(validity, padding), (height, width)

    @staticmethod
    def _crop(value, original_size):
        return value[..., :original_size[0], :original_size[1]]

    def encode_context(self, hazy_rgb, tir, route_temperature=1.0):
        padded_rgb, padded_tir, validity, original_size = self._pad_inputs(hazy_rgb, tir)
        hde_output = self.hde(padded_rgb, padded_tir)
        route = self.router(hde_output["density_map"], route_temperature)
        return {
            "density_map": hde_output["density_map"],
            "tir_structure_pyramid": hde_output["tir_structure_pyramid"],
            "route_logits": route["route_logits"], "route_soft": route["route_soft"], "route_hard": route["route_hard"],
            "rgb_pyramid": self.rgb_encoder(padded_rgb), "tir_content_pyramid": self.tir_encoder(padded_tir),
            "validity_mask": validity, "original_size": original_size,
        }

    @staticmethod
    def _scale_mask(mask, size, conservative=False):
        if conservative:
            return F.adaptive_max_pool2d(mask, size).clamp(0, 1)
        return F.interpolate(mask, size=size, mode="nearest").clamp(0, 1)

    def _memory_exclusion_at_scale(self, exclusion_full, size, scale_index):
        """Conservatively remove RGB values whose receptive field reaches Omega.

        The h2 appearance value sees the encoder stem/downsample, the local
        deformable offset and the fusion residual projection.  Deeper scales
        have a larger encoder receptive field, so the token-grid radius grows
        monotonically with scale.  This is intentionally conservative: an
        empty memory falls back to the TIR-conditioned prior rather than leak
        local RGB into a completion counterfactual.
        """
        exclusion = self._scale_mask(exclusion_full, size, conservative=True)
        radius = appearance_receptive_field_radius_by_scale(
            self.deform_max_offset, self.memory_exclusion_extra_margin,
        )[self.scale_names[scale_index]]
        if radius > 0:
            exclusion = F.max_pool2d(exclusion, 2 * radius + 1, stride=1, padding=radius)
        return exclusion.clamp(0, 1)

    @staticmethod
    def _route_gradient(route):
        dx = F.pad((route[..., :, 1:] - route[..., :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((route[..., 1:, :] - route[..., :-1, :]).abs(), (0, 0, 0, 1))
        return dx + dy

    def _boundary(self, soft_route, hard_route, mode):
        if mode == "soft":
            return self._route_gradient(soft_route).clamp(0, 1)
        dilated = F.max_pool2d(hard_route, 2 * self.boundary_width + 1, stride=1, padding=self.boundary_width)
        eroded = -F.max_pool2d(-hard_route, 2 * self.boundary_width + 1, stride=1, padding=self.boundary_width)
        return (dilated - eroded).clamp(0, 1)

    def decode_with_route(self, context: Dict[str, torch.Tensor], route_mode="soft",
                          route_override_value: Optional[torch.Tensor] = None,
                          route_override_mask: Optional[torch.Tensor] = None,
                          memory_exclude_mask: Optional[torch.Tensor] = None,
                          return_debug=False, boundary_mode=None):
        if route_mode not in ("soft", "hard"):
            raise ValueError("route_mode must be 'soft' or 'hard'")
        if boundary_mode is None:
            boundary_mode = route_mode
        if boundary_mode not in ("soft", "hard"):
            raise ValueError("boundary_mode must be 'soft' or 'hard'")
        if (route_override_value is None) != (route_override_mask is None):
            raise ValueError("route_override_value and route_override_mask must be provided together")
        original_size = context["original_size"]
        active = context["route_soft"] if route_mode == "soft" else context["route_hard"]
        soft_route, hard_route = context["route_soft"], context["route_hard"]
        expected = (active.shape[0], 1, *original_size)
        for name, tensor in (("route_override_value", route_override_value), ("route_override_mask", route_override_mask),
                             ("memory_exclude_mask", memory_exclude_mask)):
            if tensor is not None and tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")
        if route_override_value is not None:
            pad_h = active.shape[-2] - original_size[0]
            pad_w = active.shape[-1] - original_size[1]
            override_value = F.pad(route_override_value.detach().clamp(0, 1), (0, pad_w, 0, pad_h))
            override_mask = F.pad(route_override_mask.detach().clamp(0, 1), (0, pad_w, 0, pad_h))
        else:
            override_value = override_mask = None
        if memory_exclude_mask is not None:
            exclude_full = F.pad(memory_exclude_mask.detach().clamp(0, 1),
                                 (0, active.shape[-1] - original_size[1], 0, active.shape[-2] - original_size[0]))
        else:
            exclude_full = torch.zeros_like(active)
        structures, appearances, validity_scales, debug_scales = {}, {}, {}, {}
        reporting = None
        for name in self.scale_names:
            rgb = context["rgb_pyramid"][name]
            tir = context["tir_content_pyramid"][name]
            size = rgb.shape[-2:]
            valid = self._scale_mask(context["validity_mask"], size)
            validity_scales[name] = valid
            route = F.interpolate(active, size=size, mode="nearest")
            soft = F.interpolate(soft_route, size=size, mode="nearest")
            hard = F.interpolate(hard_route, size=size, mode="nearest")
            if override_mask is not None:
                mask = self._scale_mask(override_mask, size, conservative=True)
                value = self._scale_mask(override_value, size)
                route = torch.where(mask > 0.5, value, route)
                soft_for_boundary = torch.where(mask > 0.5, value, soft)
                hard_for_boundary = torch.where(mask > 0.5, value, hard)
            else:
                mask = torch.zeros_like(route)
                soft_for_boundary, hard_for_boundary = soft, hard
            exclusion = self._memory_exclusion_at_scale(exclude_full, size, self.scale_names.index(name))
            A = self.appearance[name](rgb)
            S = self.structure[name](context["tir_structure_pyramid"][name])
            O, offsets, sample_weights = self.sampler[name](S, A)
            renderer_weights = torch.softmax(self.selector[name](torch.cat((A, O), dim=1)), dim=1)
            delta = sum(renderer_weights[:, index:index + 1] * renderer(torch.cat((S, O), dim=1))
                        for index, renderer in enumerate(self.renderers[name]))
            density = F.interpolate(context["density_map"], size=size, mode="bilinear", align_corners=False)
            magnitude = torch.sigmoid(self.magnitude[name](density))
            fusion_candidate = self.fusion_residual[name](A + magnitude * delta)
            completion_candidate = self.completion[name](tir)
            route_for_boundary_gradient = soft_for_boundary if boundary_mode == "soft" else hard_for_boundary
            boundary = self._boundary(soft_for_boundary, hard_for_boundary, boundary_mode)
            boundary_feature = boundary * self.boundary[name](torch.cat((
                fusion_candidate, completion_candidate, self._route_gradient(route_for_boundary_gradient),
            ), dim=1))
            structures[name] = self.merge[name](torch.cat(((1 - route) * fusion_candidate, route * completion_candidate, boundary_feature), dim=1))
            reliability = ((1.0 - route).detach() * valid * (1.0 - exclusion))
            retrieved, confidence, fallback, mass, ratio, count = self.memory[name](S, fusion_candidate, reliability, valid)
            appearances[name] = (1.0 - route) * fusion_candidate + route * retrieved
            debug_scales[name] = {
                "effective_override_mask": mask, "candidate_count": count, "reliability": reliability,
                "offsets": offsets, "sample_weights": sample_weights, "renderer_weights": renderer_weights,
                "confidence": confidence, "fallback": fallback, "reliable_mass": mass, "reliable_ratio": ratio,
            }
            if name == "h2":
                reporting = (confidence, fallback, mass, ratio, boundary)
        decoded = None
        previous_name = None
        for name in reversed(self.scale_names):
            structural_query = structures[name]
            if decoded is not None:
                projection = self.decoder_up[f"{previous_name}_to_{name}"]
                decoded = F.interpolate(projection(decoded), size=structural_query.shape[-2:],
                                        mode="bilinear", align_corners=False)
                structural_query = structural_query + decoded
            decoded = self.decoder_cross[name](structural_query, appearances[name], validity_scales[name])
            previous_name = name
        decoded = F.silu(decoded)
        pred_clear = torch.sigmoid(self.rgb_output_head(F.interpolate(decoded, size=active.shape[-2:], mode="bilinear", align_corners=False)))
        confidence, fallback, mass, ratio, boundary = reporting
        output = {
            "pred_clear": self._crop(pred_clear, original_size),
            "density_map": self._crop(context["density_map"], original_size),
            "route_logits": self._crop(context["route_logits"], original_size),
            "route_soft": self._crop(context["route_soft"], original_size),
            "route_hard": self._crop(context["route_hard"], original_size),
            "boundary_map": self._crop(F.interpolate(boundary, size=active.shape[-2:], mode="nearest"), original_size),
            "memory_confidence": self._crop(F.interpolate(confidence, size=active.shape[-2:], mode="bilinear", align_corners=False), original_size),
            "memory_reliable_mass": mass,
            "memory_reliable_ratio": ratio,
            "memory_fallback_mask": self._crop(F.interpolate(fallback, size=active.shape[-2:], mode="nearest"), original_size),
        }
        if return_debug:
            output["debug"] = {
                "structure_tokens": structures, "appearance_tokens": appearances,
                "fusion_offsets": {name: value["offsets"] for name, value in debug_scales.items()},
                "fusion_weights": {name: value["sample_weights"] for name, value in debug_scales.items()},
                "effective_reliable_mask": {name: value["reliability"] for name, value in debug_scales.items()},
                "validity_mask": context["validity_mask"],
                "effective_override_masks": {name: value["effective_override_mask"] for name, value in debug_scales.items()},
                "memory_attention_candidate_count": {name: value["candidate_count"] for name, value in debug_scales.items()},
                "memory_reliability_weights": {name: value["reliability"] for name, value in debug_scales.items()},
                "memory_stats_by_scale": {
                    name: {
                        "confidence": value["confidence"], "fallback_mask": value["fallback"],
                        "reliable_mass": value["reliable_mass"], "reliable_ratio": value["reliable_ratio"],
                    }
                    for name, value in debug_scales.items()
                },
            }
        return output

    def forward(self, hazy_rgb, tir, route_temperature=1.0, route_mode="hard", return_debug=False):
        context = self.encode_context(hazy_rgb, tir, route_temperature=route_temperature)
        return self.decode_with_route(context, route_mode=route_mode, return_debug=return_debug)
