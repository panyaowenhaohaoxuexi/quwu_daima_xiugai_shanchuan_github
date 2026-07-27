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
from .appearance_memory import MemoryRetriever, TIRConditionedAppearancePrior
from .structure_appearance_transformer import StructureAppearanceTransformerStage
from utils.model_config_validation import require_positive_integer, validate_model_config_values


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


class FogRoutedRGBTIRDehazer(nn.Module):
    def __init__(self, base_channels=16, router_hidden_channels=8, deform_num_samples=4,
                 deform_max_offset=2.0, num_structure_renderers=2, memory_max_tokens=256,
                 memory_topk=8, memory_attention_temperature=0.07,
                 memory_reliability_epsilon=1e-6, memory_reliable_ratio_threshold=0.01,
                 memory_confidence_threshold=0.1, memory_exclusion_extra_margin=0,
                 boundary_width=1, memory_query_chunk_size=1024, decoder_num_heads=4,
                 decoder_depth=1, decoder_window_size=7, decoder_window_chunk_size=128,
                 decoder_mlp_ratio=4.0, decoder_attention_dropout=0.0,
                 decoder_projection_dropout=0.0, decoder_ffn_dropout=0.0):
        super().__init__()
        model_config = {name: value for name, value in locals().items() if name != "self"}
        validate_model_config_values(model_config)
        base_channels = require_positive_integer("base_channels", base_channels)
        router_hidden_channels = require_positive_integer("router_hidden_channels", router_hidden_channels)
        deform_num_samples = require_positive_integer("deform_num_samples", deform_num_samples)
        num_structure_renderers = require_positive_integer("num_structure_renderers", num_structure_renderers)
        memory_max_tokens = require_positive_integer("memory_max_tokens", memory_max_tokens)
        memory_topk = require_positive_integer("memory_topk", memory_topk)
        memory_query_chunk_size = require_positive_integer("memory_query_chunk_size", memory_query_chunk_size)
        boundary_width = require_positive_integer("boundary_width", boundary_width)
        decoder_num_heads = require_positive_integer("decoder_num_heads", decoder_num_heads)
        decoder_depth = require_positive_integer("decoder_depth", decoder_depth)
        decoder_window_size = require_positive_integer("decoder_window_size", decoder_window_size)
        decoder_window_chunk_size = require_positive_integer("decoder_window_chunk_size", decoder_window_chunk_size)
        self.base_channels = base_channels
        self.boundary_width = boundary_width
        self.deform_max_offset = float(deform_max_offset)
        self.memory_exclusion_extra_margin = int(memory_exclusion_extra_margin)
        self.memory_query_chunk_size = memory_query_chunk_size
        self.hde = HDE()
        self.router = MonotonicFogRouter(router_hidden_channels)
        self.rgb_encoder = PyramidEncoder(base_channels)
        self.tir_encoder = PyramidEncoder(base_channels)
        self.scale_names = ("h2", "h4", "h8", "h16")
        rgb_widths = self.rgb_encoder.widths
        expected_widths = (self.base_channels, self.base_channels * 2, self.base_channels * 3, self.base_channels * 4)
        if tuple(rgb_widths) != expected_widths:
            raise ValueError(f"unexpected decoder widths={tuple(rgb_widths)}, expected={expected_widths}")
        if any(width % int(decoder_num_heads) for width in rgb_widths):
            raise ValueError("all decoder scale widths must be divisible by decoder_num_heads; "
                             f"decoder_num_heads={decoder_num_heads}, widths={tuple(rgb_widths)}")
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
        self.decoder_stages = nn.ModuleDict({
            name: StructureAppearanceTransformerStage(
                width, decoder_num_heads, decoder_depth, decoder_window_size, decoder_window_chunk_size,
                mlp_ratio=decoder_mlp_ratio, attention_dropout=decoder_attention_dropout,
                projection_dropout=decoder_projection_dropout, ffn_dropout=decoder_ffn_dropout,
            ) for name, width in zip(self.scale_names, rgb_widths)
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
            memory_appearance, confidence, fallback, mass, ratio, count, gate = self.memory[name](S, fusion_candidate, reliability, valid)
            appearances[name] = ((1.0 - route) * fusion_candidate + route * memory_appearance) * valid
            structures[name] = structures[name] * valid
            debug_scales[name] = {
                "effective_override_mask": mask, "candidate_count": count, "reliability": reliability,
                "offsets": offsets, "sample_weights": sample_weights, "renderer_weights": renderer_weights,
                "confidence": confidence, "fallback": fallback, "reliable_mass": mass, "reliable_ratio": ratio,
                "retrieval_gate": gate,
            }
            if name == "h2":
                reporting = (confidence, fallback, mass, ratio, gate, boundary)
        decoded = None
        previous_name = None
        for name in reversed(self.scale_names):
            structural_query = structures[name]
            if decoded is not None:
                projection = self.decoder_up[f"{previous_name}_to_{name}"]
                decoded = F.interpolate(projection(decoded), size=structural_query.shape[-2:],
                                        mode="bilinear", align_corners=False)
                structural_query = structural_query + decoded
            decoded = self.decoder_stages[name](structural_query, appearances[name], validity_scales[name])
            previous_name = name
        decoded = F.silu(decoded)
        pred_clear = torch.sigmoid(self.rgb_output_head(F.interpolate(decoded, size=active.shape[-2:], mode="bilinear", align_corners=False)))
        confidence, fallback, mass, ratio, gate, boundary = reporting
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
            "memory_retrieval_gate": self._crop(F.interpolate(gate, size=active.shape[-2:], mode="bilinear", align_corners=False), original_size),
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
                        "retrieval_gate": value["retrieval_gate"],
                        "reliable_mass": value["reliable_mass"], "reliable_ratio": value["reliable_ratio"],
                    }
                    for name, value in debug_scales.items()
                },
            }
        return output

    def forward(self, hazy_rgb, tir, route_temperature=1.0, route_mode="hard", return_debug=False):
        context = self.encode_context(hazy_rgb, tir, route_temperature=route_temperature)
        return self.decode_with_route(context, route_mode=route_mode, return_debug=return_debug)
