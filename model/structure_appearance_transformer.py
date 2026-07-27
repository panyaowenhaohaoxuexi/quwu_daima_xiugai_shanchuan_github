"""Windowed structure--appearance cross-attention decoder primitives."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def _validate_dropout(name, value):
    if not 0.0 <= float(value) < 1.0:
        raise ValueError(f"{name} must be in [0, 1), received {value}")


class StructureAppearanceTransformerBlock(nn.Module):
    """Pre-norm windowed cross-attention from structure Query to appearance Key/Value.

    All tensors use ``[B, C, H, W]``.  ``validity`` is ``[B, 1, H, W]`` and
    prevents padded tokens from participating as either keys or queries.
    """

    def __init__(self, channels, num_heads, window_size, window_chunk_size, mlp_ratio=4.0,
                 attention_dropout=0.0, projection_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        if int(channels) <= 0:
            raise ValueError(f"channels must be > 0, received {channels}")
        if int(num_heads) <= 0:
            raise ValueError(f"num_heads must be > 0, received {num_heads}")
        if int(channels) % int(num_heads):
            raise ValueError(f"channels must be divisible by num_heads, received channels={channels}, num_heads={num_heads}")
        for name, value in (("window_size", window_size), ("window_chunk_size", window_chunk_size), ("mlp_ratio", mlp_ratio)):
            if float(value) <= 0:
                raise ValueError(f"{name} must be > 0, received {value}")
        _validate_dropout("attention_dropout", attention_dropout)
        _validate_dropout("projection_dropout", projection_dropout)
        _validate_dropout("ffn_dropout", ffn_dropout)
        self.channels, self.num_heads = int(channels), int(num_heads)
        self.window_size, self.window_chunk_size = int(window_size), int(window_chunk_size)
        self.head_dim = self.channels // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.structure_norm = nn.LayerNorm(self.channels)
        self.appearance_norm = nn.LayerNorm(self.channels)
        self.q_proj = nn.Linear(self.channels, self.channels)
        self.k_proj = nn.Linear(self.channels, self.channels)
        self.v_proj = nn.Linear(self.channels, self.channels)
        self.output_proj = nn.Linear(self.channels, self.channels)
        self.attention_dropout = nn.Dropout(float(attention_dropout))
        self.projection_dropout = nn.Dropout(float(projection_dropout))
        self.ffn_norm = nn.LayerNorm(self.channels)
        hidden = max(1, int(round(self.channels * float(mlp_ratio))))
        self.ffn = nn.Sequential(nn.Linear(self.channels, hidden), nn.GELU(), nn.Dropout(float(ffn_dropout)),
                                 nn.Linear(hidden, self.channels), nn.Dropout(float(ffn_dropout)))
        table_size = (2 * self.window_size - 1) ** 2
        self.relative_position_bias = nn.Parameter(torch.zeros(self.num_heads, table_size))
        coords = torch.stack(torch.meshgrid(torch.arange(self.window_size), torch.arange(self.window_size), indexing="ij"))
        relative = coords.flatten(1)[:, :, None] - coords.flatten(1)[:, None, :]
        relative[0] += self.window_size - 1
        relative[1] += self.window_size - 1
        self.register_buffer("relative_position_index", relative[0] * (2 * self.window_size - 1) + relative[1], persistent=False)

    def _partition(self, tokens, height, width):
        batch, _, channels = tokens.shape
        windows_h, windows_w = height // self.window_size, width // self.window_size
        return tokens.reshape(batch, windows_h, self.window_size, windows_w, self.window_size, channels).permute(
            0, 1, 3, 2, 4, 5
        ).reshape(batch * windows_h * windows_w, self.window_size ** 2, channels)

    def _restore(self, tokens, batch, height, width):
        windows_h, windows_w = height // self.window_size, width // self.window_size
        return tokens.reshape(batch, windows_h, windows_w, self.window_size, self.window_size, self.channels).permute(
            0, 1, 3, 2, 4, 5
        ).reshape(batch, height, width, self.channels)

    def forward(self, structure, appearance, validity):
        if structure.shape != appearance.shape or structure.ndim != 4:
            raise ValueError("structure and appearance must have matching [B,C,H,W] shape")
        if structure.shape[1] != self.channels:
            raise ValueError(f"structure channels must be {self.channels}, received {structure.shape[1]}")
        if tuple(validity.shape) != (structure.shape[0], 1, *structure.shape[-2:]):
            raise ValueError("validity must have shape [B,1,H,W]")
        batch, _, height, width = structure.shape
        valid = validity.to(dtype=structure.dtype).clamp(0, 1)
        pad_h, pad_w = (-height) % self.window_size, (-width) % self.window_size
        structure = F.pad(structure, (0, pad_w, 0, pad_h)) * F.pad(valid, (0, pad_w, 0, pad_h))
        appearance = F.pad(appearance, (0, pad_w, 0, pad_h)) * F.pad(valid, (0, pad_w, 0, pad_h))
        valid = F.pad(valid, (0, pad_w, 0, pad_h))
        padded_h, padded_w = structure.shape[-2:]
        structure_tokens = structure.permute(0, 2, 3, 1).reshape(batch, -1, self.channels)
        appearance_tokens = appearance.permute(0, 2, 3, 1).reshape(batch, -1, self.channels)
        valid_tokens = valid.permute(0, 2, 3, 1).reshape(batch, -1, 1)
        structure_windows = self._partition(structure_tokens, padded_h, padded_w)
        appearance_windows = self._partition(appearance_tokens, padded_h, padded_w)
        valid_windows = self._partition(valid_tokens, padded_h, padded_w).squeeze(-1) > 0
        query_windows = self.structure_norm(structure_windows)
        appearance_windows = self.appearance_norm(appearance_windows)
        outputs = []
        relative_bias = self.relative_position_bias[:, self.relative_position_index].unsqueeze(0)
        for start in range(0, query_windows.shape[0], self.window_chunk_size):
            stop = min(start + self.window_chunk_size, query_windows.shape[0])
            query_valid = valid_windows[start:stop].unsqueeze(-1).to(query_windows.dtype)
            key_valid = valid_windows[start:stop].unsqueeze(1).unsqueeze(1)
            q = self.q_proj(query_windows[start:stop]).reshape(-1, self.window_size ** 2, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(appearance_windows[start:stop]).reshape(-1, self.window_size ** 2, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(appearance_windows[start:stop]).reshape(-1, self.window_size ** 2, self.num_heads, self.head_dim).transpose(1, 2)
            logits = q @ k.transpose(-2, -1) * self.scale + relative_bias
            attention = torch.softmax(logits.masked_fill(~key_valid, -1e4), dim=-1)
            attention = attention * key_valid.to(attention.dtype)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attention = self.attention_dropout(attention)
            attended = (attention @ v).transpose(1, 2).reshape(-1, self.window_size ** 2, self.channels)
            attended = self.projection_dropout(self.output_proj(attended)) * query_valid
            value = (structure_windows[start:stop] + attended) * query_valid
            value = (value + self.ffn(self.ffn_norm(value))) * query_valid
            outputs.append(value)
        output = self._restore(torch.cat(outputs, dim=0), batch, padded_h, padded_w).permute(0, 3, 1, 2)
        return output[..., :height, :width] * validity.to(dtype=output.dtype)


class StructureAppearanceTransformerStage(nn.Module):
    """A stack of same-scale structure--appearance cross-attention blocks."""

    def __init__(self, channels, num_heads, depth, window_size, window_chunk_size, **kwargs):
        super().__init__()
        if int(depth) <= 0:
            raise ValueError(f"decoder_depth must be > 0, received {depth}")
        self.blocks = nn.ModuleList([
            StructureAppearanceTransformerBlock(channels, num_heads, window_size, window_chunk_size, **kwargs)
            for _ in range(int(depth))
        ])

    def forward(self, structure_tokens, appearance_tokens, validity_mask):
        value = structure_tokens * validity_mask.to(dtype=structure_tokens.dtype)
        for block in self.blocks:
            value = block(value, appearance_tokens, validity_mask)
        return value * validity_mask.to(dtype=value.dtype)
