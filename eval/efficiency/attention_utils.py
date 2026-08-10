"""Shared attention helpers for the AoH evaluation runtime."""
from __future__ import annotations

import torch


def _cached_positions(
    past_len: int,
    cached_len: int,
    sink_size: int,
    device: torch.device,
) -> torch.Tensor:
    if cached_len == past_len:
        return torch.arange(past_len, device=device)
    sink_len = min(sink_size, cached_len)
    recent_len = cached_len - sink_len
    parts = [torch.arange(sink_len, device=device)]
    if recent_len:
        parts.append(torch.arange(past_len - recent_len, past_len, device=device))
    return torch.cat(parts)


def compress_stream(
    key: torch.Tensor,
    value: torch.Tensor,
    sink_size: int,
    recent_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    budget = sink_size + recent_size
    if key.shape[2] <= budget:
        return key.contiguous(), value.contiguous()
    key_parts = []
    value_parts = []
    if sink_size:
        key_parts.append(key[:, :, :sink_size])
        value_parts.append(value[:, :, :sink_size])
    if recent_size:
        key_parts.append(key[:, :, -recent_size:])
        value_parts.append(value[:, :, -recent_size:])
    if not key_parts:
        return key[:, :, :0].contiguous(), value[:, :, :0].contiguous()
    return torch.cat(key_parts, dim=2).contiguous(), torch.cat(value_parts, dim=2).contiguous()


def streaming_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    past_len: int,
    cached_len: int,
    sink_size: int,
    recent_size: int,
) -> torch.Tensor:
    """Apply causal sink-plus-recent attention to a chunk of query tokens."""
    query_len = query.shape[-2]
    device = query.device
    old_positions = _cached_positions(past_len, cached_len, sink_size, device)
    new_positions = torch.arange(past_len, past_len + query_len, device=device)
    key_positions = torch.cat([old_positions, new_positions])

    causal = key_positions.unsqueeze(0) <= new_positions.unsqueeze(1)
    sink = key_positions.unsqueeze(0) < sink_size
    recent = key_positions.unsqueeze(0) >= (
        new_positions.unsqueeze(1) - recent_size
    )
    allowed = causal & (sink | recent)

    scores = torch.matmul(query, key.transpose(-2, -1)) * query.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probabilities, value)
