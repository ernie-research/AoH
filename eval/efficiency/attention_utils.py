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


def aoh_flash_attention(cache, layer_idx: int, q, k, v, n_ret: int, grp: int) -> torch.Tensor:
    """AoH attention on FlashAttention-2 (flash_attn_func).

    q, k, v: (B, H, S, D) after RoPE, retrieval heads first (see reorder_weights.py). KV is kept in
    the flash-attn layout (B, S, H, D). Prefill (empty cache): one causal flash_attn_func over all
    heads. Otherwise retrieval heads attend their full KV and streaming heads their sink + recent KV
    plus the new tokens, one flash_attn_func call per head group. After every call the streaming KV
    is trimmed to sink + recent in place, so streaming heads never hold more than sink + recent.
    Returns (B, S, H * D).
    """
    from flash_attn import flash_attn_func

    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    bsz, q_len = q.shape[:2]
    full_k, full_v = k[:, :, :n_ret], v[:, :, :n_ret]
    stream_k, stream_v = k[:, :, n_ret:], v[:, :, n_ret:]
    n_str = stream_k.shape[2]
    prefill = cache.full_k[layer_idx] is None and cache.stream_k[layer_idx] is None
    if prefill:
        out = flash_attn_func(q, k, v, causal=True, dropout_p=0.0)
        # k / v are views of this layer's projections; copy so they are not kept alive
        full_k, full_v = full_k.contiguous(), full_v.contiguous()
        stream_k, stream_v = stream_k.contiguous(), stream_v.contiguous()
    else:
        parts = []
        if n_ret:
            full_k = torch.cat([cache.full_k[layer_idx], full_k], dim=1)
            full_v = torch.cat([cache.full_v[layer_idx], full_v], dim=1)
            parts.append(flash_attn_func(q[:, :, : n_ret * grp], full_k, full_v, causal=True, dropout_p=0.0))
        if n_str:
            stream_k = torch.cat([cache.stream_k[layer_idx], stream_k], dim=1)
            stream_v = torch.cat([cache.stream_v[layer_idx], stream_v], dim=1)
            parts.append(flash_attn_func(q[:, :, n_ret * grp :], stream_k, stream_v, causal=True, dropout_p=0.0))
        out = torch.cat(parts, dim=2) if len(parts) > 1 else parts[0]

    sink, recent = cache.sink_size, cache.recent_size
    if n_str and stream_k.shape[1] > sink + recent:
        stream_k[:, sink : sink + recent].copy_(stream_k[:, stream_k.shape[1] - recent :].clone())
        stream_v[:, sink : sink + recent].copy_(stream_v[:, stream_v.shape[1] - recent :].clone())
        stream_k, stream_v = stream_k[:, : sink + recent], stream_v[:, : sink + recent]
    if n_ret:
        cache.full_k[layer_idx], cache.full_v[layer_idx] = full_k, full_v
    if n_str:
        cache.stream_k[layer_idx], cache.stream_v[layer_idx] = stream_k, stream_v
    return out.reshape(bsz, q_len, -1)


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

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * query.shape[-1] ** -0.5
    scores = scores.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(probabilities, value)
