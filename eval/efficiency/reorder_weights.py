"""Reorder attention projections for contiguous AoH head groups.

Offline one-time reordering of W_Q, W_K, W_V, W_O along the head dimension
so that Retrieval Heads come first as a contiguous slice, followed by
Streaming Heads. This turns scatter/gather into simple slicing at runtime.

Supports both MHA and GQA:
  - MHA: reorder per Q head
  - GQA: reorder per KV group (each group = GRP contiguous Q heads)
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch


@torch.no_grad()
def reorder_attention_weights(
    attn_module,
    head_labels: np.ndarray,
    num_q_heads: int,
    num_kv_heads: int,
):
    """Reorder Q/K/V/O projection weights so Retrieval heads come first.

    After reordering, heads [0..n_retrieval-1] are Retrieval and
    [n_retrieval..G-1] are Streaming. Both form contiguous slices.

    Args:
        attn_module: HF self_attn module with q_proj, k_proj, v_proj, o_proj.
        head_labels: (G,) binary array. 1 = Retrieval, 0 = Streaming.
        num_q_heads: Total Q heads (H).
        num_kv_heads: Total KV heads (G).
    """
    grp = num_q_heads // num_kv_heads
    head_dim = attn_module.q_proj.weight.shape[0] // num_q_heads

    # Build permutation: retrieval KV groups first, then streaming
    retrieval_kv = np.where(head_labels >= 0.5)[0]
    streaming_kv = np.where(head_labels < 0.5)[0]
    kv_perm = np.concatenate([retrieval_kv, streaming_kv])

    # Expand KV permutation to Q head permutation
    q_perm = np.concatenate([
        np.arange(g * grp, (g + 1) * grp) for g in kv_perm
    ])

    # Reorder Q projection (output dimension = H * head_dim)
    _reorder_linear_out(attn_module.q_proj, q_perm, head_dim)

    # Reorder K projection (output dimension = G * head_dim)
    _reorder_linear_out(attn_module.k_proj, kv_perm, head_dim)

    # Reorder V projection (output dimension = G * head_dim)
    _reorder_linear_out(attn_module.v_proj, kv_perm, head_dim)

    # Reorder O projection (input dimension = H * head_dim)
    _reorder_linear_in(attn_module.o_proj, q_perm, head_dim)


@torch.no_grad()
def _reorder_linear_out(linear, perm, head_dim):
    """Reorder output (row) dimension of a Linear by head permutation."""
    idx = _expand_perm(perm, head_dim)
    idx_t = torch.tensor(idx, device=linear.weight.device, dtype=torch.long)
    linear.weight.data = linear.weight.data[idx_t].contiguous()
    if linear.bias is not None:
        linear.bias.data = linear.bias.data[idx_t].contiguous()


@torch.no_grad()
def _reorder_linear_in(linear, perm, head_dim):
    """Reorder input (column) dimension of a Linear by head permutation."""
    idx = _expand_perm(perm, head_dim)
    idx_t = torch.tensor(idx, device=linear.weight.device, dtype=torch.long)
    linear.weight.data = linear.weight.data[:, idx_t].contiguous()


def _expand_perm(perm, head_dim):
    """Expand head-level permutation to element-level indices."""
    indices = []
    for h in perm:
        start = int(h) * head_dim
        indices.extend(range(start, start + head_dim))
    return indices


def compute_reordered_labels(head_labels: np.ndarray) -> np.ndarray:
    """Return head_labels after reordering (retrieval first, then streaming).

    After reordering, labels become [1,1,...,1, 0,0,...,0].
    """
    n_retrieval = int((head_labels > 0.5).sum())
    result = np.zeros_like(head_labels)
    result[:n_retrieval] = 1.0
    return result
