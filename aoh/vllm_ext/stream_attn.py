"""Sink + sliding-window attention over vLLM's paged KV cache, for AoH streaming heads.

Query token at position i attends key j iff  j <= i  and  (j < sink  or  j >= i - recent),
the same mask as eval/efficiency/attention_utils.streaming_attention (the eager reference).

One program handles TM consecutive query tokens of one request and one KV head; its TM * GRP query rows share
every key tile. Keys come from two disjoint ranges: the sink [0, sink) and the window
[max(sink, first_pos - recent), last_pos], so a program reads at most sink + TM + recent keys and the window
keys are loaded once per TM tokens instead of once per token.

With dense_len (steps that still contain dense-prefill tokens), rows at positions < dense_len[r] are skipped: the backend computes them with a
full causal FlashAttention pass first, and this kernel overwrites only the sink + window rows.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sink_window_attn_kernel(
    Q, K, V, O, QSL, SEQ_LENS, BT, DENSE_LEN,
    scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_ot, stride_oh,
    stride_bt,
    SINK: tl.constexpr, RECENT: tl.constexpr, BS: tl.constexpr,
    GRP: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr, HAS_DENSE: tl.constexpr,
):
    r = tl.program_id(0)
    m = tl.program_id(1)
    h = tl.program_id(2)
    TM: tl.constexpr = ROWS // GRP
    q_start = tl.load(QSL + r)
    q_end = tl.load(QSL + r + 1)
    seq_len = tl.load(SEQ_LENS + r)
    first_pos = seq_len - (q_end - (q_start + m * TM))
    last_pos = tl.minimum(first_pos + TM - 1, seq_len - 1)
    if HAS_DENSE:  # rows at positions < dense_len[r] are written by the dense (full causal) pass
        dl = tl.load(DENSE_LEN + r)
    else:
        dl = 0
    if (q_start + m * TM < q_end) & (last_pos >= dl):
        row = tl.arange(0, ROWS)
        t = q_start + m * TM + row // GRP
        qh = h * GRP + row % GRP
        rmask = t < q_end
        pos = seq_len - (q_end - t)  # absolute position of each row's query token
        d = tl.arange(0, D)
        q = tl.load(Q + t[:, None] * stride_qt + qh[:, None] * stride_qh + d[None, :], mask=rmask[:, None],
                    other=0.0)

        m_i = tl.full([ROWS], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([ROWS], dtype=tl.float32)
        acc = tl.zeros([ROWS, D], dtype=tl.float32)
        bt_row = BT + r * stride_bt
        win_start = tl.maximum(first_pos - RECENT, SINK)
        n_sink_tiles: tl.constexpr = (SINK + BLOCK_N - 1) // BLOCK_N
        n_tiles: tl.constexpr = n_sink_tiles + (TM + RECENT + BLOCK_N - 1) // BLOCK_N
        offs = tl.arange(0, BLOCK_N)
        for it in tl.static_range(n_tiles):
            if it < n_sink_tiles:
                kp = it * BLOCK_N + offs
                kvalid = (kp < SINK) & (kp <= last_pos)
                allowed = kp[None, :] <= pos[:, None]
            else:
                kp = win_start + (it - n_sink_tiles) * BLOCK_N + offs
                kvalid = kp <= last_pos
                allowed = (kp[None, :] <= pos[:, None]) & (kp[None, :] >= pos[:, None] - RECENT)
            kp = tl.where(kvalid, kp, 0)
            blk = tl.load(bt_row + kp // BS, mask=kvalid, other=0).to(tl.int64)
            slot = kp % BS
            k = tl.load(K + blk[:, None] * stride_kb + slot[:, None] * stride_ks + h * stride_kh + d[None, :],
                        mask=kvalid[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(allowed & kvalid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + blk[:, None] * stride_vb + slot[:, None] * stride_vs + h * stride_vh + d[None, :],
                        mask=kvalid[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new

        out = acc / tl.maximum(l_i, 1e-30)[:, None]
        tl.store(O + t[:, None] * stride_ot + qh[:, None] * stride_oh + d[None, :], out.to(O.dtype.element_ty),
                 mask=(rmask & (pos >= dl))[:, None])


def sink_window_attention(query, key_cache, value_cache, out, query_start_loc, seq_lens, block_table,
                          scale: float, sink: int, recent: int, max_query_len: int, dense_len=None,
                          block_n: int = 64):
    """query: (T, Hq, D); key/value_cache: (num_blocks, block_size, Hkv, D); out: (T, Hq, D).
    query_start_loc: (R + 1,) int32; seq_lens: (R,) int32; block_table: (R, max_blocks) int32;
    max_query_len: largest number of new tokens of one request in this batch;
    dense_len: optional (>= R,) int32, rows at positions < dense_len[r] are left untouched in `out`."""
    hq, d = query.shape[1], query.shape[2]
    hkv = key_cache.shape[2]
    grp = hq // hkv
    assert grp & (grp - 1) == 0 and grp <= 16, grp
    num_reqs = seq_lens.shape[0]
    if num_reqs == 0 or max_query_len == 0:
        return out
    rows = 16 if max_query_len * grp <= 16 else 128  # decode: few rows; prefill: 128 / grp tokens per program
    tm = rows // grp
    _sink_window_attn_kernel[(num_reqs, triton.cdiv(max_query_len, tm), hkv)](
        query, key_cache, value_cache, out, query_start_loc, seq_lens, block_table,
        seq_lens if dense_len is None else dense_len,
        scale,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        out.stride(0), out.stride(1),
        block_table.stride(0),
        SINK=sink, RECENT=recent, BS=key_cache.shape[1],
        GRP=grp, ROWS=rows, D=d, BLOCK_N=block_n, HAS_DENSE=dense_len is not None,
        num_warps=4 if rows == 16 else 8,
    )
    return out
