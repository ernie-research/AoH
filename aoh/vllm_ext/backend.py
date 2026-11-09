"""Attention backend for AoH streaming heads.

Reuses FlashAttention's metadata builder (query_start_loc, seq_lens, block_table, slot_mapping) and KV layout, and
replaces the attention call with the sink + window Triton kernel in stream_attn.py.

Dense prefill (hf_config.aoh_dense_tail, default 50): in steps that contain tokens at positions < dense_len[r]
(state.any_dense), a full causal FlashAttention pass writes every row first, then the Triton kernel overwrites the
rows at positions >= dense_len[r] with sink + window attention. The KV manager keeps the full KV of a request until
its dense part is computed, so the full pass sees every key.
"""
from __future__ import annotations

from vllm.attention.utils.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend, FlashAttentionImpl

from . import state
from .stream_attn import sink_window_attention


class AoHStreamImpl(FlashAttentionImpl):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.sliding_window[0] >= 0, "AoH streaming layers need a sliding window"
        self.recent = self.sliding_window[0]  # FlashAttentionImpl stores (sliding_window - 1, 0)
        assert self.kv_cache_dtype == "auto", "AoH streaming layers support only the model dtype for the KV cache"

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None, output_scale=None):
        assert output is not None and output_scale is None
        if attn_metadata is None:  # profiling run
            return output
        key_cache, value_cache = kv_cache.unbind(0)
        reshape_and_cache_flash(key, value, key_cache, value_cache, attn_metadata.slot_mapping, self.kv_cache_dtype,
                                layer._k_scale, layer._v_scale)
        m = attn_metadata
        dense_len = None
        if state.any_dense:
            n = m.num_actual_tokens
            flash_attn_varlen_func(q=query[:n], k=key_cache, v=value_cache, out=output[:n],
                                   cu_seqlens_q=m.query_start_loc, max_seqlen_q=m.max_query_len,
                                   seqused_k=m.seq_lens, max_seqlen_k=m.max_seq_len, softmax_scale=self.scale,
                                   causal=True, window_size=(-1, -1), block_table=m.block_table,
                                   fa_version=self.vllm_flash_attn_version)
            dense_len = state.dense_len
        sink_window_attention(query, key_cache, value_cache, output, m.query_start_loc, m.seq_lens, m.block_table,
                              self.scale, layer.aoh_sink, self.recent, m.max_query_len, dense_len)
        return output


class AoHStreamBackend(FlashAttentionBackend):

    @staticmethod
    def get_impl_cls():
        return AoHStreamImpl
