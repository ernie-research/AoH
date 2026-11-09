"""AoH Llama for vLLM: per layer, retrieval KV heads use full attention, streaming KV heads use sink + window.

The head map (head_classification.npy, shape (num_layers, num_kv_heads), >= 0.5 means retrieval) is read from
hf_config.aoh_head_map. At load time q/k/v/o weights are permuted so that the retrieval heads of every layer come
first; the fused qkv output is then split into a retrieval part and a streaming part, each with its own
vLLM Attention layer and KV cache group.

Layer names: retrieval attention "model.layers.{i}.self_attn.attn", streaming attention
"model.layers.{L + i}.self_attn.attn_stream" (vLLM needs exactly one distinct integer per attention layer name).
"""
from __future__ import annotations

from math import gcd
from typing import Iterable

import numpy as np
import torch
from torch import nn
from vllm.attention import Attention
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.models.llama import LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP
from vllm.model_executor.models.utils import extract_layer_index

from .backend import AoHStreamBackend


def load_head_map(config) -> np.ndarray:
    labels = np.load(config.aoh_head_map)
    assert labels.shape == (config.num_hidden_layers, config.num_key_value_heads), labels.shape
    return labels >= 0.5


def kv_order(is_ret_row: np.ndarray) -> list[int]:
    """KV head order with retrieval heads first (stable), as in eval/efficiency/reorder_attention_weights."""
    return [int(i) for i in np.flatnonzero(is_ret_row)] + [int(i) for i in np.flatnonzero(~is_ret_row)]


def block_sizes(base: int, n_ret: int, n_str: int) -> tuple[int, int]:
    """Block sizes for the retrieval and streaming groups with equal bytes per block (vLLM needs a uniform page size)."""
    g = gcd(n_ret, n_str)
    return base * n_str // g, base * n_ret // g


class AoHLlamaAttention(LlamaAttention):

    def __init__(self, config, hidden_size, num_heads, num_kv_heads, rope_theta=10000, rope_scaling=None,
                 max_position_embeddings=8192, quant_config=None, bias=False, bias_o_proj=False,
                 cache_config=None, prefix=""):
        nn.Module.__init__(self)
        assert get_tensor_model_parallel_world_size() == 1, "AoH vLLM model supports TP=1 only"
        layer_idx = extract_layer_index(prefix)
        is_ret = load_head_map(config)
        n_ret = int(is_ret[layer_idx].sum())
        n_str = num_kv_heads - n_ret
        assert n_ret > 0 and n_str > 0, f"layer {layer_idx}: {n_ret} retrieval / {n_str} streaming KV heads"
        counts = set(is_ret.sum(1).tolist())
        assert len(counts) == 1, f"retrieval head count must be equal across layers, got {sorted(counts)}"

        self.hidden_size = hidden_size
        self.total_num_heads = self.num_heads = num_heads
        self.total_num_kv_heads = self.num_kv_heads = num_kv_heads
        self.head_dim = getattr(config, "head_dim", None) or hidden_size // num_heads
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        grp = num_heads // num_kv_heads
        self.split_q = [n_ret * grp * self.head_dim, n_str * grp * self.head_dim]
        self.split_kv = [n_ret * self.head_dim, n_str * self.head_dim]

        self.qkv_proj = QKVParallelLinear(hidden_size=hidden_size, head_size=self.head_dim,
                                          total_num_heads=num_heads, total_num_kv_heads=num_kv_heads, bias=bias,
                                          quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(input_size=num_heads * self.head_dim, output_size=hidden_size,
                                        bias=bias_o_proj, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self._init_rotary_emb(config, rope_scaling=rope_scaling, quant_config=quant_config)

        bs_ret, bs_str = block_sizes(cache_config.block_size, n_ret, n_str)
        sink, recent = int(config.aoh_sink), int(config.aoh_recent)
        self.attn = Attention(n_ret * grp, self.head_dim, self.scaling, num_kv_heads=n_ret,
                              cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.attn")
        self.attn.aoh_block_size = bs_ret
        stream_prefix = prefix.replace(f".{layer_idx}.", f".{config.num_hidden_layers + layer_idx}.")
        self.attn_stream = Attention(n_str * grp, self.head_dim, self.scaling, num_kv_heads=n_str,
                                     cache_config=cache_config, quant_config=quant_config,
                                     per_layer_sliding_window=recent + 1, attn_backend=AoHStreamBackend,
                                     prefix=f"{stream_prefix}.attn_stream")
        self.attn_stream.aoh_sink = sink
        self.attn_stream.aoh_block_size = bs_str
        self.attn_stream.aoh_dense_tail = int(getattr(config, "aoh_dense_tail", 50))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        q_r, q_s = q.split(self.split_q, dim=-1)
        k_r, k_s = k.split(self.split_kv, dim=-1)
        v_r, v_s = v.split(self.split_kv, dim=-1)
        out = torch.cat([self.attn(q_r, k_r, v_r), self.attn_stream(q_s, k_s, v_s)], dim=-1)
        output, _ = self.o_proj(out)
        return output


class AoHLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config, cache_config=None, quant_config=None, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
        attention_bias = getattr(config, "attention_bias", False) or getattr(config, "bias", False)
        self.self_attn = AoHLlamaAttention(
            config=config, hidden_size=self.hidden_size, num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            rope_theta=getattr(config, "rope_theta", 10000), rope_scaling=rope_scaling,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192), quant_config=quant_config,
            bias=attention_bias, bias_o_proj=attention_bias, cache_config=cache_config, prefix=f"{prefix}.self_attn")
        self.mlp = LlamaMLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size,
                            hidden_act=config.hidden_act, quant_config=quant_config,
                            bias=getattr(config, "mlp_bias", False), prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class AoHLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=AoHLlamaDecoderLayer)
        cfg = self.config
        grp = cfg.num_attention_heads // cfg.num_key_value_heads
        self._kv_order, self._q_order = [], []
        for row in load_head_map(cfg):
            order = kv_order(row)
            self._kv_order.append(order)
            self._q_order.append([h * grp + g for h in order for g in range(grp)])

    def _reorder(self, name: str, w: torch.Tensor) -> torch.Tensor:
        """Permute HF q/k/v rows and o_proj columns so that each layer's retrieval heads come first."""
        if ".self_attn." not in name or not name.endswith(".weight"):
            return w
        layer = extract_layer_index(name)
        d = self.config.head_dim if getattr(self.config, "head_dim", None) else \
            self.config.hidden_size // self.config.num_attention_heads
        q_idx = torch.tensor(self._q_order[layer], device=w.device)
        if ".q_proj." in name:
            return w.view(-1, d, w.shape[-1])[q_idx].reshape(w.shape)
        if ".k_proj." in name or ".v_proj." in name:
            return w.view(-1, d, w.shape[-1])[torch.tensor(self._kv_order[layer], device=w.device)].reshape(w.shape)
        if ".o_proj." in name:
            return w.view(w.shape[0], -1, d)[:, q_idx].reshape(w.shape)
        return w

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return super().load_weights((n, self._reorder(n, w)) for n, w in weights)
