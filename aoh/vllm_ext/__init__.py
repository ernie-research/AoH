"""AoH (retrieval + streaming heads) plugin for vLLM V1 (tested with vLLM 0.10.1).

Registered through the `vllm.general_plugins` entry point, so it runs in the API process, the engine core and every
worker. Use it with

    LLM(model_path, hf_overrides={"architectures": ["AoHLlamaForCausalLM"],
                                  "aoh_head_map": ".../head_classification.npy",
                                  "aoh_sink": 128, "aoh_recent": 256,
                                  "aoh_dense_tail": 50},   # optional, default 50
        enable_prefix_caching=False)

Streaming heads follow the DuoAttention LongBench protocol: prompt tokens before num_prompt_tokens - aoh_dense_tail
use full causal attention; the last aoh_dense_tail prompt tokens and all generated tokens use sink + recent.
Streaming KV is kept in full until the dense part is computed, then dropped to sink + recent.
"""
from __future__ import annotations

import dataclasses

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    _registered = True

    import numpy as np
    import torch
    from vllm import ModelRegistry
    from vllm.v1.core import kv_cache_manager as kcm
    from vllm.v1.core import single_type_kv_cache_manager as stm
    from vllm.v1.worker import gpu_model_runner as gmr

    from . import state
    from .kv_spec import SinkSlidingWindowManager, SinkSlidingWindowSpec

    stm.spec_manager_map[SinkSlidingWindowSpec] = SinkSlidingWindowManager

    orig_alloc = kcm.KVCacheManager.allocate_slots

    def allocate_slots(self, request, *args, **kwargs):
        SinkSlidingWindowManager.prompt_lens[request.request_id] = request.num_prompt_tokens
        return orig_alloc(self, request, *args, **kwargs)

    kcm.KVCacheManager.allocate_slots = allocate_slots

    orig_spec = gmr.GPUModelRunner.get_kv_cache_spec

    def get_kv_cache_spec(self):
        from vllm.attention import Attention
        from vllm.config import get_layers_from_vllm_config
        specs = orig_spec(self)
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for name, spec in specs.items():
            attn = layers[name]
            bs = getattr(attn, "aoh_block_size", None)
            if bs is None:
                continue
            if hasattr(attn, "aoh_sink"):
                fields = {f.name: getattr(spec, f.name) for f in dataclasses.fields(spec)}
                specs[name] = SinkSlidingWindowSpec(**dict(fields, block_size=bs, sink=attn.aoh_sink,
                                                           dense_tail=attn.aoh_dense_tail))
            else:
                specs[name] = dataclasses.replace(spec, block_size=bs)
        return specs

    gmr.GPUModelRunner.get_kv_cache_spec = get_kv_cache_spec

    orig_prep = gmr.GPUModelRunner._prepare_inputs

    def _prepare_inputs(self, scheduler_output):
        out = orig_prep(self, scheduler_output)
        cfg = self.model_config.hf_config
        if getattr(cfg, "aoh_head_map", None) is None:  # not an AoH model
            return out
        # per-request dense length, in the (final) order of this step's batch
        b = self.input_batch
        n = b.num_reqs
        dense = np.maximum(b.num_prompt_tokens[:n] - int(getattr(cfg, "aoh_dense_tail", 50)), 0).astype(np.int32)
        state.any_dense = bool((b.num_computed_tokens_cpu[:n] < dense).any())
        if state.dense_len is None or state.dense_len.device != self.device:
            state.dense_len = torch.zeros(b.max_num_reqs, dtype=torch.int32, device=self.device)
        state.dense_len[:n].copy_(torch.from_numpy(dense))
        return out

    gmr.GPUModelRunner._prepare_inputs = _prepare_inputs
    ModelRegistry.register_model("AoHLlamaForCausalLM", "aoh.vllm_ext.model:AoHLlamaForCausalLM")
