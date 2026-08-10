#!/usr/bin/env python3
"""Optimized AoH vs Full latency/KV-storage benchmark for Llama-3.1-8B-Instruct.

Requested semantics:
  - Llama-3.1-8B-Instruct, AoH sp=0.25/0.5/0.75
  - Offline head reorder: retrieval KV heads are contiguous first; streaming KV heads after
  - Retrieval heads keep full KV and attend full history
  - Streaming heads use SWA: sink + recent KV only
  - FlashAttention/mem-efficient SDPA are disabled for all measured attention calls.

This benchmark reports two quantities:
  1. measured latency (prefill total seconds, decode ms/token)
  2. analytical KV storage GB (not total torch allocator peak), because the paper claim is
     about retained KV storage: full keeps all KV heads for all tokens; AoH keeps full KV only
     for retrieval heads and sink+recent for streaming heads.

Implementation note: for long contexts we keep chunked prefill. Full attention uses the local
Transformers Llama eager/math path with cache. Optimized AoH patches Llama attention and builds
an AoHKVCache directly during prefill/decode, never materializing a full DynamicCache for stream
heads.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig
try:
    from transformers.cache_utils import Cache, DynamicLayer
except ImportError:
    from transformers.cache_utils import Cache
    DynamicLayer = None
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

_HERE = Path(__file__).resolve().parent
try:
    from .attention_utils import compress_stream, streaming_attention
    from .benchmark_io import benchmark_config, load_progress, save_progress
    from .reorder_weights import reorder_attention_weights
except ImportError:
    from attention_utils import compress_stream, streaming_attention
    from benchmark_io import benchmark_config, load_progress, save_progress
    from reorder_weights import reorder_attention_weights

MODEL_NAME = "Llama-3.1-8B-Instruct"
DEFAULT_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PREFILL_LENS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
DECODE_CONTEXT_LENS = [8192, 16384, 32768, 65536, 131072, 262144]
DEFAULT_SP = ["0.25", "0.5", "0.75"]


@contextmanager
def math_sdpa_only():
    """Disable fused SDPA backends without intercepting model exceptions."""
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        context = torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_mem_efficient=False,
            enable_math=True,
        )
    else:
        context = sdpa_kernel([SDPBackend.MATH])
    with context:
        yield


def parse_gpus(s: str) -> list[int]:
    return [int(x.replace("cuda:", "")) for x in s.split(",") if x.strip()]


def primary_device(s: str) -> str:
    return f"cuda:{parse_gpus(s)[0]}"


def max_memory(gpus: list[int], per_gpu_gb: int) -> dict[int, str]:
    return {i: f"{per_gpu_gb}GiB" for i in gpus}


def rand_input(seq_len: int, device: str, vocab_size: int) -> torch.Tensor:
    return torch.randint(1, min(vocab_size, 32000), (1, seq_len), dtype=torch.long, device=device)


def synchronize(device: str):
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # (B, G, S, D) -> (B, G*n_rep, S, D)
    if n_rep == 1:
        return x
    b, g, s, d = x.shape
    return x[:, :, None, :, :].expand(b, g, n_rep, s, d).reshape(b, g * n_rep, s, d)


def causal_mask(q_len: int, k_len: int, past_len: int, device: torch.device) -> torch.Tensor:
    # True means allowed. Shape broadcastable to (B,H,Q,K).
    q_pos = torch.arange(past_len, past_len + q_len, device=device)[:, None]
    k_pos = torch.arange(k_len, device=device)[None, :]
    return k_pos <= q_pos


def eager_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, past_len: int, causal: bool) -> torch.Tensor:
    # q,k,v: (B,H,Q/S,D). Manual eager attention avoids FlashAttention entirely.
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        mask = causal_mask(q.shape[-2], k.shape[-2], past_len, q.device)
        scores = scores.masked_fill(~mask[None, None, :, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


class AoHKVCache(Cache):
    def __init__(self, num_layers: int, n_ret: list[int], n_str: list[int], sink_size: int, recent_size: int):
        if DynamicLayer is None:
            super().__init__()
        else:
            super().__init__(layers=[DynamicLayer() for _ in range(num_layers)])
        self.num_layers = num_layers
        self.n_ret = n_ret
        self.n_str = n_str
        self.sink_size = sink_size
        self.recent_size = recent_size
        self.kv_seq_len = 0
        self.full_k: list[Optional[torch.Tensor]] = [None] * num_layers
        self.full_v: list[Optional[torch.Tensor]] = [None] * num_layers
        self.stream_k: list[Optional[torch.Tensor]] = [None] * num_layers
        self.stream_v: list[Optional[torch.Tensor]] = [None] * num_layers

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None, **kwargs):
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.kv_seq_len

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int = 0):
        q_len = int(cache_position.numel()) if cache_position is not None else 1
        return self.kv_seq_len + q_len, 0

    def _compress_stream(self, k: torch.Tensor, v: torch.Tensor):
        return compress_stream(k, v, self.sink_size, self.recent_size)

    def append_layer(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        nr = self.n_ret[layer_idx]
        if nr > 0:
            rk, rv = k[:, :nr], v[:, :nr]
            self.full_k[layer_idx] = rk.contiguous() if self.full_k[layer_idx] is None else torch.cat([self.full_k[layer_idx], rk], dim=2).contiguous()
            self.full_v[layer_idx] = rv.contiguous() if self.full_v[layer_idx] is None else torch.cat([self.full_v[layer_idx], rv], dim=2).contiguous()
        if self.n_str[layer_idx] > 0:
            sk, sv = k[:, nr:], v[:, nr:]
            if self.stream_k[layer_idx] is None:
                self.stream_k[layer_idx], self.stream_v[layer_idx] = self._compress_stream(sk, sv)
            else:
                sk = torch.cat([self.stream_k[layer_idx], sk], dim=2)
                sv = torch.cat([self.stream_v[layer_idx], sv], dim=2)
                self.stream_k[layer_idx], self.stream_v[layer_idx] = self._compress_stream(sk, sv)


def make_aoh_forward(layer_idx: int, n_ret: int, n_str: int, num_q: int, num_kv: int, head_dim: int):
    grp = num_q // num_kv
    n_ret_q = n_ret * grp

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, cache_position=None, **kwargs):
        if not isinstance(past_key_values, AoHKVCache):
            return self._orig_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, num_q, head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, num_kv, head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        kv_cache: AoHKVCache = past_key_values
        past_len = kv_cache.kv_seq_len
        outs = []
        if n_ret > 0:
            old_fk = kv_cache.full_k[layer_idx]
            old_fv = kv_cache.full_v[layer_idx]
            new_fk, new_fv = k[:, :n_ret], v[:, :n_ret]
            fk = new_fk if old_fk is None else torch.cat([old_fk, new_fk], dim=2)
            fv = new_fv if old_fv is None else torch.cat([old_fv, new_fv], dim=2)
            ret_q = q[:, :n_ret_q]
            ret_out = eager_attention(ret_q, repeat_kv(fk, grp), repeat_kv(fv, grp), past_len=past_len, causal=(q_len > 1))
            outs.append(ret_out)
        if n_str > 0:
            old_sk = kv_cache.stream_k[layer_idx]
            old_sv = kv_cache.stream_v[layer_idx]
            new_sk, new_sv = k[:, n_ret:], v[:, n_ret:]
            sk_att = new_sk if old_sk is None else torch.cat([old_sk, new_sk], dim=2)
            sv_att = new_sv if old_sv is None else torch.cat([old_sv, new_sv], dim=2)
            str_q = q[:, n_ret_q:]
            str_out = streaming_attention(
                str_q,
                repeat_kv(sk_att, grp),
                repeat_kv(sv_att, grp),
                past_len=past_len,
                cached_len=0 if old_sk is None else old_sk.shape[2],
                sink_size=kv_cache.sink_size,
                recent_size=kv_cache.recent_size,
            )
            outs.append(str_out)

        attn_out = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
        attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, num_q * head_dim)
        kv_cache.append_layer(layer_idx, k, v)
        if layer_idx == kv_cache.num_layers - 1:
            kv_cache.kv_seq_len += q_len
        return self.o_proj(attn_out), None

    return forward


def patch_aoh_model(model, labels: np.ndarray, sink_size: int, recent_size: int):
    if sink_size < 0 or recent_size < 0:
        raise ValueError("sink_size and recent_size must be non-negative")
    cfg = model.config
    layers = model.model.layers
    num_q = cfg.num_attention_heads
    num_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_q)
    if labels.shape != (len(layers), num_kv):
        raise ValueError(f"head labels shape {labels.shape} != {(len(layers), num_kv)}")
    n_ret, n_str = [], []
    for i, layer in enumerate(layers):
        row = labels[i]
        nr = int((row >= 0.5).sum())
        ns = num_kv - nr
        reorder_attention_weights(layer.self_attn, row, num_q, num_kv)
        layer.self_attn._orig_forward = layer.self_attn.forward
        layer.self_attn.forward = types.MethodType(make_aoh_forward(i, nr, ns, num_q, num_kv, head_dim), layer.self_attn)
        n_ret.append(nr); n_str.append(ns)
    return lambda: AoHKVCache(len(layers), n_ret, n_str, sink_size, recent_size), n_ret, n_str


def full_prefill(model, input_ids: torch.Tensor, chunk_size: int):
    pkv = None
    last_logits = None
    seq = input_ids.shape[1]
    with torch.no_grad(), math_sdpa_only():
        for start in range(0, seq, chunk_size):
            chunk = input_ids[:, start:start+chunk_size]
            pos = torch.arange(start, start + chunk.shape[1], device=input_ids.device).unsqueeze(0)
            out = model(input_ids=chunk, position_ids=pos, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            last_logits = out.logits
    return pkv, last_logits


def aoh_prefill(model, make_cache, input_ids: torch.Tensor, chunk_size: int):
    cache = make_cache()
    last_logits = None
    seq = input_ids.shape[1]
    with torch.no_grad(), math_sdpa_only():
        for start in range(0, seq, chunk_size):
            chunk = input_ids[:, start:start+chunk_size]
            pos = torch.arange(start, start + chunk.shape[1], device=input_ids.device).unsqueeze(0)
            out = model(input_ids=chunk, position_ids=pos, past_key_values=cache, use_cache=True)
            last_logits = out.logits
    return cache, last_logits


def time_call(fn, n_reps: int, device: str):
    vals = []
    for _ in range(n_reps):
        cleanup(); synchronize(device)
        t0 = time.perf_counter()
        obj = fn()
        synchronize(device)
        vals.append(time.perf_counter() - t0)
        del obj
        cleanup()
    vals.sort()
    return vals[len(vals)//2]


def bench_decode(model, prefill_fn, input_ids: torch.Tensor, chunk_size: int, decode_steps: int, device: str, n_reps: int):
    vals = []
    for _ in range(n_reps):
        cleanup()
        cache, last_logits = prefill_fn(input_ids, chunk_size)
        cur = last_logits[:, -1:, :].argmax(dim=-1)
        seq = input_ids.shape[1]
        synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad(), math_sdpa_only():
            for step in range(decode_steps):
                pos = torch.tensor([[seq + step]], dtype=torch.long, device=input_ids.device)
                out = model(input_ids=cur, position_ids=pos, past_key_values=cache, use_cache=True)
                cur = out.logits[:, -1:, :].argmax(dim=-1)
        synchronize(device)
        vals.append((time.perf_counter() - t0) / decode_steps * 1000.0)
        del cache, last_logits, cur, out
        cleanup()
    vals.sort()
    return vals[len(vals)//2]


def kv_storage_gb_full(cfg, total_len: int, dtype_bytes: int = 2) -> float:
    return cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * total_len * getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads) * dtype_bytes / 1e9


def kv_storage_gb_aoh(cfg, n_ret: list[int], n_str: list[int], total_len: int, sink_size: int, recent_size: int, dtype_bytes: int = 2) -> float:
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    stream_len = min(total_len, sink_size + recent_size)
    elems = 0
    for nr, ns in zip(n_ret, n_str):
        elems += 2 * (nr * total_len + ns * stream_len) * hd
    return elems * dtype_bytes / 1e9


def load_model(model_path: str, gpus: str, per_gpu_gb: int, eager: bool = True):
    # Transformers may pre-allocate a large temporary cache during from_pretrained()
    # based on max_memory.  On this shared machine nvidia-smi can expose stale CUDA
    # contexts; disabling the loading-time warmup avoids a spurious OOM before the
    # model is actually loaded, without changing the benchmarked sequence lengths.
    from transformers import modeling_utils as _modeling_utils
    old_warmup = getattr(_modeling_utils, "caching_allocator_warmup", None)
    if old_warmup is not None:
        _modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="eager" if eager else "sdpa",
            device_map="sequential",
            max_memory=max_memory(parse_gpus(gpus), per_gpu_gb),
        ).eval()
    finally:
        if old_warmup is not None:
            _modeling_utils.caching_allocator_warmup = old_warmup


def bench_full_baseline(args, cfg, vocab_size: int):
    out_dir = Path(args.out_dir) / MODEL_NAME / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "full_baseline.json"
    run_config = benchmark_config(args, "full")
    done = load_progress(out_json, run_config, args.force)
    model = load_model(args.model_path, args.gpus, args.per_gpu_gb, eager=True)
    device = primary_device(args.gpus)
    rows_prefill = done.get("prefill", [])
    rows_decode = done.get("decode", [])
    have_pf = {r["context_len"] for r in rows_prefill}
    have_dec = {r["context_len"] for r in rows_decode}

    for ctx in args.prefill_lens:
        if ctx in have_pf: continue
        inp = rand_input(ctx, device, vocab_size)
        cs = min(args.chunk_size, ctx)
        print(f"[full][prefill] ctx={ctx} chunk={cs}", flush=True)
        try:
            lat = time_call(lambda: full_prefill(model, inp, cs), args.n_reps, device)
            row = {"context_len": ctx, "latency_s": round(lat, 4), "kv_storage_gb": round(kv_storage_gb_full(cfg, ctx), 4), "oom": False}
        except torch.OutOfMemoryError:
            row = {"context_len": ctx, "latency_s": None, "kv_storage_gb": round(kv_storage_gb_full(cfg, ctx), 4), "oom": True}
        rows_prefill.append(row); save_progress(out_json, run_config, rows_prefill, rows_decode)
        del inp; cleanup()

    for ctx in args.decode_lens:
        if ctx in have_dec: continue
        inp = rand_input(ctx, device, vocab_size)
        cs = min(args.chunk_size, ctx)
        print(f"[full][decode] ctx={ctx} steps={args.n_decode_steps}", flush=True)
        try:
            lat = bench_decode(model, lambda ids, chunk: full_prefill(model, ids, chunk), inp, cs, args.n_decode_steps, device, args.n_decode_reps)
            row = {"context_len": ctx, "decode_steps": args.n_decode_steps, "latency_ms_per_tok": round(lat, 4), "kv_storage_gb": round(kv_storage_gb_full(cfg, ctx + args.n_decode_steps), 4), "oom": False}
        except torch.OutOfMemoryError:
            row = {"context_len": ctx, "decode_steps": args.n_decode_steps, "latency_ms_per_tok": None, "kv_storage_gb": round(kv_storage_gb_full(cfg, ctx + args.n_decode_steps), 4), "oom": True}
        rows_decode.append(row); save_progress(out_json, run_config, rows_prefill, rows_decode)
        del inp; cleanup()
    del model; cleanup()
    return {"prefill": rows_prefill, "decode": rows_decode}


def bench_aoh_sp(args, cfg, vocab_size: int, sp: str):
    labels_path = args.aoh_root / f"aoh_mean_sp{sp}" / "head_classification.npy"
    labels = np.load(labels_path)
    out_dir = Path(args.out_dir) / MODEL_NAME / f"aoh_sp{sp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "aoh_bench.json"
    run_config = benchmark_config(args, f"aoh_sp{sp}", labels_path)
    done = load_progress(out_json, run_config, args.force)
    model = load_model(args.model_path, args.gpus, args.per_gpu_gb, eager=True)
    make_cache, n_ret, n_str = patch_aoh_model(model, labels, args.sink_size, args.recent_size)
    device = primary_device(args.gpus)
    rows_prefill = done.get("prefill", [])
    rows_decode = done.get("decode", [])
    have_pf = {r["context_len"] for r in rows_prefill}
    have_dec = {r["context_len"] for r in rows_decode}
    meta = {"sp": sp, "labels_path": str(labels_path), "sink_size": args.sink_size, "recent_size": args.recent_size,
            "avg_retrieval_kv_heads": float(np.mean(n_ret)), "avg_stream_kv_heads": float(np.mean(n_str))}

    for ctx in args.prefill_lens:
        if ctx in have_pf: continue
        inp = rand_input(ctx, device, vocab_size)
        cs = min(args.chunk_size, ctx)
        print(f"[aoh sp={sp}][prefill] ctx={ctx} chunk={cs}", flush=True)
        try:
            lat = time_call(lambda: aoh_prefill(model, make_cache, inp, cs), args.n_reps, device)
            row = {"context_len": ctx, "latency_s": round(lat, 4), "kv_storage_gb": round(kv_storage_gb_aoh(cfg, n_ret, n_str, ctx, args.sink_size, args.recent_size), 4), "oom": False}
        except torch.OutOfMemoryError:
            row = {"context_len": ctx, "latency_s": None, "kv_storage_gb": round(kv_storage_gb_aoh(cfg, n_ret, n_str, ctx, args.sink_size, args.recent_size), 4), "oom": True}
        rows_prefill.append(row); save_progress(out_json, run_config, rows_prefill, rows_decode)
        del inp; cleanup()

    for ctx in args.decode_lens:
        if ctx in have_dec: continue
        inp = rand_input(ctx, device, vocab_size)
        cs = min(args.chunk_size, ctx)
        print(f"[aoh sp={sp}][decode] ctx={ctx} steps={args.n_decode_steps}", flush=True)
        try:
            lat = bench_decode(model, lambda ids, chunk: aoh_prefill(model, make_cache, ids, chunk), inp, cs, args.n_decode_steps, device, args.n_decode_reps)
            row = {"context_len": ctx, "decode_steps": args.n_decode_steps, "latency_ms_per_tok": round(lat, 4), "kv_storage_gb": round(kv_storage_gb_aoh(cfg, n_ret, n_str, ctx + args.n_decode_steps, args.sink_size, args.recent_size), 4), "oom": False}
        except torch.OutOfMemoryError:
            row = {"context_len": ctx, "decode_steps": args.n_decode_steps, "latency_ms_per_tok": None, "kv_storage_gb": round(kv_storage_gb_aoh(cfg, n_ret, n_str, ctx + args.n_decode_steps, args.sink_size, args.recent_size), 4), "oom": True}
        rows_decode.append(row); save_progress(out_json, run_config, rows_prefill, rows_decode)
        del inp; cleanup()
    del model; cleanup()
    return {"meta": meta, "prefill": rows_prefill, "decode": rows_decode}


def write_comparison(args):
    root = Path(args.out_dir) / MODEL_NAME
    full_path = root / "full/full_baseline.json"
    if not full_path.exists():
        return
    full = json.load(open(full_path))
    full_pf = {r["context_len"]: r for r in full.get("prefill", [])}
    full_dec = {r["context_len"]: r for r in full.get("decode", [])}
    rows_pf, rows_dec = [], []
    for sp in args.sps:
        p = root / f"aoh_sp{sp}/aoh_bench.json"
        if not p.exists(): continue
        a = json.load(open(p))
        for r in a.get("prefill", []):
            f = full_pf.get(r["context_len"])
            if not f: continue
            speed = None if (r.get("latency_s") is None or f.get("latency_s") is None) else f["latency_s"] / r["latency_s"]
            rows_pf.append({"sp": sp, "context_len": r["context_len"], "full_latency_s": f.get("latency_s"), "aoh_latency_s": r.get("latency_s"), "speedup": None if speed is None else round(speed, 3), "full_kv_gb": f.get("kv_storage_gb"), "aoh_kv_gb": r.get("kv_storage_gb"), "kv_reduction": None if not r.get("kv_storage_gb") else round(f.get("kv_storage_gb")/r.get("kv_storage_gb"), 3)})
        for r in a.get("decode", []):
            f = full_dec.get(r["context_len"])
            if not f: continue
            speed = None if (r.get("latency_ms_per_tok") is None or f.get("latency_ms_per_tok") is None) else f["latency_ms_per_tok"] / r["latency_ms_per_tok"]
            rows_dec.append({"sp": sp, "context_len": r["context_len"], "full_latency_ms_tok": f.get("latency_ms_per_tok"), "aoh_latency_ms_tok": r.get("latency_ms_per_tok"), "speedup": None if speed is None else round(speed, 3), "full_kv_gb": f.get("kv_storage_gb"), "aoh_kv_gb": r.get("kv_storage_gb"), "kv_reduction": None if not r.get("kv_storage_gb") else round(f.get("kv_storage_gb")/r.get("kv_storage_gb"), 3)})
    for name, rows in [("prefill_comparison.csv", rows_pf), ("decode_comparison.csv", rows_dec)]:
        if rows:
            with open(root / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump({"prefill": rows_pf, "decode": rows_dec}, open(root / "comparison.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Hugging Face model ID or local model directory")
    ap.add_argument(
        "--head-root",
        type=Path,
        help="Directory containing aoh_mean_sp{ratio}/head_classification.npy",
    )
    ap.add_argument("--mode", choices=["all", "full", "aoh", "compare"], default="all")
    ap.add_argument("--sps", default=",".join(DEFAULT_SP))
    ap.add_argument("--gpus", default="cuda:0,cuda:1,cuda:2,cuda:3")
    ap.add_argument("--per_gpu_gb", type=int, default=72)
    ap.add_argument("--chunk_size", type=int, default=256, help="Small default to keep eager/math attention memory bounded")
    ap.add_argument("--prefill_lens", default=",".join(map(str, PREFILL_LENS)))
    ap.add_argument("--decode_lens", default=",".join(map(str, DECODE_CONTEXT_LENS)))
    ap.add_argument("--n_reps", type=int, default=1)
    ap.add_argument("--n_decode_reps", type=int, default=1)
    ap.add_argument("--n_decode_steps", type=int, default=20)
    ap.add_argument("--sink_size", type=int, default=128)
    ap.add_argument("--recent_size", type=int, default=256)
    ap.add_argument("--out_dir", default=str(_HERE / "outputs"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.sps = [s.strip() for s in args.sps.split(",") if s.strip()]
    args.prefill_lens = [int(x) for x in args.prefill_lens.split(",") if x.strip()]
    args.decode_lens = [int(x) for x in args.decode_lens.split(",") if x.strip()]
    args.aoh_root = args.head_root
    if args.mode in ("all", "aoh") and args.aoh_root is None:
        ap.error("--head-root is required when running AoH measurements")
    os.makedirs(Path(args.out_dir) / MODEL_NAME, exist_ok=True)
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    vocab_size = getattr(cfg, "vocab_size", 128256)
    print(json.dumps({"model": MODEL_NAME, "mode": args.mode, "sps": args.sps, "prefill_lens": args.prefill_lens, "decode_lens": args.decode_lens, "chunk_size": args.chunk_size, "n_decode_steps": args.n_decode_steps, "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled()}, indent=2), flush=True)
    if args.mode in ("all", "full"):
        bench_full_baseline(args, cfg, vocab_size)
    if args.mode in ("all", "aoh"):
        for sp in args.sps:
            bench_aoh_sp(args, cfg, vocab_size, sp)
    write_comparison(args)
    print(f"[done] results under {Path(args.out_dir) / MODEL_NAME}")


if __name__ == "__main__":
    main()
