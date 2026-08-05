"""Weight extraction utilities for HuggingFace transformer models.

Extracts per-head W_Q and per-KV-group W_K weight matrices from
self-attention layers.  Supports both MHA (H == G) and GQA (H > G).
"""

import torch
from transformers import AutoModelForCausalLM


def load_model(
    model_path: str,
    torch_dtype: torch.dtype | str = "auto",
    device: str = "cpu",
) -> tuple[torch.nn.Module, object]:
    """Load a HuggingFace causal LM and return (model, config).

    Args:
        model_path: HuggingFace model name or local path.
        torch_dtype: Checkpoint loading dtype. ``"auto"`` preserves its stored dtype.
        device: Device placement (for example, ``"cpu"`` or ``"cuda:0"``).

    Returns:
        (model, config) tuple.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    config = model.config
    # Some models (e.g. Qwen3.5) wrap config in text_config
    if hasattr(config, "text_config"):
        config = config.text_config
    return model, config


def get_model_info(config) -> dict:
    """Extract architecture info from model config.

    Returns dict with keys: num_layers, H, G, GRP, d_head, d_model, is_gqa.

    Supports models with explicit ``head_dim`` config attribute
    (e.g. Qwen3, Mistral-v0.3), falling back to ``hidden_size // num_heads``.
    """
    num_layers = config.num_hidden_layers
    H = config.num_attention_heads
    G = getattr(config, "num_key_value_heads", H)
    d_model = config.hidden_size
    # Qwen3 and some models set head_dim explicitly in config
    d_head = getattr(config, "head_dim", d_model // H)
    GRP = H // G
    return {
        "num_layers": num_layers,
        "H": H,
        "G": G,
        "GRP": GRP,
        "d_head": d_head,
        "d_model": d_model,
        "is_gqa": G < H,
    }


def _get_layer(model, layer_idx: int):
    """Get the self-attention module for a given layer index."""
    # Standard HF layout: model.model.layers[i].self_attn
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx].self_attn
    # Bare backbone (e.g. after model = full_model.model)
    if hasattr(model, "layers"):
        return model.layers[layer_idx].self_attn
    raise AttributeError(
        f"Cannot locate layers in model of type {type(model).__name__}. "
        "Expected model.model.layers or model.layers."
    )


@torch.no_grad()
def extract_qk_weights(
    model,
    config,
    layer_idx: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract per-head W_Q and per-KV-group W_K for a single layer.

    Args:
        model: HuggingFace model (full or bare backbone).
        config: Model config with num_attention_heads, num_key_value_heads, etc.
        layer_idx: Layer index.
        device: Target device for the returned tensors.

    Returns:
        W_Q: (H, d_head, d_model)  — all H query heads.
        W_K: (G, d_head, d_model)  — all G KV heads.
    """
    info = get_model_info(config)
    H, G, d_head, d_model = info["H"], info["G"], info["d_head"], info["d_model"]

    attn = _get_layer(model, layer_idx)

    # q_proj.weight: (H * d_head, d_model)
    W_Q = attn.q_proj.weight.data.to(device=device, dtype=torch.float32)
    W_Q = W_Q.reshape(H, d_head, d_model)

    # k_proj.weight: (G * d_head, d_model)
    W_K = attn.k_proj.weight.data.to(device=device, dtype=torch.float32)
    W_K = W_K.reshape(G, d_head, d_model)

    return W_Q, W_K
