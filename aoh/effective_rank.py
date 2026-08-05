"""Effective rank computation via d_head-space Gram matrices.

Core formulas:
    M_h = W_K^{gT} @ W_Q^h          (d_model x d_model, rank <= d_head)
    C   = (W_Q @ W_Q^T) @ (W_K @ W_K^T)   (d_head x d_head)
    eigenvalues(C) = singular_values(M_h)^2

    sigma_k     = sqrt(|eig_k|)
    sigma_hat_k = sigma_k / sum(sigma)
    H           = -sum(sigma_hat_k * log(sigma_hat_k))
    eff_rank    = exp(H)

Binary approximation:
    C^B = sign(W_Q)@sign(W_Q)^T @ sign(W_K)@sign(W_K)^T
    spectrum(M_h^B) ≡ spectrum(M_h)  (under Gaussian assumption)
"""

import numpy as np
import torch
from tqdm import tqdm

from .weight_extract import extract_qk_weights, get_model_info

EPS = 1e-12


def compute_gram_matrix(
    W_Q_h: torch.Tensor,
    W_K_g: torch.Tensor,
) -> torch.Tensor:
    """Compute C = (W_Q @ W_Q^T) @ (W_K @ W_K^T) in d_head space.

    Args:
        W_Q_h: (d_head, d_model) — single Q head weight.
        W_K_g: (d_head, d_model) — single KV head weight.

    Returns:
        C: (d_head, d_head).
    """
    QQ = W_Q_h @ W_Q_h.T
    KK = W_K_g @ W_K_g.T
    return QQ @ KK


def compute_gram_matrix_binary(
    W_Q_h: torch.Tensor,
    W_K_g: torch.Tensor,
) -> torch.Tensor:
    """Binary Gram matrix: C^B = sign(W_Q)@sign(W_Q)^T @ sign(W_K)@sign(W_K)^T.

    Uses torch.sign() approximation.  True XNOR+popcount is a future optimisation.

    Args / Returns: same as compute_gram_matrix.
    """
    sQ = torch.sign(W_Q_h)
    sK = torch.sign(W_K_g)
    QQ_b = sQ @ sQ.T
    KK_b = sK @ sK.T
    return QQ_b @ KK_b


def eigenvalues_to_eff_rank(eigenvalues: torch.Tensor) -> float:
    """Convert eigenvalues of C to effective rank.

    Steps:
        1. sigma_k = sqrt(|eigenvalues|)
        2. Normalize to probability distribution
        3. Shannon entropy  H = -sum(p_k * log(p_k))
        4. eff_rank = exp(H)

    Args:
        eigenvalues: 1-D tensor (possibly complex from non-symmetric C).

    Returns:
        Effective rank (float scalar).
    """
    sigma = torch.sqrt(eigenvalues.abs().real + EPS)
    sigma_sum = sigma.sum()
    if sigma_sum < EPS:
        return 1.0
    p = sigma / sigma_sum
    # Shannon entropy (filter zeros for numerical safety)
    log_p = torch.log(p + EPS)
    H = -(p * log_p).sum()
    return float(torch.exp(H).item())


def compute_eff_rank(
    W_Q_h: torch.Tensor,
    W_K_g: torch.Tensor,
    binary: bool = False,
) -> float:
    """End-to-end effective rank for a single (Q head, KV head) pair.

    Args:
        W_Q_h: (d_head, d_model).
        W_K_g: (d_head, d_model).
        binary: Use sign() approximation.

    Returns:
        Effective rank (float).
    """
    gram_fn = compute_gram_matrix_binary if binary else compute_gram_matrix
    C = gram_fn(W_Q_h, W_K_g)
    eigvals = torch.linalg.eigvals(C)
    return eigenvalues_to_eff_rank(eigvals)


@torch.no_grad()
def compute_all_eff_ranks(
    model,
    config,
    binary: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Compute effective ranks for all Q heads across all layers.

    For each layer l, for each Q head h:
        W_K used = W_K[g(h)]  where g(h) = h // GRP

    Args:
        model:  HuggingFace model.
        config: Model config.
        binary: Use binary approximation.
        device: Compute device.

    Returns:
        eff_ranks: np.ndarray of shape (num_layers, H).
    """
    info = get_model_info(config)
    num_layers, H, GRP = info["num_layers"], info["H"], info["GRP"]

    eff_ranks = np.zeros((num_layers, H), dtype=np.float64)

    for l in tqdm(range(num_layers), desc="Computing effective ranks"):
        W_Q, W_K = extract_qk_weights(model, config, l, device=device)
        for h in range(H):
            g = h // GRP
            eff_ranks[l, h] = compute_eff_rank(W_Q[h], W_K[g], binary=binary)
        del W_Q, W_K

    return eff_ranks
