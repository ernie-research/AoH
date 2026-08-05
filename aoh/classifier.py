"""AoHClassifier — training-free attention head classifier.

Classifies each KV head (GQA) or Q head (MHA) as Retrieval or Streaming
based on the effective rank of M_h = W_K^T @ W_Q.

    Low  eff_rank → Retrieval Head (specialised, needs Full Attention)
    High eff_rank → Streaming Head (diffuse, SWA suffices)
"""

import numpy as np
import torch

from .weight_extract import load_model, get_model_info
from .effective_rank import compute_all_eff_ranks
from .io import save_results, load_results


class AoHClassifier:
    """Training-free attention head classifier using effective rank.

    Supports both MHA (H == G) and GQA (H > G).  For GQA, per-Q-head
    effective ranks are aggregated to per-KV-group via ``agg`` in {mean, min, max}.
    """

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype | str = "auto",
        device: str = "cpu",
        binary: bool = False,
        agg: str = "mean",
    ):
        """
        Args:
            model_path: HuggingFace model name/local path.
            torch_dtype: Checkpoint loading dtype. ``"auto"`` preserves its stored dtype.
            device: Compute device ("cpu", "cuda:0", ...).
            binary: Use sign() approximation (~64x faster, approximate).
            agg: GQA aggregation: "mean", "min", or "max". Ignored for MHA.
        """
        if agg not in ("mean", "min", "max"):
            raise ValueError(f"agg must be 'mean', 'min', or 'max', got {agg!r}")

        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device = device
        self.binary = binary
        self.agg = agg
        self._model = None
        self._config = None
        self._info = None

        # Results populated by classify().
        self.eff_ranks_q: np.ndarray | None = None    # (num_layers, H)
        self.eff_ranks_kv: np.ndarray | None = None   # (num_layers, G)
        self.head_labels: np.ndarray | None = None     # (num_layers, G)
        self.sparsity: float | None = None

    def _ensure_model(self):
        """Load model if not yet loaded."""
        if self._model is None:
            print(f"Loading model: {self.model_path}")
            self._model, self._config = load_model(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto",
            )
            self._info = get_model_info(self._config)
            info = self._info
            mode = "GQA" if info["is_gqa"] else "MHA"
            print(
                f"  {mode}: H={info['H']} G={info['G']} GRP={info['GRP']} "
                f"d_head={info['d_head']} layers={info['num_layers']}"
            )
            if info["is_gqa"]:
                print(f"  Aggregation strategy: {self.agg}")

    def classify(self, sparsity: float = 0.5) -> np.ndarray:
        """Classify all heads across all layers.

        Args:
            sparsity: Fraction of KV heads per layer classified as Streaming.
                      E.g. 0.5 → half Streaming, half Retrieval.

        Returns:
            head_labels: (num_layers, G) with 1.0 = Retrieval, 0.0 = Streaming.
        """
        self.sparsity = sparsity
        self._ensure_model()
        info = self._info
        num_layers, G, GRP = info["num_layers"], info["G"], info["GRP"]

        # Step 1: per-Q-head effective ranks
        print(f"\nStep 1/2: Computing per-Q-head effective ranks "
              f"({'binary' if self.binary else 'float'} mode) ...")
        self.eff_ranks_q = compute_all_eff_ranks(
            self._model, self._config,
            binary=self.binary, device=self.device,
        )

        # Step 2: aggregate to KV-group level
        if info["is_gqa"]:
            print(f"Step 2/2: Aggregating to KV-group level (agg={self.agg}) ...")
            eff_ranks_kv = np.zeros((num_layers, G), dtype=np.float64)
            for l in range(num_layers):
                for g in range(G):
                    q_ranks = self.eff_ranks_q[l, g * GRP : (g + 1) * GRP]
                    if self.agg == "mean":
                        eff_ranks_kv[l, g] = q_ranks.mean()
                    elif self.agg == "min":
                        eff_ranks_kv[l, g] = q_ranks.min()
                    else:  # max
                        eff_ranks_kv[l, g] = q_ranks.max()
        else:
            print("Step 2/2: MHA — no aggregation needed.")
            eff_ranks_kv = self.eff_ranks_q.copy()

        self.eff_ranks_kv = eff_ranks_kv

        # Step 3: per-layer classification
        k = max(0, min(G, round((1.0 - sparsity) * G)))  # Retrieval count
        head_labels = np.zeros((num_layers, G), dtype=np.float64)
        for l in range(num_layers):
            sorted_idx = np.argsort(eff_ranks_kv[l])  # ascending
            retrieval_idx = sorted_idx[:k]
            head_labels[l, retrieval_idx] = 1.0

        self.head_labels = head_labels

        retrieval_ratio = head_labels.mean()
        print(f"\nClassification done:  Retrieval={retrieval_ratio:.2%}  "
              f"Streaming={1 - retrieval_ratio:.2%}  "
              f"(k={k}/{G} per layer)")

        return head_labels

    def save(self, output_dir: str) -> None:
        """Save all results.

        Args:
            output_dir: Target directory (created if needed).
        """
        if self.head_labels is None:
            raise RuntimeError("Call classify() before save().")

        info = self._info
        metadata = {
            "model": self.model_path,
            "method": "AoH (Autonomy-of-Heads)",
            "mode": "GQA" if info["is_gqa"] else "MHA",
            "agg": self.agg if info["is_gqa"] else "N/A",
            "binary": self.binary,
            "sparsity": self.sparsity,
            "num_layers": info["num_layers"],
            "num_q_heads": info["H"],
            "num_kv_heads": info["G"],
            "group_size": info["GRP"],
            "d_head": info["d_head"],
        }

        save_results(
            output_dir,
            eff_ranks_q=self.eff_ranks_q,
            eff_ranks_kv=self.eff_ranks_kv,
            head_labels=self.head_labels,
            metadata=metadata,
        )

        print(f"Results saved to: {output_dir}")

    @classmethod
    def from_saved(cls, output_dir: str) -> "AoHClassifier":
        """Load previously saved results (without model or recomputation)."""
        data = load_results(output_dir)
        meta = data["metadata"] or {}

        obj = cls.__new__(cls)
        obj.model_path = meta.get("model", "")
        obj.torch_dtype = torch.float32
        obj.device = "cpu"
        obj.binary = meta.get("binary", False)
        obj.agg = meta.get("agg", "mean")
        obj.sparsity = meta.get("sparsity", None)
        obj._model = None
        obj._config = None
        obj._info = {
            "num_layers": meta.get("num_layers", 0),
            "H": meta.get("num_q_heads", 0),
            "G": meta.get("num_kv_heads", 0),
            "GRP": meta.get("group_size", 1),
            "d_head": meta.get("d_head", 0),
            "d_model": 0,
            "is_gqa": meta.get("mode") == "GQA",
        }
        obj.eff_ranks_q = data["eff_ranks_q"]
        obj.eff_ranks_kv = data["eff_ranks_kv"]
        obj.head_labels = data["head_labels"]
        return obj
