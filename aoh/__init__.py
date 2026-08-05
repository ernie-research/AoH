"""AoH (Autonomy-of-Heads): Training-free head classification via effective rank.

Public API
----------

.. code-block:: python

    from aoh import AoHClassifier

    # GQA model (e.g. Qwen2.5-7B: H=28, G=4)
    cls = AoHClassifier(
        "Qwen/Qwen2.5-7B",
        device="cuda:0",
        agg="mean",          # also supports "min" and "max"
    )
    labels = cls.classify(sparsity=0.5)   # (28 layers, 4 KV groups)
    cls.save("outputs/aoh")

    # MHA model (e.g. Llama-2-7B: H=32, G=32)
    cls = AoHClassifier("meta-llama/Llama-2-7b-hf", device="cuda:0")
    labels = cls.classify(sparsity=0.5)   # (32 layers, 32 heads)

CLI::

    python -m aoh --model_path Qwen/Qwen2.5-7B --sparsity 0.5 --output_dir out/
"""

from .classifier import AoHClassifier
from .effective_rank import compute_eff_rank, compute_all_eff_ranks
from .io import load_results

__all__ = [
    "AoHClassifier",
    "compute_eff_rank",
    "compute_all_eff_ranks",
    "load_results",
]
