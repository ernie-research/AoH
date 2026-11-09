"""Per-step worker state for dense prefill, set by the GPUModelRunner._prepare_inputs patch in __init__.

dense_len[r]: request r of the current batch uses full causal attention on the streaming heads for positions
< dense_len[r]. any_dense: some scheduled token of this step has such a position.
"""
from typing import Optional

import torch

dense_len: Optional[torch.Tensor] = None
any_dense: bool = False
