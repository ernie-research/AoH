"""KV cache spec + manager for AoH streaming heads: keep the first `sink` tokens and the last `sliding_window` tokens.

vLLM's SlidingWindowManager frees every block before the window, which would also drop the attention sinks.
SinkSlidingWindowManager keeps the first ceil(sink / block_size) blocks of each request and frees only the
blocks strictly between the sinks and the window.

Dense prefill (the DuoAttention LongBench protocol): prompt tokens before num_prompt_tokens - dense_tail use full
causal attention on the streaming heads too, so no block is freed until that many tokens are computed; afterwards
the request drops to sink + window.
"""
from __future__ import annotations

from dataclasses import dataclass

from vllm.utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec


@dataclass(frozen=True)
class SinkSlidingWindowSpec(SlidingWindowSpec):
    sink: int = 0
    dense_tail: int = 50  # the last dense_tail prompt tokens use sink + window, earlier ones full attention

    def max_memory_usage_bytes(self, vllm_config) -> int:
        # a request holds its whole prompt until the dense part is computed
        return cdiv(vllm_config.model_config.max_model_len, self.block_size) * self.page_size_bytes


class SinkSlidingWindowManager(SlidingWindowManager):
    # request_id -> num_prompt_tokens, filled by the KVCacheManager.allocate_slots wrapper in __init__.register
    prompt_lens: dict[str, int] = {}

    def __init__(self, kv_cache_spec: SinkSlidingWindowSpec, block_pool, **kwargs) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.num_sink_blocks = cdiv(kv_cache_spec.sink, self.block_size)
        self.dense_tail = kv_cache_spec.dense_tail

    @classmethod
    def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids, block_pool, kv_cache_spec,
                               use_eagle):
        # Prefix caching is not supported for sink + window layers: report no hit.
        return tuple([] for _ in kv_cache_group_ids)

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        if num_computed_tokens < self.prompt_lens.get(request_id, 0) - self.dense_tail:
            return
        last_useful_block = (num_computed_tokens - self.sliding_window + 1) // self.block_size
        blocks = self.req_to_blocks[request_id]
        removed = []
        for i in range(last_useful_block - 1, self.num_sink_blocks - 1, -1):
            if blocks[i] == self._null_block:
                break
            removed.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed)

    def free(self, request_id: str) -> None:
        self.prompt_lens.pop(request_id, None)
        super().free(request_id)
