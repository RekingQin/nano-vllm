from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    is_prefill: 是否为“预填充”阶段（如大模型推理时，区分prefill和decode阶段）。
    cu_seqlens_q: 查询序列的累积长度（cumulative sequence lengths），通常用于batch序列处理。
    cu_seqlens_k: 键序列的累积长度。
    max_seqlen_q: 查询序列的最大长度。
    max_seqlen_k: 键序列的最大长度。
    slot_mapping: 槽位映射，常用于缓存或分配机制。
    context_lens: 上下文长度，可能用于动态padding或mask。
    block_tables: 块表，可能用于分块存储或索引。
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
