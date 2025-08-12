"""
这段代码实现了一个高效的自注意力（Attention）模块，支持KV缓存（KV Cache）和高性能的FlashAttention推理，
适用于大模型推理/训练。代码包含Triton核函数、KV缓存写入、以及Attention前向实现
"""
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    高效地将当前step的key、value写入到全局KV缓存（k_cache, v_cache）中，支持乱序写入（通过slot_mapping）
    用于推理/生成时的KV缓存管理

    idx = tl.program_id(0)：每个线程处理一个token。
    key_offsets/value_offsets：计算当前token在输入key/value中的偏移。
    slot = tl.load(slot_mapping_ptr + idx)：查找当前token应该写入缓存的哪个slot（位置）。
    cache_offsets：计算在缓存中的写入位置。
    tl.store(...)：将key/value写入全局缓存。
    """
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    N：token数量。
    D：每个token的总特征维度（num_heads * head_dim）。
    store_kvcache_kernel[(N,)](...)：每个token一个线程块，批量写入。
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        1. 输入变形
            q, k, v 变形为 [token数, num_heads, head_dim] 或 [token数, num_kv_heads, head_dim]。
        2. 获取上下文
            context = get_context()
            获取当前推理/训练的上下文（如序列长度、slot映射、block tables等）。
        3. KV缓存写入
            如果k_cache和v_cache已分配（numel()>0），则调用store_kvcache，将本step的k/v写入全局缓存。
            context.slot_mapping：指明每个token应该写入缓存的哪个slot（支持乱序/多batch）。
        4. 注意力计算
            Prefill阶段（通常是prompt填充/批量推理）：
                如果有block_tables（前缀缓存），则直接用缓存的k/v。
                调用flash_attn_varlen_func，支持变长序列的高效FlashAttention。
                传入序列长度、分段信息、block_table等。
            Decode阶段（逐token生成）：
                只用缓存的k/v，调用flash_attn_with_kvcache，高效增量推理。
        5. 输出变形
            输出o变形为 [token数, num_heads * head_dim]，与后续线性层对接。
        """
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():  # save kv cache before attention calculation
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)  # store kv cache for each layer
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
