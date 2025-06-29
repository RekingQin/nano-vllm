from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    :param x: for query or key tensor
    :param cos: pre-computed position encoding info
    :param sin: pre-computed position encoding info
    :return: embeddings + positions
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    x1, x2 = torch.chunk(x.to(torch.float32), 2, dim=-1)  # split x last dim into 2 parts
    # rotate x from a two-dimensional view
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    construct rope cosine/sine encoding table

    head_size：每个注意力头的维度。
    rotary_dim：参与RoPE的维度（通常等于head_size）。
    max_position_embeddings：最大支持的位置数。
    base：频率基数，常用10000。
    inv_freq：每个维度的倒频率，决定不同维度的旋转速度。
    freqs：每个位置和每个维度的角度。
    cos/sin：每个位置、每个维度的余弦/正弦值。
    cache：将cos和sin拼在一起，shape为(max_position, rotary_dim*2)。
    register_buffer：将cache注册为模型的buffer，随模型保存/加载。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # pay attention to the assertion
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        positions：每个token的位置索引，shape为(num_tokens,)。
        cos_sin = self.cos_sin_cache[positions]：查表，获得每个token的cos/sin编码。
        chunk(2, dim=-1)：分成cos和sin两部分。
        view操作：保证query/key的shape适配RoPE操作。
        apply_rotary_emb：对query/key分别应用旋转编码。
        最后恢复原shape，返回
        """
        num_tokens = positions.size(0)
        cos_sin = self.cos_sin_cache[positions] # get each token cos/sin encoding value
        cos, sin = cos_sin.chunk(2, dim=-1)
        # it seems query & key could be fused into one operation
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
