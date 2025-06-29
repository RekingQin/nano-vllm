import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        """
        hidden_size：输入最后一维的特征数。
        eps：防止除零的小常数，默认1e-6。
        self.weight：可学习的缩放参数（与LayerNorm类似），shape为(hidden_size,)，初始为1。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        类型转换：先转为float32，保证数值稳定。
        均方根归一化：
            var = x.pow(2).mean(dim=-1, keepdim=True)：计算最后一维的均方（方差）。
            torch.rsqrt(var + self.eps)：开方再取倒数，得到RMS的倒数。
            x.mul_(...)：原地归一化。
        恢复原始类型并缩放：转回原始dtype（如float16/bfloat16），再乘以可学习参数self.weight。
        返回归一化后的x。
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        支持残差连接的RMSNorm归一化，常用于Transformer结构。
        类型转换：x和residual都转为float32。
        加残差：x += residual。
        保存加完残差的结果：residual = x.to(orig_dtype)。
        归一化：同rms_forward。
        返回：归一化后的x，以及加完残差但未归一化的residual。
        """
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        如果没有残差，直接做RMSNorm。
        如果有残差，先加残差再做RMSNorm，并返回归一化结果和加完残差的结果。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
