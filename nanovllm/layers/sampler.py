import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        :param logits: [bs, vocab_size]
        :param temperatures: [bs]

        输入 logits 和温度。
        温度缩放 logits。
        softmax 得到概率分布。
        Gumbel-Max Trick 采样 token。
        温度为0时用贪婪解码，否则用采样。
        """
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1) # greedy token ids [bs, ]
        logits.div_(temperatures.unsqueeze(dim=1)) # logits / temperature, if temperature == 1, no effects, near to 0, nearly one-hot sampling(argmax)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)   # calculate softmax, get [bs, vocab_size]
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10
        # Gumbel-Max Trick sampling, equally to argmax(log(probs) - Gumbel(0,1)), finally, get sampled token id, get [bs, ]
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1)  
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)  # choose greedy tokens or sample tokens
