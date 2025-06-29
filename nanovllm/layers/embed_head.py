import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    分布式环境
        self.tp_rank：当前进程的rank（编号）。
        self.tp_size：总进程数（通常等于参与并行的GPU数）。
    词表切分
        num_embeddings：总词表大小。
        embedding_dim：每个词的向量维度。
        num_embeddings_per_partition：每个进程负责的词表分片大小。
        vocab_start_idx/vocab_end_idx：本进程负责的词表索引范围。
    参数定义
        self.weight：本进程负责的词向量参数，形状为 [num_embeddings_per_partition, embedding_dim]。
        weight_loader：用于加载分片权重的方法。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0  # vocal size % tp_size == 0 ??? may not need
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader  # assign a weight loader to the class

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        作用：从完整的词表权重中，切分出本进程负责的部分，加载到本地参数。
        shard_size：本进程负责的词数。
        start_idx：本进程负责的词表起始索引。
        loaded_weight.narrow(0, start_idx, shard_size)：从完整权重中切出本进程的分片。
        param_data.copy_(loaded_weight)：将分片权重复制到本地参数。
        """
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.size() == loaded_weight.size()
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        1. 输入索引处理
            x：输入的词索引（如 [batch, seq_len]）。
            如果并行数大于1（即分布式），则：
                mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
                    生成一个布尔掩码，标记哪些词属于本进程负责的词表分片。
                x = mask * (x - self.vocab_start_idx)
                    只保留本分片的词索引，并将其映射到本地分片的索引空间（其余为0）。
        2. 嵌入查找
            y = F.embedding(x, self.weight)
                查找本地分片的词向量。对于不属于本分片的词，查到的是第0行（通常为无效向量）。
        3. 掩码和聚合
            y = mask.unsqueeze(1) * y
                只保留本分片负责的词向量，其余位置为0。
            dist.all_reduce(y)
                所有进程将自己的结果加起来，最终每个进程都得到完整的嵌入输出（只要有一个进程负责该词，输出就正确）。
        4. 返回
            返回聚合后的嵌入结果。
        """
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)  # index start from 0 (shard weight)
        y = F.embedding(x, self.weight)  # look up weight table
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    这个类实现了词表并行的语言模型输出头（Parallel Language Model Head），常用于大模型（如GPT、Llama等）分布式训练的最后一层（即 logits 计算层）

    继承自 VocabParallelEmbedding，即词表并行嵌入层。
    这样可以复用词表分片、权重加载等机制。

    假设词表大小为 10000，embedding_dim=4096，4卡并行：
        每卡负责 2500 个词的 logits 计算。
        输入 x shape 为 [batch, 4096]。
        每卡输出 logits shape 为 [batch, 2500]。
        rank 0 收集所有卡的 logits，拼接后 shape 为 [batch, 10000]，可用于 softmax、采样等。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        """
        num_embeddings：词表大小。
        embedding_dim：嵌入/隐藏层维度。
        bias：是否加偏置。
        通过 super().__init__ 调用父类（VocabParallelEmbedding）初始化，自动完成词表分片、权重分配等。
        如果需要偏置，则每个分片分配一段偏置参数（与本分片词表大小一致），并挂上权重加载器。
        """
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """
        1. 上下文处理
            context = get_context()
                获取当前推理/训练的上下文对象（通常包含序列长度、模式等信息）。
            if context.is_prefill:
                如果是“prefill”阶段（通常指推理时的prompt填充阶段），只取每个序列的最后一个token的隐藏状态用于输出。
                last_indices = context.cu_seqlens_q[1:] - 1
                    计算每个序列的最后一个token的索引。
                x = x[last_indices].contiguous()
                    只保留这些token的隐藏状态。
        2. 线性变换（logits计算）
            logits = F.linear(x, self.weight, self.bias)
                计算 logits，等价于 logits = x @ self.weight.T + self.bias。
                这里的 self.weight 只包含本分片的词表权重，输出 shape 为 [batch, num_embeddings_per_partition]。
        3. 并行聚合
            if self.tp_size > 1:
                如果是多卡并行，每个进程只算了部分词表的 logits。
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
                只有 rank 0 进程分配一个列表用于收集所有分片的 logits。
            dist.gather(logits, all_logits, 0)
                所有进程将自己的 logits 发送到 rank 0，rank 0 收集到 all_logits。
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
                rank 0 把所有分片的 logits 按最后一维拼接，得到完整词表的 logits。
            其他 rank 返回 None（通常只在 rank 0 上做后续处理，如 softmax、采样等）
        """
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)  # gather into rank 0
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
