import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """
    作用：将加载的权重张量loaded_weight复制到模型参数param的.data属性中。
    用法：作为默认的权重加载函数。
    """
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    尝试从模型对象model获取packed_modules_mapping属性，如果没有则用空字典。
    这个映射通常用于处理模型参数名的映射（比如分片、合并等）。


    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():  # read all parameters name
                """
                遍历packed_modules_mapping中的每个键k，如果k在weight_name中出现：
                v, shard_id = packed_modules_mapping[k]：获取映射后的参数名v和分片IDshard_id。
                param_name = weight_name.replace(k, v)：将权重名中的k替换为v，得到模型中实际参数名。
                param = model.get_parameter(param_name)：获取模型参数对象。
                weight_loader = getattr(param, "weight_loader")：获取参数的weight_loader属性（必须存在）。
                weight_loader(param, f.get_tensor(weight_name), shard_id)：调用权重加载函数，传入分片ID。
                """
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")  # reflection get corresponding weight loader
                        weight_loader(param, f.get_tensor(weight_name), shard_id)  # get shard_id corresponding weight
                        break
                else:
                    """
                    如果weight_name不在packed_modules_mapping中：
                    直接用weight_name获取参数。
                    获取参数的weight_loader属性，如果没有则用default_weight_loader。
                    调用权重加载函数。
                    """
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)  # using default weight loader
                    weight_loader(param, f.get_tensor(weight_name))
