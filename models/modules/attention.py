from torch import Tensor
import torch.nn as nn

class MultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim: int, num_heads: int = 8):
        assert embed_dim % num_heads == 0, 'num_heads cannot devide embed_dim.'
        super().__init__(embed_dim, num_heads, batch_first=True)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None, return_attn_map=False) -> Tensor:
        out, attn_map = super().forward(query, key, value, attn_mask=mask)
        if return_attn_map:
            return out, attn_map
        else:
            return out
    
    def count_extra_flops(m, args, kwargs, output):
        from utils.count_ops import count_mha
        count_mha(m, args, kwargs, output)
