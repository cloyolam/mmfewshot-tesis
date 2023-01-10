# TODO: implement Multi-head attention here.
import torch.nn as nn

class DotProductAttention():
    pass


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward():
        pass


class TransformerBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 query_shape: tuple,
                 target_shape: tuple):
        super().__init__()
        self.num_heads = num_heads  # delete if it is not used again
        # self.multihead_attention = MultiHeadAttention(self.num_heads)
        self.norm_1 = nn.LayerNorm(normalized_shape=query_shape[1:])  # change these for BN?
        self.norm_2 = nn.LayerNorm(normalized_shape=target_shape[1:])

    def forward(self, x_query, x_support):
        x_query = self.norm_1(x_query)
        x_support = self.norm_2(x_support)
        return x_query, x_support
