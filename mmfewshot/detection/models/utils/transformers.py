# TODO: implement Multi-head attention here.
import torch.nn as nn

class MultiHeadAttention():
    pass


class TransformerBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 embed_size: int)
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size

    def forward():
        pass


class CrossAttentionTransformer(nn.Module):
    """Calculate num_layers stacked layers of CAT. Based on https://arxiv.org/abs/2104.14984

    Args:
        query_feat (Tensor): Input query features with shape (N, C, H_q, W_q).
        support_feat (Tensor): Input support features with shape
            (N, C, H_s, W_s).

    Returns:
        Tensor: Aggregated features for query, its shape is (N, C, H_q, W_q).
        Tensor: Aggregated features for support, its shape is (N, C, H_s, W_s).
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 embed_size: int):
        super().__init__()
        self.num_layers = num_layers  # delete these if they aren't used another time
        self.num_heads = num_heads
        self.embed_size = embed_size
        # TODO: initialize dropout layer and positional embedding

        self.layers = nn.ModuleList(
            [
                TransformerBlock(num_heads=self.num_heads,
                                 embed_size= self.embed_size
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x_query, x_support):
        # TODO: apply dropout and add positional encoding to both feature maps
        for layer in self.layers:
            x_query, x_support = layer(x_query, x_support)

        return x_query, x_support
