# TODO: implement Multi-head attention here.
import torch.nn as nn
from torch import flatten

class MultiHeadAttention():
    pass


class CrossAttentionTransformer(nn.Module):
    """A single CAT layer. Based on https://arxiv.org/abs/2104.14984.

    Args:
        num_heads (int): Number of heads used in each CAT layer for Multi-head attention.
        embed_size (int): dimension for the hidden embeddings.

    """
    def __init__(self,
                 num_heads: int,
                 embed_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size

    def forward(self, x_query, x_support):
        """Calculate a single layer of CAT. Based on https://arxiv.org/abs/2104.14984

        Args:
            x_query (Tensor): Input query features with shape (N, C, H_q, W_q).
            x_support (Tensor): Input support features with shape (N, C, H_s, W_s).

        Returns:
            Tensor: Aggregated features for query, its shape is (N, C, H_q, W_q).
            Tensor: Aggregated features for support, its shape is (N, C, H_s, W_s).
        """
        return x_query, x_support


class CrossAttentionTransformerBlock(nn.Module):
    """Cross-Attention Transformer block. Based on https://arxiv.org/abs/2104.14984.

    Args:
        in_channels (int): Number of input features channels.
        num_layers (int): Number of CAT layers.
        num_heads (int): Number of heads used in each CAT layer for Multi-head attention.
        embed_size (int): dimension for the hidden embeddings.

    """
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 num_heads: int,
                 embed_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers  # delete these if they aren't used anymore
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.query_compression = nn.Sequential(
            # TODO: reduce output channels here? Add ReLU?
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same'),
            nn.Conv2d(self.in_channels, self.embed_size, 1, padding='same'),
        )
        self.support_compression = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same'),
            nn.Conv2d(self.in_channels, self.embed_size, 1, padding='same'),
        )
        # TODO: initialize dropout layers

        self.layers = nn.ModuleList(
            [
                CrossAttentionTransformer(num_heads=self.num_heads,
                                          embed_size= self.embed_size
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, x_query, x_support):
        """Calculate num_layers stacked layers of CAT. Based on https://arxiv.org/abs/2104.14984

        Args:
            x_query (Tensor): Input query features with shape (N, C, H_q, W_q).
            x_support (Tensor): Input support features with shape (N, C, H_s, W_s).

        Returns:
            Tensor: Aggregated features for query, its shape is (N, C, H_q, W_q).
            Tensor: Aggregated features for support, its shape is (N, C, H_s, W_s).
        """
        print("ENTERING forward in CrossAttentionTransformerBlock...")
        print(f"x_query.size() = {x_query.size()}")
        print(f"x_support.size() = {x_support.size()}")
        x_query = self.query_compression(x_query)
        x_support = self.support_compression(x_support)
        print("  Sizes after compression:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")
        print("  Sizes after reshape:")
        print(f"    x_query.size() = {flatten(x_query).size()}")
        print(f"    x_support.size() = {x_support.size()}")

        # TODO: apply dropout to both feature maps
        for layer in self.layers:
            x_query, x_support = layer(x_query, x_support)

        return x_query, x_support
