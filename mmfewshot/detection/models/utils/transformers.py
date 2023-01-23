# TODO: implement Multi-head attention here.
import torch.nn as nn
from torch import flatten, reshape, permute

class MultiHeadAttention(nn.Module):
    pass


class TransformerBlock(nn.Module):
    pass


class CrossAttentionTransformer(nn.Module):
    """A single CAT layer. Based on https://arxiv.org/abs/2104.14984.

    Args:
        in_channels (int): Number of input features channels.
        num_heads (int): Number of heads used in each CAT layer for Multi-head attention.
        embed_size (int): dimension for the hidden embeddings.
        dropout_prob (float): Dropout probability.

    """
    def __init__(self,
                 in_channels: int,
                 num_heads: int,
                 embed_size: int,
                 dropout_prob: float):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.dropout_prob = dropout_prob
        '''
        self.attention_layer_query = MultiHeadAttention(in_channels=self.in_channels,
                                                        num_heads=self.num_heads,
                                                        embed_size=self.embed_size)
        self.attention_layer_support = MultiHeadAttention(in_channels=self.in_channels,
                                                          num_heads=self.num_heads,
                                                          embed_size=self.embed_size)
        self.dropout = nn.Dropout(dropout_prob)

        self.query_norm1 = nn.LayerNorm(self.embed_size)



    def forward(self, x_query, x_support):
        """Calculate a single layer of CAT. Based on https://arxiv.org/abs/2104.14984

        Args:
            x_query (Tensor): Input query features with shape (N, C, H_q * W_q).
            x_support (Tensor): Input support features with shape (N, C, H_s * W_s).

        Returns:
            Tensor: Aggregated features for query, its shape is (N, C, H_q * W_q).
            Tensor: Aggregated features for support, its shape is (N, C, H_s * W_s).
        """
        # TODO: add positional encodings to both feature maps before attention.
        '''
        attention_query = attention_layer_query(queries=x_query,
                                                keys=x_support,
                                                values=x_support)
        attention_support = attention_layer_support(queries=x_support,
                                                    keys=x_query,
                                                    values=x_query)

        x_query = attention_query + x_query
        '''


        return x_query, x_support


class CrossAttentionTransformerBlock(nn.Module):
    """Cross-Attention Transformer block. Based on https://arxiv.org/abs/2104.14984.

    Args:
        in_channels (int): Number of input features channels.
        num_layers (int): Number of CAT layers.
        num_heads (int): Number of heads used in each CAT layer for Multi-head attention.
        embed_size (int): Dimension for the hidden embeddings.
        dropout_prob (float): Dropout probability.

    """
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 num_heads: int,
                 embed_size: int,
                 dropout_prob: float):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers  # delete these if they aren't used anymore
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.dropout_prob = dropout_prob
        self.query_compression = nn.Sequential(
            # TODO: reduce output channels here? Add ReLU?
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same'),
            nn.Conv2d(self.in_channels, self.embed_size, 1, padding='same'),
        )
        self.support_compression = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, padding='same'),
            nn.Conv2d(self.in_channels, self.embed_size, 1, padding='same'),
        )
        self.dropout = nn.Dropout(dropout_prob)

        self.layers = nn.ModuleList(
            [
                CrossAttentionTransformer(in_channels=self.in_channels,
                                          num_heads=self.num_heads,
                                          embed_size= self.embed_size,
                                          dropout_prob=self.dropout_prob,
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

        # Compress channels from feature maps
        x_query = self.query_compression(x_query)        # (N, embed_size, H_q, W_q)
        x_support = self.support_compression(x_support)  # (N, embed_size, H_s, W_s)

        print("  Sizes after channel compression:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")

        # Save original sizes to reshape the output from attention
        x_query_shape = x_query.size()
        x_support_shape = x_support.size()

        # Flatten spatial dimensions
        x_query = flatten(x_query, start_dim=-2, end_dim=-1)      # (N, embed_size, H_q * W_q)
        x_support = flatten(x_support, start_dim=-2, end_dim=-1)  # (N, embed_size, H_s * W_s)
        print("  Sizes after flatten of spatial dimensions:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")

        # Permute last dimensions
        x_query = permute(x_query, (0, 2, 1))      # (N, H_q * W_q, embed_size)
        x_support = permute(x_support, (0, 2, 1))  # (N, H_s * W_s, embed_size)
        print("  Sizes after dims permutation:")
        print(f"    x_query.size() = {x_query.size()}")
        print(f"    x_support.size() = {x_support.size()}")

        x_query = self.dropout(x_query)
        x_support = self.dropout(x_support)

        for layer in self.layers:
            x_query, x_support = layer(x_query, x_support)

        # Reshape both query and support features to its original sizes
        x_query = reshape(x_query, x_query_shape)
        x_support = reshape(x_support, x_support_shape)

        return x_query, x_support
