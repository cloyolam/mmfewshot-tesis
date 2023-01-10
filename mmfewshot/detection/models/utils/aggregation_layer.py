# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmcv.utils import ConfigDict
from mmdet.models.builder import MODELS
from torch import Tensor
from .transformers import TransformerBlock

# AGGREGATORS are used for aggregate features from different data
# pipelines in meta-learning methods, such as attention rpn.
AGGREGATORS = MODELS


def build_aggregator(cfg: ConfigDict) -> nn.Module:
    """Build aggregator."""
    return AGGREGATORS.build(cfg)


@AGGREGATORS.register_module()
class AggregationLayer(BaseModule):
    """Aggregate query and support features with single or multiple aggregator.
    Each aggregator return aggregated results in different way.

    Args:
        aggregator_cfgs (list[ConfigDict]): List of fusion function.
        init_cfg (ConfigDict | None): Initialization config dict. Default: None
    """

    def __init__(self,
                 aggregator_cfgs: List[ConfigDict],
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.aggregator_list = nn.ModuleList()
        self.num_aggregators = len(aggregator_cfgs)
        aggregator_cfgs_ = copy.deepcopy(aggregator_cfgs)
        for cfg in aggregator_cfgs_:
            self.aggregator_list.append(build_aggregator(cfg))

    def forward(self, query_feat: Tensor,
                support_feat: Tensor) -> List[Tensor]:
        """Return aggregated features of query and support through single or
        multiple aggregators.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with
                shape (N, C, H, W).

        Returns:
            list[Tensor]: List of aggregated features.
        """
        out = []
        for i in range(self.num_aggregators):
            out.append(self.aggregator_list[i](query_feat, support_feat))
        return out


@AGGREGATORS.register_module()
class DepthWiseCorrelationAggregator(BaseModule):
    """Depth-wise correlation aggregator.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
            Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert in_channels is not None, \
            "DepthWiseCorrelationAggregator require config of 'in_channels'."
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_fc = with_fc
        if with_fc:
            assert out_channels is not None, 'out_channels is expected.'
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (1, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """

        '''
        print("ENTERING DepthWiseCorrelation Aggregator")
        print(f"query_feat.size() = {query_feat.size()}")
        print(f"support_feat.size() = {support_feat.size()}")
        '''
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'

        feat = F.conv2d(
            query_feat,
            support_feat.permute(1, 0, 2, 3),
            groups=self.in_channels)
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DifferenceAggregator(BaseModule):
    """Difference aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        feat = query_feat - support_feat
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class DotProductAggregator(BaseModule):
    """Dot product aggregator.

    Args:
        in_channels (int): Number of input features channels. Default: None.
        out_channels (int): Number of output features channels. Default: None.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels: Optional[int] = None,
                 out_channels: Optional[int] = None,
                 with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.with_fc = with_fc
        self.in_channels = in_channels
        self.out_channels = out_channels
        if with_fc:
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (N, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        assert query_feat.size()[1:] == support_feat.size()[1:], \
            'mismatch channel number between query and support features.'
        feat = query_feat.mul(support_feat)
        if self.with_fc:
            assert feat.size(2) == 1 and feat.size(3) == 1, \
                'fc layer requires the features with shape (N, C, 1, 1)'
            feat = self.fc(feat.squeeze(3).squeeze(2))
            feat = self.norm(feat)
            feat = self.relu(feat)
        return feat


@AGGREGATORS.register_module()
class CrossAttentionAggregator(BaseModule):
    """Cross-Attention Transformer aggregator. Based on https://arxiv.org/abs/2104.14984.

    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
            Default: None.  # TODO: delete?
        num_layers (int): Number of CAT layers to use (N = 4 in the paper).
        num_heads (int): Number of heads to use in each CAT layer (M = 8 in the
            paper).
        dropout_ratio (float): Dropout ratio.
        with_fc (bool): Use fully connected layer for aggregated features.
            If set True, `in_channels` and `out_channels` are required.
            Default: False.  # TODO: delete?
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.

    """
    def __init__(self,
                 in_channels: int,
                 # out_channels: Optional[int] = None,
                 num_layers: int,
                 num_heads: int,
                 # dropout_ratio: float = 0.,  # TODO: apply dropout to both target and query?
                 # with_fc: bool = False,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert in_channels is not None, \
            "CrossAttentionAggregator require config of 'in_channels'."
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.transformer_block = TransformerBlock(num_heads=8)
        # self.dropout_layer = nn.Dropout(p=dropout_ratio)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.with_fc = with_fc
        if with_fc:
            assert out_channels is not None, 'out_channels is expected.'
            self.fc = nn.Linear(in_channels, out_channels)
            self.norm = nn.BatchNorm1d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        """

    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H, W).
            support_feat (Tensor): Input support features with shape
                (1, C, H, W).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H, W).
        """
        print("ENTERING CrossAttentionAggregator:")
        print(f"num_layers = {self.num_layers}, num_heads = {self.num_heads}")
        print(f"query_feat.size() = {query_feat.size()}")
        print(f"support_feat.size() = {support_feat.size()}")

        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        x_query = query_feat  # TODO: add Dropout here?
        x_support = support_feat
        print("Applying TransformerBlock...")
        x_query, x_support = self.transformer_block(x_query, x_support)
        print(f"query_feat.size() = {query_feat.size()}")
        print(f"support_feat.size() = {support_feat.size()}")
        '''
        for _ in range(self.num_layers):
            x_query, x_support = transformer_block(self.num_heads, x_query, x_support)
        # TODO: x_query goes to RPN, but x_support should be used in RoI matching?
        '''
        return x_query
