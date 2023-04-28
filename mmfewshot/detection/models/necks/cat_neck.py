import torch.nn as nn
from ..builder import NECKS

@NECKS.register_module()
class CATNeck(nn.Module):
    '''
    Performs Cross Attention Transformer block between query and support features.
    Based on https://arxiv.org/abs/2104.14984

    Create a symbolic link to this file in:
    miniconda3/envs/openmmlab/lib/python3.7/site-packages/mmdet/models/necks
    '''

    # TODO: define arguments for the constructor.
    def __init__(self,
                in_channels,
                out_channels,):
                # num_outs,
                # start_level=0,
                # end_level=-1,
                # add_extra_convs=False):
        print("Building CATNeck block...")
        super(CATNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    # def forward(self, inputs):
    def forward(self, query_feats, support_feats):
        print("Entering forward in CATNeck...")
        # TODO: use the same CAT block implemented on the AggregationLayer.
        # Study the structure of query and support features: backbone's output
        # is just the last ResNet block or all of them?
        return query_feats, support_feats
