from ..builder import NECKS

@NECKS.register_module()
class CATNeck(nn.Module):

    '''
    Performs Cross Attention Transformer block between query and support features.
    Base on https://arxiv.org/abs/2104.14984
    '''

    # TODO: define arguments for the constructor.
    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                start_level=0,
                end_level=-1,
                add_extra_convs=False):
        pass

    # def forward(self, inputs):
    def forward(self, query_feats, support_feats):
        # TODO: use the same CAT block implemented on the AggregationLayer.
        # Study the structure of query and support features: backbone's output
        # is just the last ResNet block or all of them?
        pass
