# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv import ConfigDict

from mmfewshot.detection.models.dense_heads import TransformerNeckRPNHead


def test_transformer_neck_rpn_head():
    num_support_ways = 2
    num_support_shots = 5
    # Tests attention_rpn loss when truth is empty and non-empty.
    s = 256
    img_metas = [{
        'img_shape': (s, s, 3),
        'scale_factor': 1,
        'pad_shape': (s, s, 3)
    }]
    config = ConfigDict(
        # in_channels=64,
        # in_channels=256,  # Needs to match emded_size from CrossAttentionAggregator
        in_channels=1024,
        # feat_channels=64,
        feat_channels=1024,
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            # out_channels=64,
            out_channels=1024,
            featmap_strides=[16]),
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0],
            scale_major=False,
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        aggregation_layer=dict(
            type='AggregationLayer',
            aggregator_cfgs=[
                dict(type='DummyAggregator')
            ]),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=6000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0))
    proposal_cfg = ConfigDict(
        nms_pre=12000,
        max_per_img=2000,
        nms=dict(type='nms', iou_threshold=0.7),
        min_bbox_size=0)

    self = TransformerNeckRPNHead(**config)
    gt_bboxes = [
        torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]),
    ]
    # query_feats = [torch.rand(1, 64, s // 8, s // 8)]  # (N, C, H_q, W_q)
    # support_feats = [torch.rand(4, 64, 20, 20)]  # (N * num_support_ways * num_support_shots, C, H_s, W_s)
    batch_size = 2
    samples_per_query = num_support_ways * num_support_shots
    print(f"batch_size = {batch_size}")
    print(f"num_support_ways = {num_support_ways}")
    print(f"num_support_shots = {num_support_shots}")
    print(f"samples_per_query = {samples_per_query}")

    query_feats = [torch.rand(batch_size * samples_per_query, 1024, s // 16, s // 16)]  # (N * num_support_ways * num_support_shots, C, H_q, W_q)
    support_feats = [torch.rand(batch_size * samples_per_query, 1024, 20, 20)]  # (N * num_support_ways * num_support_shots, C, H_s, W_s)

    print(f"Original query features shape: {query_feats[0].size()}")
    print(f"Original support features shape: {support_feats[0].size()}")

    print("First call to forward_train...")
    losses, proposal_list = self.forward_train(
        query_feats,
        support_feats,
        query_img_metas=img_metas * batch_size,
        query_gt_bboxes=gt_bboxes * batch_size,
        support_img_metas=img_metas * samples_per_query * batch_size,
        support_gt_bboxes=gt_bboxes * samples_per_query * batch_size,
        proposal_cfg=proposal_cfg)

    assert sum(losses['loss_rpn_cls']).item() > 0
    assert sum(losses['loss_rpn_bbox']).item() > 0
    # print(f"len(proposal_list) = {len(proposal_list)}")
    assert len(proposal_list) == num_support_ways * num_support_shots * batch_size

    print("Second call to forward_train...")
    losses, proposal_list = self.forward_train(
        query_feats,
        support_feats,
        query_img_metas=img_metas * batch_size,
        query_gt_bboxes=[torch.empty((0, 4))] * batch_size,
        support_img_metas=img_metas * samples_per_query * batch_size,
        support_gt_bboxes=gt_bboxes * samples_per_query * batch_size,
        proposal_cfg=proposal_cfg)

    assert sum(losses['loss_rpn_cls']).item() > 0
    assert sum(losses['loss_rpn_bbox']).item() == 0
    # print(f"len(proposal_list) = {len(proposal_list)}")
    assert len(proposal_list) == num_support_ways * num_support_shots * batch_size

    # Test simple test
    print("Simple test...")
    proposal_list = self.simple_test(query_feats, torch.rand(1, 1024, 20, 20),
                                     img_metas)
    assert proposal_list[0].size(0) == 100


if __name__ == "__main__":
    test_transformer_neck_rpn_head()
