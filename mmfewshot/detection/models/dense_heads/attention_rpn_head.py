# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner import force_fp32
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi, images_to_levels, multi_apply
from mmdet.models import RPNHead
from mmdet.models.builder import HEADS, build_roi_extractor
from torch import Tensor

from mmfewshot.detection.models.utils import build_aggregator


@HEADS.register_module()
class AttentionRPNHead(RPNHead):
    """RPN head for `Attention RPN <https://arxiv.org/abs/1908.01998>`_.

    Args:
        num_support_ways (int): Number of sampled classes (pos + neg).
        num_support_shots (int): Number of shot for each classes.
        aggregation_layer (dict): Config of `aggregation_layer`.
        roi_extractor (dict): Config of `roi_extractor`.
    """

    def __init__(self,
                 num_support_ways: int,
                 num_support_shots: int,
                 aggregation_layer: Dict = dict(
                     type='AggregationLayer',
                     aggregator_cfgs=[
                         dict(
                             type='DepthWiseCorrelationAggregator',
                             in_channels=1024,
                             with_fc=False)
                     ]),
                 roi_extractor: Dict = dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=14, sampling_ratio=0),
                     out_channels=1024,
                     featmap_strides=[16]),
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        assert roi_extractor is not None, \
            'missing config of roi_extractor.'
        assert aggregation_layer is not None, \
            'missing config of aggregation_layer.'
        self.aggregation_layer = \
            build_aggregator(copy.deepcopy(aggregation_layer))
        self.roi_extractor = \
            build_roi_extractor(copy.deepcopy(roi_extractor))

    def extract_roi_feat(self, feats: List[Tensor], rois: Tensor) -> Tensor:
        """Forward function.

        Args:
            feats (list[Tensor]): Input features with shape (N, C, H, W).
            rois (Tensor): with shape (m, 5).

         Returns:
            Tensor: RoI features with shape (N, C, H, W).
        """
        return self.roi_extractor(feats, rois)

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      query_gt_bboxes: List[Tensor],
                      query_img_metas: List[Dict],
                      support_gt_bboxes: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      proposal_cfg: Optional[ConfigDict] = None,
                      **kwargs) -> Tuple[Dict, List[Tuple]]:
        """Forward function in training phase.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W)..
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            query_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                query image, each item with shape (num_gts, 4).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            support_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                support image, each item with shape (num_gts, 4).
            query_gt_bboxes_ignore (list[Tensor]): List of ground truth bboxes
                to be ignored of query image with shape (num_ignored_gts, 4).
                Default: None.
            proposal_cfg (:obj:`ConfigDict`): Test / postprocessing
                configuration. if None, test_cfg would be used. Default: None.

        Returns:
            tuple: loss components and proposals of each image.

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - proposal_list (list[Tensor]): Proposals of each image.
        """

        '''
        print("Entering forward_train in AttentionRPNHead...")
        print(f"  self.num_support_ways = {self.num_support_ways}")
        print(f"  self.num_support_shots = {self.num_support_shots}")
        print(f"  self.cls_out_channels = {self.cls_out_channels}")
        print(f"  self.num_anchors = {self.num_anchors}")

        print(f"  query_feats[0].size() = {query_feats[0].size()}")
        print(f"  support_feats[0].size() = {support_feats[0].size()}")
        print(f"  len(support_gt_bboxes) = {len(support_gt_bboxes)}")
        '''

        query_feat = query_feats[0]
        # convert a list of bboxes to roi format.
        # returns Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
        support_rois = bbox2roi([bboxes for bboxes in support_gt_bboxes])
        support_roi_feats = self.extract_roi_feat(support_feats, support_rois)

        '''
        print(f"  query_feat.size() = {query_feat.size()}")
        print(f"  support_rois.size() = {support_rois.size()}")
        print(f"  suppot_rois = {support_rois}")
        print(f"  support_roi_feats.size() = {support_roi_feats.size()}")
        '''

        # support features are placed in follow order:
        # [pos * num_support_shots,
        #  neg * num_support_shots * (num_support_ways - 1 )] * batch size

        # get the average features:
        # [pos_avg, neg_avg * (num_support_ways - 1 )] * batch size
        # [(1, C, H_q, W_q)_{q1_pos}, (1, C, H_q, W_q)_{q1_neg},...,(1, C, H_q, W_q)_{qN_pos}, (1, C, H_q, W_q)_{qN_neg}]
        avg_support_feats = [
            support_roi_feats[i * self.num_support_shots:(i + 1) *
                              self.num_support_shots].mean([0, 2, 3],
                                                           keepdim=True)
            for i in range(
                support_roi_feats.size(0) // self.num_support_shots)
        ]

        '''
        print("  avg_support_feats:")
        print(f"    len(avg_support_feats) = {len(avg_support_feats)}")
        for ix, tensor in enumerate(avg_support_feats):
            print(f"    {ix}: {tensor.size()}")
        '''

        # Concat all positive pair features
        # [(1, C, H_q, W_q)_q1, ... (1, C, H_q, W_q)_qN]
        pos_pair_feats = [
            self.aggregation_layer(
                query_feat=query_feat[i].unsqueeze(0),
                support_feat=avg_support_feats[i * self.num_support_ways])[0]
            for i in range(query_feat.size(0))
        ]

        '''
        print("  pos_pair_feats:")
        print(f"    len(pos_pair_feats) = {len(pos_pair_feats)}")
        for ix in range(len(pos_pair_feats)):
            print(f"    pos_pair_feats[{ix}] = {pos_pair_feats[ix].size()}")
        '''

        # Concat all negative pair features
        # [(1, C, H_q, W_q)_q1, ... (1, C, H_q, W_q)_qN]
        neg_pair_feats = [
            self.aggregation_layer(
                query_feat=query_feat[i].unsqueeze(0),
                support_feat=avg_support_feats[i * self.num_support_ways + j +
                                               1])[0]
            for i in range(query_feat.size(0))
            for j in range(self.num_support_ways - 1)
        ]

        '''
        print("  neg_pair_feats:")
        print(f"    len(neg_pair_feats) = {len(neg_pair_feats)}")
        for ix in range(len(neg_pair_feats)):
            print(f"    neg_pair_feats[{ix}] = {neg_pair_feats[ix].size()}")
        '''

        batch_size = len(query_img_metas)
        # print(f"  batch_size = {batch_size}")
        # input features for losses: [pos_pair_feats, neg_pair_feats]
        # pair_flags are used to set all the gt_label from negative pairs to
        # bg classes in losses. True means positive pairs and False means
        # negative pairs

        # add positive pairs
        pair_flags = [True for _ in range(batch_size)]
        repeat_query_img_metas = copy.deepcopy(query_img_metas)
        repeat_query_gt_bboxes = copy.deepcopy(query_gt_bboxes)

        '''
        print(f"  len(pair_flags) = {len(pair_flags)}")
        print(f"  len(repeat_query_img_metas) = {len(repeat_query_img_metas)}")
        print(f"  len(repeat_query_gt_bboxes) = {len(repeat_query_gt_bboxes)}")
        '''

        # repeat the query_img_metas and query_gt_bboxes to match
        # the order of positive and negative pairs
        for i in range(batch_size):
            repeat_query_img_metas.extend([query_img_metas[i]] *
                                          (self.num_support_ways - 1))  # N * n_ways
            repeat_query_gt_bboxes.extend([query_gt_bboxes[i]] *
                                          (self.num_support_ways - 1))  # N * n_ways
            # add negative pairs
            pair_flags.extend([False] * (self.num_support_ways - 1))

        '''
        print("  After repeating:")
        print(f"  len(repeat_query_img_metas) = {len(repeat_query_img_metas)}")
        print(f"  repeat_query_img_metas: {repeat_query_img_metas}")
        print(f"  len(repeat_query_gt_bboxes) = {len(repeat_query_gt_bboxes)}")
        for ix, tensor in enumerate(repeat_query_gt_bboxes):
            print(f"    {ix}: {tensor.size()}")
        print(f"  pair_flags = {pair_flags}")
        '''

        outs = self([torch.cat(pos_pair_feats + neg_pair_feats)])

        '''
        print(f"  len(outs) = {len(outs)}")
        for ix, lista in enumerate(outs):
            print(f"  {ix}: {type(lista)}, len(lista) = {len(lista)}")
            for ix2, tensor in enumerate(lista):
                print(f"    {ix}: {tensor.size()}")
        '''

        loss_inputs = outs + (repeat_query_gt_bboxes, repeat_query_img_metas)

        '''
        print(f"  type(loss_inputs) = {type(loss_inputs)}")
        print(f"  len(loss_inputs) = {len(loss_inputs)}")
        print("  loss_inputs:")
        for ix, lista in enumerate(loss_inputs):
            print(f"    {ix}: {type(lista)}, len(tensor)={len(lista)}")
            for ix2, tensor in enumerate(lista):
                if ix < len(loss_inputs) - 1:
                    print(f"      {ix2}: {tensor.size()}")
                else:
                    print(f"      {ix2}: {tensor}")
        '''

        losses = self.loss(
            *loss_inputs,
            gt_bboxes_ignore=query_gt_bboxes_ignore,
            pair_flags=pair_flags)
        proposal_list = self.get_bboxes(
            *outs, img_metas=repeat_query_img_metas, cfg=proposal_cfg)

        '''
        print(f"  len(proposal_list) = {len(proposal_list)}")
        for ix, proposal in enumerate(proposal_list):
            print(f"    {ix}: {proposal.size()}")
            print(f"          {proposal}")
        print(f"  proposal_list = {proposal_list}")
        '''
        return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores: List[Tensor],
             bbox_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             img_metas: List[Dict],
             gt_labels: Optional[List[Tensor]] = None,
             gt_bboxes_ignore: Optional[List[Tensor]] = None,
             pair_flags: Optional[List[bool]] = None) -> Dict:
        """Compute losses of rpn head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
                Default: None.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss. Default: None
            pair_flags (list[bool]): Indicate predicted result is from positive
                pair or negative pair with shape (N). Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        '''
        print("Entering loss in AttentionRPNHead...")
        print(f"  len(cls_scores) = {len(cls_scores)}")
        print(f"  cls_scores[0].size() = {cls_scores[0].size()}")
        print(f"  len(bbox_preds) = {len(bbox_preds)}")
        print(f"  bbox_preds[0].size() = {bbox_preds[0].size()}")
        print(f"  len(gt_bboxes) = {len(gt_bboxes)}")
        print(f"  gt_labels = {gt_labels}")
        for ix, tensor in enumerate(gt_bboxes):
            print(f"    {ix}: {tensor.size()}")
        print(f"  len(img_metas) = {len(img_metas)}")
        print(f"  img_metas: {img_metas}")
        print(f"  pair_flags = {pair_flags}")
        '''

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]  # [[H_q, W_q]]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        # print(f"  featmap_sizes = {featmap_sizes}")  # [[16, 16]]

        device = cls_scores[0].device
        # get anchors and training targets
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)

        '''
        print(f"  len(anchor_list) = {len(anchor_list)}")
        for ix, lst in enumerate(anchor_list):
            print(f"    {ix}: {lst[0].size()}")
        print(f"  len(valid_flag_list) = {len(valid_flag_list)}")
        for ix, lst in enumerate(valid_flag_list):
            print(f"    {ix}: {lst[0].size()}; torch.all(lst[0]) = {torch.all(lst[0])}")
        '''

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1  # 1
        # print(f"  label_channels = {label_channels}")
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        # print(f"  Targets computed!")

        # labels_list:  [H_q x W_q x num_anchors] * num_imgs = torch.Size([4, 3840]) -> fg (0) / bg (1) labels for anchors
        # label_weights_list: torch.Size([4, 3840]) -> a weight of 1.0 is assigned to sampled anchors
        # bbox_targets_list: torch.Size([4, 3840, 4]) -> encoded deltas between positive anchors and ground truth boxes
        # bbox_weights_list: torch.Size([4, 3840, 4]) -> [1.0, 1.0, 1.0, 1.0] is assigned only to positive sampled anchors
        # num_total_pos: sums the number of postive anchors sampled for each image (16)
        # num_total_neg: sums the number of negative anchors sampled for each image (1008)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        '''
        print(f"  labels_list = {labels_list}")
        print(f"  labels_list[0].size()) = {labels_list[0].size()}")
        print(f"  label_weights_list = {label_weights_list}")
        print(f"  label_weights_list[0].size()) = {label_weights_list[0].size()}")
        print(f"  bbox_targets_list[0].size()) = {bbox_targets_list[0].size()}")
        print(f"  bbox_weights_list[0].size()) = {bbox_weights_list[0].size()}")
        '''

        # get the indices of negative pairs
        neg_idxes = [not f for f in pair_flags]  # [False, False, True, True]
        num_pos_from_neg_pairs = 0
        # all the gt_labels in negative pairs will be set to background
        for lvl in range(len(labels_list)):  # just 1 level
            num_pos_from_neg_pairs += (
                labels_list[lvl][neg_idxes] == 0).sum().item()
            labels_list[lvl][neg_idxes] = 1
            bbox_weights_list[lvl][neg_idxes] = 0
        # print(f"  self.sampling = {self.sampling}")
        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg  # 1024 (16 + 1008)
        else:
            num_total_samples = num_total_pos - num_pos_from_neg_pairs
        # print(f"  num_total_samples = {num_total_samples}")
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]  # [3840], just one level
        # print(f"  num_level_anchors = {num_level_anchors}")
        # concat all level anchors and flags to a single Tensor
        # [H_q x W_q x num_anchors, 4] x num_imgs = [torch.Size([3840, 4])] * 4
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        '''
        print(f"  len(concat_anchor_list) = {len(concat_anchor_list)}")
        for ix, tensor in enumerate(concat_anchor_list):
            print(f"    {ix}: {tensor.size()}")
        '''
        # convert targets by image to targets by feature level
        # [target_img0, target_img1] -> [target_level0, target_level1, ...]
        # [torch.Size([4, 3840, 4])], just one level
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        '''
        print(f"  len(all_anchor_list) = {len(all_anchor_list)}")
        for ix, tensor in enumerate(all_anchor_list):
            print(f"    {ix}: {tensor.size()}")
        '''

        # binary cross-entropy for losses_cls, l1 loss for losses_bbox
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feat: Tensor,
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[Tensor]:
        """Test function without test time augmentation.

        Args:
            query_feats (list[Tensor]): List of query features, each item with
                shape(N, C, H, W).
            support_feat (Tensor): Support features with shape (N, C, H, W).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results.
                Default: False.

        Returns:
            List[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        # fuse support and query features
        # print("Entering simple_test in AttentionRPNHead...")
        # print(f"  query_feats[0].size() = {query_feats[0].size()}")
        # print(f"  support_feat.size() = {support_feat.size()}")
        feats = self.aggregation_layer(
            query_feat=query_feats[0], support_feat=support_feat)
        # mmdet/models/dense_heads/dense_test_mixins.py
        proposal_list = self.simple_test_rpn(feats, query_img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, query_img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])

        return proposal_list
