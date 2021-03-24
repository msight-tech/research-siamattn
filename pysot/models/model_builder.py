from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

import numpy as np
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, mask_loss_bce, det_loss_smooth_l1
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.neck.feature_fusion import FeatureFusionNeck
from pysot.models.neck.enhance import FeatureEnhance

from pysot.models.head.mask import FusedSemanticHead
from pysot.models.head.detection import FCx2DetHead
from pysot.utils.mask_target_builder import _build_proposal_target, _build_mask_target, _convert_loc_to_bbox


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.feature_enhance = FeatureEnhance(in_channels=256, out_channels=256)
            self.feature_fusion = FeatureFusionNeck(num_ins=5, fusion_level=1,
                                                    in_channels=[64, 256, 256, 256, 256], conv_out_channels=256)
            self.mask_head = FusedSemanticHead(pooling_func=None,
                                               num_convs=4, in_channels=256,
                                               upsample_ratio=(cfg.MASK.MASK_OUTSIZE // cfg.TRAIN.ROIPOOL_OUTSIZE))
            self.bbox_head = FCx2DetHead(pooling_func=None,
                                         in_channels=256 * (cfg.TRAIN.ROIPOOL_OUTSIZE // 4)**2)

    def template(self, z):
        with torch.no_grad():
            zf = self.backbone(z)
            if cfg.ADJUST.ADJUST:
                zf[2:] = self.neck(zf[2:])
            self.zf = zf

    def track(self, x):
        with torch.no_grad():
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                xf[2:] = self.neck(xf[2:])

            zf, xf[2:] = self.feature_enhance(self.zf[2:], xf[2:])
            cls, loc = self.rpn_head(zf, xf[2:])
            enhanced_zf = self.zf[:2] + zf
            if cfg.MASK.MASK:
                self.b_fused_features, self.m_fused_features = self.feature_fusion(enhanced_zf, xf)
            return {
                'cls': cls,
                'loc': loc
            }

    def mask_refine(self, roi):
        with torch.no_grad():
            mask_pred = self.mask_head(self.m_fused_features, roi)

        return mask_pred

    def bbox_refine(self, roi):
        with torch.no_grad():
            bbox_pred = self.bbox_head(self.b_fused_features, roi)

        return bbox_pred

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        search_mask = data['search_mask'].cuda()
        mask_weight = data['mask_weight'].cuda()
        bbox_weight = data['bbox_weight'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        gt_bboxes = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf[2:] = self.neck(zf[2:])
            xf[2:] = self.neck(xf[2:])

        zf[2:], xf[2:] = self.feature_enhance(zf[2:], xf[2:])
        cls, loc = self.rpn_head(zf[2:], xf[2:])

        # get loss
        cls_sm = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls_sm, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # Convert loc coordinate to (x1,y1,x2,y2)
            loc = loc.detach()
            bbox = _convert_loc_to_bbox(loc)
            rois, cls_ind, regression_target = _build_proposal_target(bbox, gt_bboxes)
            mask_targets, select_roi_list = _build_mask_target(rois, cls_ind, search_mask)

            # for deformable roi pooling
            batch_inds = torch.from_numpy(np.arange(
                cfg.TRAIN.BATCH_SIZE).repeat(cfg.TRAIN.ROI_PER_IMG).reshape(cfg.TRAIN.BATCH_SIZE*cfg.TRAIN.ROI_PER_IMG, 1)).cuda().float()
            rois = torch.cat((batch_inds, torch.stack(select_roi_list).view(-1, 4)), dim=1)

            b_fused_features, m_fused_features = self.feature_fusion(zf, xf)
            bbox_pred = self.bbox_head(b_fused_features, rois)
            bbox_pred = bbox_pred.view_as(regression_target)

            mask_pred = self.mask_head(m_fused_features, rois)
            mask_pred = mask_pred.view_as(mask_targets)

            # compute loss
            mask_loss, iou_m, iou_5, iou_7 = mask_loss_bce(mask_pred, mask_targets, mask_weight)
            bbox_loss = det_loss_smooth_l1(bbox_pred, regression_target, bbox_weight)

            # outputs['bbox'] = select_roi_list
            # outputs['crop_feature'] = crop_feature
            outputs['mask_labels'] = mask_targets
            outputs['mask_preds'] = mask_pred
            # mask_loss = None
            outputs['total_loss'] += (cfg.TRAIN.MASK_WEIGHT * mask_loss + cfg.TRAIN.BBOX_WEIGHT * bbox_loss)
            outputs['bbox_loss'] = bbox_loss
            outputs['mask_loss'] = mask_loss
            outputs['iou_m'] = iou_m
            outputs['iou_5'] = iou_5
            outputs['iou_7'] = iou_7
        return outputs

