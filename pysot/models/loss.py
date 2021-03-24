from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    #return F.nll_loss(pred, label, reduction='none')
    return F.nll_loss(pred, label)

def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    #loss_pos = torch.mean(loss_pos)
    #n_hard_neg = n_pos.nelement() * 3.0
    #hn_idxes = torch.topk(loss_neg, int(n_hard_neg))
    #loss_neg = torch.mean(loss_neg[hn_idxes[1]])
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def mask_loss_sml(masks_pred, masks_gt, mask_weight):
    """Mask R-CNN specific losses."""
    mask_weight = mask_weight.view(-1)
    pos = mask_weight.data.eq(1).nonzero().squeeze()
    if pos.nelement() == 0:
        return masks_pred.sum() * 0, masks_pred.sum() * 0, masks_pred.sum() * 0, masks_pred.sum() * 0

    masks_pred = torch.index_select(masks_pred, 0, pos)
    masks_gt = torch.index_select(masks_gt, 0, pos)

    #weight = (masks_gt > -1).float()
    #loss = F.binary_cross_entropy_with_logits(
    #    masks_pred.view(n_rois, -1), masks_gt, weight, size_average=False)
    #loss /= weight.sum()

    #print(masks_pred.size(), masks_gt.size())
    _, _, h, w = masks_pred.size()
    masks_pred = masks_pred.view(-1, h*w)
    masks_gt   = Variable(masks_gt.view(-1, h*w), requires_grad=False)
    loss = F.soft_margin_loss(masks_pred, masks_gt)
    iou_m, iou_5, iou_7 = iou_measure(masks_pred, masks_gt)
    return loss, iou_m, iou_5, iou_7


def mask_loss_bce(masks_pred, masks_gt, mask_weight, ohem=True):
    """Mask R-CNN specific losses."""
    mask_weight = mask_weight.view(-1)
    pos = mask_weight.data.eq(1).nonzero().squeeze()
    if pos.nelement() == 0:
        return masks_pred.sum() * 0, masks_pred.sum() * 0, masks_pred.sum() * 0, masks_pred.sum() * 0

    masks_pred = torch.index_select(masks_pred, 0, pos)
    masks_gt = torch.index_select(masks_gt, 0, pos)

    _, _, h, w = masks_pred.size()
    masks_pred = masks_pred.view(-1, h*w)
    masks_gt = Variable(masks_gt.view(-1, h*w), requires_grad=False)

    if ohem:
        top_k = 0.7
        loss = F.binary_cross_entropy_with_logits(masks_pred, masks_gt, reduction='none')
        loss = loss.view(-1)
        index = torch.topk(loss, int(top_k * loss.size()[0]))
        loss = torch.mean(loss[index[1]])
    else:
        loss = F.binary_cross_entropy_with_logits(masks_pred, masks_gt)

    iou_m, iou_5, iou_7 = iou_measure(masks_pred, masks_gt)
    return loss, iou_m, iou_5, iou_7


def iou_measure(pred, label):
    pred = pred.ge(0)
    mask_sum = pred.eq(1).add(label.eq(1))
    intxn = torch.sum(mask_sum == 2, dim=1).float()
    union = torch.sum(mask_sum > 0, dim=1).float()
    iou = intxn / (union + 1e-5)
    return torch.mean(iou), (torch.sum(iou > 0.5).float()/iou.shape[0]), (torch.sum(iou > 0.7).float()/iou.shape[0])


def det_loss_smooth_l1(bboxes_pred, bboxes_gt, bbox_weight):

    bbox_weight = bbox_weight.view(-1)
    pos = bbox_weight.data.eq(1).nonzero().squeeze()
    if pos.nelement() == 0:
        return bboxes_pred.sum() * 0

    bboxes_pred = torch.index_select(bboxes_pred, 0, pos)
    bboxes_gt = torch.index_select(bboxes_gt, 0, pos)
    bboxes_gt = Variable(bboxes_gt, requires_grad=False)

    bbox_loss = F.smooth_l1_loss(bboxes_pred, bboxes_gt)

    return bbox_loss





