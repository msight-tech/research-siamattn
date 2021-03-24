from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
import cv2


def proposal_target_layer(rpn_rois, gt_boxes):
    batch_size = len(gt_boxes)
    gt_boxes = gt_boxes.unsqueeze(1)

    if batch_size > 0:
        # multiple images
        rois_list = []
        labels_list = []
        bbox_targets_list = []

        for i in range(batch_size):
            batch_roi = rpn_rois[i].transpose(1, 0)  # (4, N) -> (N, 4)

            rois, labels, bbox_targets = single_proposal_target_layer(i, batch_roi, gt_boxes[i])
            # TODO assume each batch in order
            rois_list.append(rois.unsqueeze(0))
            labels_list.append(labels.unsqueeze(0))
            bbox_targets_list.append(bbox_targets.unsqueeze(0))

        return torch.cat(rois_list, dim=0), torch.cat(labels_list, dim=0), torch.cat(bbox_targets_list, dim=0)
    else:
        # single image
        rois, labels, bbox_targets = single_proposal_target_layer(0, rpn_rois, gt_boxes)
        return rois.unsqueeze(0), labels.unsqueeze(0), bbox_targets.unsqueeze(0)


def single_proposal_target_layer(batch, rpn_rois, gt_box):
    """
  Assign object detection proposals to ground-truth targets. Produces proposal
  classification labels and bounding-box regression targets.
  """
    # Proposal ROIs (n, x1, y1, x2, y2) coming from RPN

    all_rois = rpn_rois
    # Include ground-truth boxes in the set of candidate rois
    TRAIN_USE_GT = True  # TO BE WRITTEN IN CFG
    if TRAIN_USE_GT:
        all_rois = torch.cat((all_rois, gt_box), 0)

    rois_per_image = cfg.TRAIN.ROI_PER_IMG
    fg_rois_per_image = int(round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    labels, rois, bbox_target = _sample_rois(
                        all_rois,
                        gt_box,
                        fg_rois_per_image,
                        rois_per_image
                    )

    rois = rois.view(-1, 4)
    labels = labels.view(-1, 1)
    bbox_target = bbox_target.view(-1, 4)

    return rois, labels, bbox_target


def _sample_rois(all_rois, gt_box, fg_rois_per_image, rois_per_image):
    """
        Generate a random sample of RoIs comprising foreground and background examples.
    """

    def _rand_choice_idx(x, k, to_replace=False):
        idxs = np.random.choice(x.numel(), k, replace=to_replace)
        return x[torch.cuda.LongTensor(idxs)]

    fg_inds, bg_inds = bbox_assignment(
        all_rois, gt_box,
        cfg.TRAIN.FG_THRESH, cfg.TRAIN.BG_THRESH_HIGH,
        cfg.TRAIN.BG_THRESH_LOW)

    # balanced sample rois
    #fg_rois_per_image = min(fg_rois_per_image, fg_inds.numel())
    bg_rois_per_image = rois_per_image - fg_rois_per_image
    if fg_inds.numel() > fg_rois_per_image:
        fg_inds = _rand_choice_idx(fg_inds, fg_rois_per_image)
    else:
        # TODO random sample some area around the gtbox
        times = fg_rois_per_image // fg_inds.numel()
        mod = fg_rois_per_image % fg_inds.numel()
        fg_inds = torch.cat((fg_inds.repeat(times), _rand_choice_idx(fg_inds, mod)), dim=0)

    if bg_inds.numel() > bg_rois_per_image:
        bg_inds = _rand_choice_idx(bg_inds, bg_rois_per_image)
    else:
        times = bg_rois_per_image // bg_inds.numel()
        mod = bg_rois_per_image % bg_inds.numel()
        bg_inds = torch.cat((bg_inds.repeat(times), _rand_choice_idx(bg_inds, mod)), dim=0)

    # The indices that we're selecting (both fg and bg)
    keep_inds = torch.cat([fg_inds, bg_inds], 0)

    # Select sampled values from various arrays:
    labels = torch.ones(keep_inds.size(0)).cuda()
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds].contiguous()

    bbox_target = _compute_targets(rois, gt_box.expand(rois.shape[0], -1))

    return labels, rois, bbox_target

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    # Inputs are tensor

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - targets.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS)) /
                   targets.new(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return targets


def bbox_transform(ex_rois, gt_rois, clip=-1):
    # type: (Tensor, Tensor) -> Tensor
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0]
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1]
    ex_ctr_x = (ex_rois[:, 0] + ex_rois[:, 2]) * 0.5
    ex_ctr_y = (ex_rois[:, 1] + ex_rois[:, 3]) * 0.5

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0]
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1]
    gt_ctr_x = (gt_rois[:, 0] + gt_rois[:, 2]) * 0.5
    gt_ctr_y = (gt_rois[:, 1] + gt_rois[:, 3]) * 0.5

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = torch.log(gt_widths / ex_widths)
    targets_dh = torch.log(gt_heights / ex_heights)

    if clip > 0:
        targets_dw = torch.clamp_max(targets_dw, clip)
        targets_dh = torch.clamp_max(targets_dh, clip)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), 1)

    return targets


def bbox_assignment(boxes, gt_boxes, pos_iou_th, neg_iou_th_high, neg_iou_th_low):
    overlaps = bbox_overlaps(boxes, gt_boxes, mode="iou")
    max_overlaps, _ = overlaps.max(dim=1)

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = max_overlaps >= pos_iou_th
    fg_inds = fg_inds.nonzero().view(-1)

    #bg_inds = ((max_overlaps < neg_iou_th_high) & (max_overlaps >= neg_iou_th_low)).nonzero().view(-1)
    bg_inds = (max_overlaps < neg_iou_th_high).nonzero().view(-1)
    #if bg_inds.numel() == 0:
    #    bg_inds = (max_overlaps < neg_iou_th_high).nonzero().view(-1)

    return fg_inds, bg_inds


def bbox_overlaps(boxes, query_boxes, mode="iou"):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray or tensor or variable
    query_boxes: (K, 4) ndarray or tensor or variable
    Returns
    -------
    overlaps: (N, K) overlap between boxes and query_boxes
    """
    if isinstance(boxes, np.ndarray):
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = lambda x: x.numpy()  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    box_areas = (boxes[:, 2] - boxes[:, 0]) * \
                (boxes[:, 3] - boxes[:, 1])
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0]) * \
                  (query_boxes[:, 3] - query_boxes[:, 1])

    iw = (torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t()) -
          torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())).clamp(min=0)
    ih = (torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t()) -
          torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t())).clamp(min=0)
    iarea = iw * ih
    if mode == "iou":
        ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iarea
    elif mode == "iof":
        ua = box_areas.view(-1, 1)
    else:
        raise NotImplementedError

    overlaps = iarea / ua
    return out_fn(overlaps)


def build_mask_target(boxes, masks, label, mask_size):

    def mask_to_target(mask, roi, mask_size, is_binary=False):
        x1 = int(min(max(roi[0], 0), gt_mask.shape[1] - 1))
        y1 = int(min(max(roi[1], 0), gt_mask.shape[0] - 1))
        x2 = int(min(max(roi[2], 0), gt_mask.shape[1] - 1))
        y2 = int(min(max(roi[3], 0), gt_mask.shape[0] - 1))
        mask = mask[y1:y2 + 1, x1:x2 + 1]
        if is_binary is True:
            target = cv2.resize(mask, tuple([mask_size, mask_size]), interpolation=cv2.INTER_LINEAR)
            target = target > 128
        else:
            target = cv2.resize(mask, tuple([mask_size, mask_size]), interpolation=cv2.INTER_NEAREST)
        return target

    num_batches = boxes.shape[0]
    crop_masks = []
    select_rois = []
    for i in range(num_batches):
        batch_label = label[i]
        num_rois = boxes[i].shape[0]
        batch_crop_masks = []
        batch_select_rois = []
        for j in range(num_rois):
            if batch_label[j] <= 0:
                continue
            gt_mask = masks[i].squeeze(0)
            roi = boxes[i, j, :]
            gt_mask = mask_to_target(gt_mask, roi, mask_size)
            batch_crop_masks.append(gt_mask)
            batch_select_rois.append(roi)

        crop_masks.append(np.array(batch_crop_masks, np.float32))
        batch_select_rois = torch.from_numpy(np.array(batch_select_rois, np.float32)).cuda()
        select_rois.append(batch_select_rois)

    if len(crop_masks) > 0:
        mask_targets = torch.from_numpy(np.array(crop_masks, np.float32)).cuda()
        return mask_targets, select_rois
    else:
        return None, None


def _build_proposal_target(rois, gt_boxes):
    rois = rois.detach()
    rois, cls_ind, regression_target = proposal_target_layer(rois, gt_boxes)
    cls_ind = cls_ind.long().detach()
    return rois, cls_ind, regression_target


def _build_mask_target(rois, cls_ind, gt_masks):
    pred_boxes = rois
    # crop mask wrt roi
    mask_targets, select_roi_list = build_mask_target(
        pred_boxes.data.cpu().numpy(),
        gt_masks.data.cpu().numpy(),
        cls_ind.data.cpu().numpy(),
        cfg.MASK.MASK_OUTSIZE)

    # deal with empty target
    if mask_targets is not None:
        return mask_targets, select_roi_list
    else:
        return None, None


def _convert_loc_to_bbox(loc):

    def _generate_anchor(score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bbox(delta, anchor):
        anchor = torch.from_numpy(anchor).cuda()
        bsz = delta.size()[0]
        delta = delta.view(bsz, 4, -1)

        delta[:, 0, :] = delta[:, 0, :] * anchor[:, 2] + anchor[:, 0]
        delta[:, 1, :] = delta[:, 1, :] * anchor[:, 3] + anchor[:, 1]
        delta[:, 2, :] = torch.exp(delta[:, 2, :]) * anchor[:, 2]
        delta[:, 3, :] = torch.exp(delta[:, 3, :]) * anchor[:, 3]
        return delta

    '''
    def _convert_score(score):
        bsz = score.size()[0]
        score = score.view(bsz, 2, -1).permute(0, 2, 1)
        score = F.softmax(score, dim=2)[:, :, 1]
        return score
    '''

    def _bbox_clip(cx, cy, width, height, boundary):
        cx = torch.max(torch.tensor(0).float().cuda(),
                       other=torch.min(cx, other=torch.tensor(boundary[1]-1).float().cuda()))
        cy = torch.max(torch.tensor(0).float().cuda(),
                       other=torch.min(cy, other=torch.tensor(boundary[0]-1).float().cuda()))
        width = torch.max(torch.tensor(10).float().cuda(),
                          other=torch.min(width, other=torch.tensor(boundary[1]-1).float().cuda()))
        height = torch.max(torch.tensor(10).float().cuda(),
                           other=torch.min(height, other=torch.tensor(boundary[0]-1).float().cuda()))
        return cx, cy, width, height

    anchors = _generate_anchor(
        (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) // cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
    )

    pred_bbox = _convert_bbox(loc, anchors)
    center_pos = np.array([cfg.TRAIN.SEARCH_SIZE // 2, cfg.TRAIN.SEARCH_SIZE // 2])

    cx = pred_bbox[:, 0, :] + center_pos[0]
    cy = pred_bbox[:, 1, :] + center_pos[1]
    width = pred_bbox[:, 2, :]
    height = pred_bbox[:, 3, :]

    cx, cy, width, height = _bbox_clip(cx, cy, width,
                                       height, [cfg.TRAIN.SEARCH_SIZE, cfg.TRAIN.SEARCH_SIZE])

    bbox = torch.stack((cx - width / 2,
                        cy - height / 2,
                        cx + width / 2,
                        cy + height / 2), dim=1)

    return bbox
