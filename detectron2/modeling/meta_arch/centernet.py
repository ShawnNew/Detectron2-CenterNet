# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from ..anchor_generator import build_anchor_generator
from ..backbone import build_backbone
from ..box_regression import Box2BoxTransform
from ..matcher import Matcher
from ..postprocessing import detector_postprocess
import fvcore.nn.weight_init as weight_init
from ..backbone import dla34, DLAUp, IDAUp
from .build import META_ARCH_REGISTRY

__all__ = [
    "CenterNet",
]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

@META_ARCH_REGISTRY.register()
class CenterNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        down_ratio       = cfg.MODEL.CENTERNET.DOWN_RATIO
        head_conv        = cfg.MODEL.CENTERNET.HEAD_CONV
        final_kernel     = cfg.MODEL.CENTERNET.FINAL_KERNEL
        self.num_classes = cfg.MODEL.CENTERNET.NUM_CLASSES
        self.last_level  = cfg.MODEL.CENTERNET.LAST_LEVEL
        self.heads       = cfg.MODEL.CENTERNET.TASK
        self.hm_weight   = cfg.MODEL.CENTERNET.HM_WEIGHT
        self.wh_weight   = cfg.MODEL.CENTERNET.WH_WEIGHT
        self.off_weight  = cfg.MODEL.CENTERNET.OFF_WEIGHT

        self.heads["HM"] = self.num_classes
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.base = dla34(pretrained=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales)

        out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)])

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        x = self.base(images.tensor)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            # gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_hm        = [x["hm"].unsqueeze(0).to(self.device) for x in batched_inputs]
            gt_reg_mask  = [x["reg_mask"].unsqueeze(0).to(self.device) for x in batched_inputs]
            gt_ind       = [x["ind"].unsqueeze(0).to(self.device) for x in batched_inputs]
            gt_wh        = [x["wh"].unsqueeze(0).to(self.device) for x in batched_inputs]
            gt_reg       = [x["reg"].unsqueeze(0).to(self.device) for x in batched_inputs]
            gt_hm        = torch.cat(gt_hm, dim=0)
            gt_reg_mask  = torch.cat(gt_reg_mask, dim=0)
            gt_ind       = torch.cat(gt_ind, dim=0)
            gt_wh        = torch.cat(gt_wh, dim=0)
            gt_reg       = torch.cat(gt_reg, dim=0)
            # gt_labels, gt_boxes = self.transform_anchors(gt_instances)

            losses = self.losses(z, gt_hm, gt_reg_mask, gt_ind, gt_wh, gt_reg)

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         results = self.inference(
            #             anchors, pred_logits, pred_anchor_deltas, images.image_sizes
            #         )
            #         self.visualize_training(batched_inputs, results)

            return losses
        # else:
        #     results = self.inference(anchors, pred_logits, pred_anchor_deltas, images.image_sizes)
        #     processed_results = []
        #     for results_per_image, input_per_image, image_size in zip(
        #         results, batched_inputs, images.image_sizes
        #     ):
        #         height = input_per_image.get("height", image_size[0])
        #         width = input_per_image.get("width", image_size[1])
        #         r = detector_postprocess(results_per_image, height, width)
        #         processed_results.append({"instances": r})
        #     return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, 32)
        return images

    def losses(self, outputs, gt_hm, gt_reg_mask, gt_ind, gt_wh, gt_reg):
        _crit = FocalLoss()
        reg_crit = RegL1Loss()
        outputs['HM'] = torch.clamp(outputs['HM'].sigmoid_(), min=1e-4, max=1-1e-4)
        hm_loss = _crit(outputs['HM'], gt_hm)
        wh_loss = reg_crit(outputs['WH'], gt_reg_mask, gt_ind, gt_wh)
        off_loss = reg_crit(outputs['REG'], gt_reg_mask, gt_ind, gt_reg)

        return {
            "hm_loss": hm_loss * self.hm_weight,
            "wh_loss": wh_loss * self.wh_weight,
            "off_loss": off_loss * self.off_weight
        }

    @torch.no_grad()
    def transform_anchors(self, gt_instances):
        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            gt_labels.append(gt_per_image.gt_classes)
            matched_gt_boxes.append(gt_per_image.gt_boxes)

        return gt_labels, matched_gt_boxes



class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat
