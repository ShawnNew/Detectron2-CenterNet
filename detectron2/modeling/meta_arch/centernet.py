# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.structures import Boxes, ImageList, Instances
from ..postprocessing import detector_postprocess
from ..backbone import DLAUp, IDAUp, build_backbone
from .build import META_ARCH_REGISTRY
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.data.detection_utils import gen_heatmap

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

        # fmt: off
        # backbone - DLA
        head_conv                     = cfg.MODEL.CENTERNET.HEAD_CONV
        final_kernel                  = cfg.MODEL.CENTERNET.FINAL_KERNEL
        # heads
        self.heads                    = cfg.MODEL.CENTERNET.TASK
        # loss
        self.hm_weight                = cfg.MODEL.CENTERNET.HM_WEIGHT
        self.wh_weight                = cfg.MODEL.CENTERNET.WH_WEIGHT
        self.off_weight               = cfg.MODEL.CENTERNET.OFF_WEIGHT
        # inference
        self.score_threshold          = cfg.MODEL.CENTERNET.SCORE_THRESH_TEST
        self.topk_candidates          = cfg.MODEL.CENTERNET.TOPK_CANDIDATES_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        # fmt: on

        # other
        given_dataset = cfg.DATASETS.TRAIN[0]
        DatasetCatalog.get(given_dataset)
        self.meta = MetadataCatalog.get(given_dataset)
        self.num_classes = len(self.meta.thing_classes) # modify num_classes by meta data of given dataset
        self.heads["HM"] = self.num_classes
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        # self modules
        self.backbone = build_backbone(cfg)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(self.backbone.channels[self.backbone.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=final_kernel, stride=1,
                              padding=final_kernel // 2, bias=True))
                if 'hm' in head.lower():
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(self.backbone.channels[self.backbone.first_level], classes,
                               kernel_size=final_kernel, stride=1,
                               padding=final_kernel // 2, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head.lower(), fc)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images, instances = self.preprocess_image(batched_inputs)
        y = self.backbone(images.tensor)

        z = {}
        for head in self.heads:
            if head.lower() == 'hm':
                head_output = torch.clamp(
                    self.__getattr__(head.lower())(y[-1]).sigmoid_(),
                    min=1e-4,
                    max=1-1e-4
                )
                z[head.lower()] = head_output
            else:
                z[head.lower()] = self.__getattr__(head.lower())(y[-1])

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            losses = self.losses(z, instances)
            return losses
        else:
            results = self.inference(z, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                original_height = input_per_image.get("height", image_size[0])
                original_width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, original_height, original_width)
                processed_results.append({"instances": r})
            return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) / 255. for x in batched_inputs]

        if self.training:
            instances = [x["instances"].to(self.device) for x in batched_inputs]
        else: instances = []

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        input_shape = np.array(images.tensor.shape[2:])
        output_shape = input_shape // self.backbone.down_ratio
        if not self.training: return images, []
        instances = [gen_heatmap(x, output_shape, self.meta) for x in instances]
        return images, instances

    def losses(self, outputs, instances):
        # loss function for heatmap and wh+reg.
        _crit = FocalLoss()
        reg_crit = RegL1Loss()

        # get ground truth from batched_inputs
        gt_hm = torch.cat([x["hm"].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_reg_mask = torch.cat([x["reg_mask"].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_ind = torch.cat([x["ind"].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_wh = torch.cat([x["wh"].unsqueeze(0).to(self.device) for x in instances], dim=0)
        gt_reg = torch.cat([x["reg"].unsqueeze(0).to(self.device) for x in instances], dim=0)

        # sigmoid for heatmap
        hm_loss = _crit(outputs['hm'], gt_hm)
        wh_loss = reg_crit(outputs['wh'], gt_reg_mask, gt_ind, gt_wh)
        off_loss = reg_crit(outputs['reg'], gt_reg_mask, gt_ind, gt_reg)

        return {
            "hm_loss": hm_loss * self.hm_weight,
            "wh_loss": wh_loss * self.wh_weight,
            "off_loss": off_loss * self.off_weight
        }

    def inference(self, outputs, image_sizes):
        """
        Inference on outputs.
        :param outputs: outputs of centernet
        :param image_sizes: 
        :return: 
        """"""results (List[Instances]): a list of #images elements.
        """
        results = []
        for img_idx, image_size in enumerate(image_sizes):
            output = {
                'hm': outputs['hm'][img_idx].unsqueeze(0),
                'wh': outputs['wh'][img_idx].unsqueeze(0),
                'reg': outputs['reg'][img_idx].unsqueeze(0)
            }
            results_per_image = self.inference_single_image(
                output, tuple(image_size)
            )
            results.append(results_per_image)
        return results

    def inference_single_image(self, output, image_size):
        """
        Inference on one image.
        :param output:
        :param image_size:
        :return:
        """

        # decode centernet output and keep top k top scoring indices.
        boxes_all, scores_all, class_idxs_all = ctdet_decode(output['hm'],
                                                             output['wh'],
                                                             reg=output['reg'],
                                                             down_ratio=self.down_ratio,
                                                             K=self.topk_candidates)

        # take max number of detections per image
        max_num_detections_per_image = min(self.max_detections_per_image, self.topk_candidates)
        scores_all = scores_all[:max_num_detections_per_image]
        boxes_all = boxes_all[:max_num_detections_per_image]
        class_idxs_all = class_idxs_all[:max_num_detections_per_image]

        # filter out by threshold
        keep_idxs = scores_all > self.score_threshold
        scores_all = scores_all[keep_idxs]
        boxes_all = boxes_all[keep_idxs]
        class_idxs_all = class_idxs_all[keep_idxs]
        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all)
        result.scores = scores_all
        result.pred_classes = class_idxs_all
        return result



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
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
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

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def ctdet_decode(heat, wh, reg=None, down_ratio=1, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)

    clses = clses.view(K)
    scores = scores.view(K)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2) * down_ratio
    bboxes = bboxes.view(K, 4)

    return bboxes, scores, clses
