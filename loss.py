# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.metrics import batch_probiou

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
            self,
            pred_dist: torch.Tensor,
            pred_bboxes: torch.Tensor,
            anchor_points: torch.Tensor,
            target_bboxes: torch.Tensor,
            target_scores: torch.Tensor,
            target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes with WIoU v3."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
            self,
            pred_dist: torch.Tensor,
            pred_bboxes: torch.Tensor,
            anchor_points: torch.Tensor,
            target_bboxes: torch.Tensor,
            target_scores: torch.Tensor,
            target_scores_sum: torch.Tensor,
            fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute WIoU v3 and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # 1. 计算原生的旋转高斯概率 IoU (ProbIoU)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])

        # ====================================================================
        # 🔥 WIoU v3 动态非单调聚焦机制 (带消融实验开关)
        # ====================================================================

        # 💡 这里就是你的控制台！设为 True 开启 WIoU，设为 False 关闭 WIoU
        USE_WIOU = False

        if USE_WIOU:
            with torch.no_grad():
                # 这里面保持你原来的 WIoU 计算代码完全不变
                dist = 1.0 - iou
                dist_mean = dist.mean()
                beta = dist / (dist_mean + 1e-7)


                alpha = float(os.environ.get('WIOU_ALPHA', '1.4'))
                delta = float(os.environ.get('WIOU_DELTA', '3.0'))
                clamp_min = float(os.environ.get('WIOU_MIN', '0.5'))
                clamp_max = float(os.environ.get('WIOU_MAX', '1.2'))

                gamma = beta / (delta * (alpha ** (beta - delta)))
                gamma = torch.clamp(gamma, min=clamp_min, max=clamp_max)

                if torch.rand(1).item() < 0.005:
                    print(f"\n[🚀 WIoU v3 探针] 动态流形观测:")
                    print(f"    ├─ 离群度 Beta 均值: {beta.mean().item():.4f}")
                    print(
                        f"    └─ 聚焦增益 Gamma: [Min: {gamma.min().item():.4f}, Mean: {gamma.mean().item():.4f}, Max: {gamma.max().item():.4f}]")
        else:
            # ⛔ 当关闭 WIoU 时，令 gamma 恒等于 1.0
            # 这样下方的 ((1.0 - iou) * gamma * weight) 就会退化为纯净的原生 Loss
            gamma = 1.0
        # ====================================================================

        # 下面这行代码不需要改动，它会自动根据上面的 gamma 值决定是否施加 WIoU 增益
        loss_iou = ((1.0 - iou) * gamma * weight).sum() / target_scores_sum

        # 3. DFL loss (保持原生分布离散化逻辑不变)
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
            self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl)

    @staticmethod
    def single_mask_loss(
            gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
            self,
            fg_mask: torch.Tensor,
            masks: torch.Tensor,
            target_gt_idx: torch.Tensor,
            target_bboxes: torch.Tensor,
            batch_idx: torch.Tensor,
            proto: torch.Tensor,
            pred_masks: torch.Tensor,
            imgsz: torch.Tensor,
            overlap: bool,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
            self,
            masks: torch.Tensor,
            target_gt_idx: torch.Tensor,
            keypoints: torch.Tensor,
            batch_idx: torch.Tensor,
            stride_tensor: torch.Tensor,
            target_bboxes: torch.Tensor,
            pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    # ====================================================================
    # 🌟 [工程创新点：序列化防崩机制]
    # 目的：解决 Dataloader 深度拷贝与弱引用(weakref)导致模型权重无法存入硬盘的 Bug
    # ====================================================================
    def __getstate__(self):
        """告诉 torch.save 在保存模型时，剔除掉 _trainer_ref 这个临时变量"""
        state = self.__dict__.copy()
        state.pop('_trainer_ref', None)
        return state

    # ====================================================================
    # 🚀 核心物理引擎：可微物理渲染 (Differentiable Physical Rendering)
    # ====================================================================
    def compute_physics_loss_tensorized(self, pred_boxes_scaled, pred_t, input_images, fg_mask, render_stride=8):

        if not hasattr(self, '_has_checked_tensors'):
            img_min = input_images.min().item()
            img_max = input_images.max().item()
            print(f"\n[🚨 物理张量安检] 输入图像极值: Min = {img_min:.4f}, Max = {img_max:.4f}")
            assert img_min >= 0.0 and img_max <= 1.0, f"❌ 图像张量必须归一化在 [0,1] 之间"
            self._has_checked_tensors = True

        B = pred_boxes_scaled.shape[0]
        device = pred_boxes_scaled.device
        H, W = input_images.shape[2] // render_stride, input_images.shape[3] // render_stride

        phys_loss = torch.tensor(0.0, device=device)
        total_base_phys = 0.0
        total_repulsion = 0.0
        total_t_std = 0.0
        total_t_val = 0.0
        valid_batches = 0

        for b in range(B):
            mask_b = fg_mask[b]
            if mask_b.sum() < 1:
                continue

            boxes = pred_boxes_scaled[b, mask_b].float()
            t_vals = pred_t[b, mask_b].squeeze(-1).float().sigmoid()
            num_pos = boxes.shape[0]

            # ⚡ [极致提速 1：随机高危焦点轮询]
            if num_pos > 15:
                centers = boxes[:, :2]
                dist_matrix = torch.cdist(centers, centers, p=2.0)
                neighbors = (dist_matrix < 50.0).sum(dim=1)
                dense_candidates = torch.nonzero(neighbors >= 2).squeeze(-1)

                if len(dense_candidates) > 0:
                    random_idx = torch.randint(0, len(dense_candidates), (1,)).item()
                    core_idx = dense_candidates[random_idx]
                else:
                    core_idx = neighbors.argmax()

                _, topk_indices = torch.topk(dist_matrix[core_idx], k=15, largest=False)
                boxes = boxes[topk_indices]
                t_vals = t_vals[topk_indices]
                num_pos = 15

            cx, cy, w, h, theta = boxes.unbind(-1)
            w = torch.clamp(torch.abs(w), min=1.0)
            h = torch.clamp(torch.abs(h), min=1.0)

            long_axis = torch.max(w, h)
            short_axis = torch.clamp(torch.min(w, h), min=1.0)
            aspect_ratio = long_axis / short_axis

            # 🌟 [创新点 2 进化：前置形状阻断 (Early Prior Gating)]
            valid_shape_mask = (aspect_ratio > 5.5) & (short_axis < 40.0)

            if valid_shape_mask.sum() < 1:
                continue  # 全部是弯曲废框，跳过本轮物理微雕

            cx, cy = cx[valid_shape_mask], cy[valid_shape_mask]
            w, h, theta = w[valid_shape_mask], h[valid_shape_mask], theta[valid_shape_mask]
            t_vals = t_vals[valid_shape_mask]
            long_axis, short_axis = long_axis[valid_shape_mask], short_axis[valid_shape_mask]
            num_pos = cx.shape[0]

            radius = (w + h) / 2
            min_x = torch.clamp(torch.min(cx - radius) / render_stride, min=0).long()
            max_x = torch.clamp(torch.max(cx + radius) / render_stride, max=W - 1).long()
            min_y = torch.clamp(torch.min(cy - radius) / render_stride, min=0).long()
            max_y = torch.clamp(torch.max(cy + radius) / render_stride, max=H - 1).long()

            if max_x <= min_x or max_y <= min_y:
                continue

            local_H, local_W = max_y - min_y + 1, max_x - min_x + 1
            if local_H * local_W > 40000:
                continue

            sy = torch.arange(min_y, max_y + 1, device=device, dtype=torch.float32) * render_stride
            sx = torch.arange(min_x, max_x + 1, device=device, dtype=torch.float32) * render_stride
            y, x = torch.meshgrid(sy, sx, indexing='ij') if torch.__version__ >= '1.10' else torch.meshgrid(sy, sx)
            local_grid = torch.stack((x, y), dim=-1).view(-1, 2)

            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            cos_t_exp = cos_t.unsqueeze(-1)
            sin_t_exp = sin_t.unsqueeze(-1)

            diff = local_grid.unsqueeze(0) - torch.stack((cx, cy), dim=-1).unsqueeze(1)
            dx, dy = diff[..., 0], diff[..., 1]
            u = dx * cos_t_exp + dy * sin_t_exp
            v = -dx * sin_t_exp + dy * cos_t_exp

            u_norm = u / (w.unsqueeze(-1) / 2.0 + 1e-5)
            v_norm = v / (h.unsqueeze(-1) / 2.0 + 1e-5)

            # ====================================================================
            # 🌟 [创新点 3 进化：全向对称胶囊体 SDF 场 (Omnidirectional SDF)]
            # 彻底解决 YOLO OBB 的 w/h 翻转引发的“巨型黑洞” Bug！
            # 动态识别长短轴：半径 R 恒定为短轴，中心骨架动态匹配长轴方向。
            # ====================================================================
            w_half = w.unsqueeze(-1) / 2.0
            h_half = h.unsqueeze(-1) / 2.0

            # 1. 真正的物理半径 R 永远是宽和高里面最小的那个
            R = torch.minimum(w_half, h_half) + 1e-5

            # 2. 分别计算 u 方向和 v 方向的中心骨架长度 (必有一个是 0)
            L_core_u = F.relu(w_half - h_half)
            L_core_v = F.relu(h_half - w_half)

            # 3. 计算绝对欧式距离 (自动适配旋转)
            u_dist = F.relu(torch.abs(u) - L_core_u)
            v_dist = F.relu(torch.abs(v) - L_core_v)
            d_sq = u_dist ** 2 + v_dist ** 2
            d_norm_sq = d_sq / (R ** 2)
            mahalanobis = (d_norm_sq * d_norm_sq) ** 2
            alpha = torch.exp(-0.5 * mahalanobis)

            # 🌟 [创新点 4：形态学物理掩码]
            EXPAND_PX = 4.0
            SHRINK_PX = 1.5

            outer_mask = (torch.abs(u) <= w_half + EXPAND_PX) & (torch.abs(v) <= h_half + EXPAND_PX)
            inner_mask = (torch.abs(u) <= w_half) & (torch.abs(v) <= h_half)
            ring_mask = outer_mask & ~inner_mask
            global_ring_mask = ring_mask.any(dim=0).view(local_H, local_W)

            shrunk_w_half = torch.clamp(w_half - SHRINK_PX, min=0.5)
            shrunk_h_half = torch.clamp(h_half - SHRINK_PX, min=0.5)
            shrunk_mask = (torch.abs(u) <= shrunk_w_half) & (torch.abs(v) <= shrunk_h_half)
            global_shrunk_mask = shrunk_mask.any(dim=0).view(local_H, local_W)

            orig_min_y, orig_max_y = min_y * render_stride, (max_y + 1) * render_stride
            orig_min_x, orig_max_x = min_x * render_stride, (max_x + 1) * render_stride
            local_crop_gray = input_images[b, :, orig_min_y:orig_max_y, orig_min_x:orig_max_x].mean(dim=0)
            local_real_img = F.interpolate(
                local_crop_gray.unsqueeze(0).unsqueeze(0),
                size=(local_H, local_W), mode='bilinear', align_corners=False
            ).squeeze()

            # ====================================================================
            # 🌟 [回归：2D 形态学池化动态背景场 (Morphological Pooling)]
            # 通过 Max-Pooling 膨胀亮色背景，吞噬暗色导线，应对显微镜极度不均的渐变光照。
            # ====================================================================
            # 1. 升维以匹配 PyTorch 的池化输入格式 (1, 1, H, W)
            img_4d = local_real_img.unsqueeze(0).unsqueeze(0)

            # 2. 膨胀操作 (Max Pooling)：用周围最亮的像素取代中心像素，强行擦除暗色导线
            # 这里的 kernel_size=15 意味着它能擦除宽度在 15 像素以内的暗线。如果你们的线更粗，可以改大 (如 21)。
            kernel_size = 15
            pad = kernel_size // 2
            dilated_bg = F.max_pool2d(img_4d, kernel_size=kernel_size, stride=1, padding=pad)

            # 3. 平滑操作 (Avg Pooling)：让膨胀后的背景光斑块更加柔和、自然过渡
            smooth_kernel = 9
            smooth_pad = smooth_kernel // 2
            dynamic_I0_map = F.avg_pool2d(dilated_bg, kernel_size=smooth_kernel, stride=1, padding=smooth_pad)

            # 4. 降维并截断断底
            dynamic_I0_map = dynamic_I0_map.squeeze().detach()
            dynamic_I0_map = torch.clamp(dynamic_I0_map, min=0.05)

            # --- 🚀 比尔-朗伯多重透射率渲染 ---
            t_vals_expanded = t_vals.unsqueeze(-1)
            attenuations = 1.0 - alpha * (1.0 - t_vals_expanded)
            expected_transmittance = attenuations.prod(dim=0).view(local_H, local_W) * dynamic_I0_map

            # ====================================================================
            # 核心计算域：只在有效区域结算所有物理代价
            # ====================================================================
            if global_shrunk_mask.sum() > 1.0:
                abs_diff = torch.abs(expected_transmittance - local_real_img)

                # ====================================================================
                # 🌟 [创新点 5 进化：Leaky 容差减震器 (Leaky Margin)]
                # 打破 ReLU 带来的“梯度绝对死区”！
                # 当误差 > 0.10 时，全额计算惩罚。
                # 当误差 < 0.10 时，不直接归零，而是保留 1% 的微弱引力，保持 T 值的梯度活性！
                # ====================================================================
                MARGIN = float(os.environ.get('DPR_MARGIN', '0.13'))
                excess_diff = torch.where(
                    abs_diff > MARGIN,
                    abs_diff - MARGIN,
                    abs_diff * 0.01  # 保留 1% 的生命特征梯度
                )

                # ====================================================================
                # 🌟 [实战破局进化版：基于真实先验的背景屏蔽 (GT-Prior Background Shielding)]
                # 彻底摒弃依赖网络预测的误差阈值！
                # 直接计算真实图像像素与“绝对背景光 (dynamic_I0_map)”的接近程度。
                # 完全免疫网络预测初期的剧烈波动，只将物理显微镜聚焦在真实的导线实体上！
                # ====================================================================

                # 1. 剥离预测，只看现实：计算真实图像中每个像素距离“纯净背景光”有多近
                # real_bg_diff 越小，说明这个像素越白，越确定它是背景
                real_bg_diff = torch.abs(dynamic_I0_map - local_real_img)

                # 2. 设定真实背景的判定范围 (即你说的“一定范围”)
                # 0.20 意味着：只要这个像素的亮度，距离最亮的背景光在 20% 以内，我们就判定它是背景
                BG_TOLERANCE = float(os.environ.get('DPR_BG_TOL', '0.06'))

                # 3. 生成基于真实内容的 Leaky 权重掩码
                # 条件反转：
                # - 如果它在原图上是背景 (diff < 0.20) -> 权重压低至 0.01。因为这里本来就没线，胖框盖在这里产生的阴影误差直接被屏蔽，消除震荡。
                # - 如果它在原图上是导线 (diff >= 0.20) -> 权重 1.0。这里有实实在在的黑线，哪怕网络错得再离谱，也要全额保留梯度，逼它对齐！
                leaky_weight = torch.where(real_bg_diff < BG_TOLERANCE,
                                           torch.full_like(excess_diff, 0.01),
                                           torch.ones_like(excess_diff))

                # 4. 施加终极软屏蔽
                excess_diff = excess_diff * leaky_weight

                # 📸 [物理切片导出器]
                import random
                if random.random() < 0.005:
                    try:
                        import cv2
                        import numpy as np
                        import time
                        os.makedirs('dpr_debug_renders', exist_ok=True)
                        real_img_np = (local_real_img.detach().cpu().numpy() * 255).astype(np.uint8)
                        render_img_np = (expected_transmittance.detach().cpu().numpy() * 255).astype(np.uint8)
                        error_img_np = (excess_diff.detach().cpu().numpy() * 255).astype(np.uint8)
                        error_heatmap = cv2.applyColorMap(error_img_np, cv2.COLORMAP_JET)
                        real_bgr = cv2.cvtColor(real_img_np, cv2.COLOR_GRAY2BGR)
                        render_bgr = cv2.cvtColor(render_img_np, cv2.COLOR_GRAY2BGR)

                        # ====================================================================
                        # 🌟 新增：在渲染图和真图上精准标注每根导线的预测透射率 T
                        # ====================================================================
                        for i in range(num_pos):
                            # 1. 坐标系转换：将全局网络坐标 (cx, cy) 映射到当前局部切片的像素坐标 (loc_x, loc_y)
                            loc_x = int(cx[i].item() / render_stride - min_x.item())
                            loc_y = int(cy[i].item() / render_stride - min_y.item())
                            t_val_i = t_vals[i].item()

                            text = f"T:{t_val_i:.2f}"

                            # 2. 画一个极小的准星 (红点)，精确定位导线框的中心
                            cv2.circle(render_bgr, (loc_x, loc_y), 1, (0, 0, 255), -1)
                            cv2.circle(real_bgr, (loc_x, loc_y), 1, (0, 0, 255), -1)

                            # 3. 打印 T 值文字
                            # 技巧：先画粗的黑色底层 (描边)，再画细的绿色表层。
                            # 这样无论导线是纯黑还是纯白，文字都清晰可见！
                            # -> 画在物理渲染图上
                            cv2.putText(render_bgr, text, (loc_x + 2, loc_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 0, 0), 2)
                            cv2.putText(render_bgr, text, (loc_x + 2, loc_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 255, 0), 1)

                            # -> 顺便也画在真实切片图上，方便你左右对比
                            cv2.putText(real_bgr, text, (loc_x + 2, loc_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 0, 0), 2)
                            cv2.putText(real_bgr, text, (loc_x + 2, loc_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 255, 0), 1)
                        # ====================================================================

                        combined = np.hstack((real_bgr, render_bgr, error_heatmap))
                        cv2.imwrite(f'dpr_debug_renders/render_{int(time.time() * 1000)}_{b}.png', combined)
                    except Exception as e:
                        print(f"\n[⚠️ 图像保存崩溃] 详细信息: {e}")

                # 🌟 [自适应高斯空间权重]
                spatial_weight = torch.exp(-1.0 * ((u_norm ** 2) + (v_norm ** 2)))
                global_spatial_weight = spatial_weight.max(dim=0)[0].view(local_H, local_W)
                weighted_excess_diff = excess_diff * global_spatial_weight

                valid_excess_diff = weighted_excess_diff[global_shrunk_mask]
                base_phys_loss = valid_excess_diff.sum() / global_shrunk_mask.sum()

                # # 🌟 [创新点 6：降维拓扑斥力网络]
                # repulsion_penalty = torch.tensor(0.0, device=device)
                # if num_pos > 1:
                #     intersect_area = torch.matmul(alpha, alpha.transpose(0, 1))
                #
                #     delta_theta = torch.abs(theta.unsqueeze(0) - theta.unsqueeze(1))
                #     delta_theta = torch.min(delta_theta, torch.pi - delta_theta)
                #     angle_gate = (delta_theta < 0.1745).float()
                #
                #     L_grid = long_axis / render_stride
                #     avg_L_ij = (L_grid.unsqueeze(0) + L_grid.unsqueeze(1)) / 2.0
                #     triu_mask = torch.triu(torch.ones(num_pos, num_pos, device=device), diagonal=1)
                #
                #     centers = torch.stack((cx, cy), dim=-1)
                #     dist_matrix = torch.cdist(centers, centers, p=2.0)
                #     avg_short_axis = (short_axis.unsqueeze(0) + short_axis.unsqueeze(1)) / 2.0
                #     duplicate_gate = (dist_matrix > (avg_short_axis * 0.3)).float()
                #
                #     pair_repulsion = intersect_area * triu_mask * angle_gate * duplicate_gate / (avg_L_ij + 1e-5)
                #     repulsion_penalty = pair_repulsion.sum() * 0.002
                # ====================================================================
                # 🌟 [创新点 6 进化：奥卡姆剃刀 (已退役)]
                # 因为加入了同源去重物理门控，进入物理引擎的必然是独立的真实实体。
                # 彻底删除降维拓扑斥力网络和 duplicate_gate，防止误伤真实的密集平行导线！
                # ====================================================================
                repulsion_penalty = torch.tensor(0.0, device=device)

                # 🚫 [反作弊机制：材质惩罚 (放宽至0.8红线)]
                # 因为加入了 Leaky 截断，网络不再极度恐慌，T值红线可以放到 0.8
                T_REDLINE = float(os.environ.get('DPR_T_REDLINE', '0.69'))
                t_penalty = torch.relu(t_vals - T_REDLINE).pow(2).mean() * 5.0

                # ====================================================================
                # ⚡ [反作弊机制 2：离散度强制激活 (Variance Defibrillator)]
                # 网络想偷懒全输出 0.7291？我们直接对“方差(Variance)”进行考核！
                # 如果这批框的 T 值方差小于 0.001 (约等于 Std < 0.03)，施加高额惩罚！
                # 这根鞭子会强迫卷积核去读取图像特征，产生差异化的预测！
                # ====================================================================
                t_var_penalty = torch.tensor(0.0, device=device)
                if t_vals.numel() > 1:
                    t_var = t_vals.var()  # 计算方差
                    t_var_penalty = F.relu(0.001 - t_var) * 20.0  # 惩罚权重给到 20.0，电击力度拉满！

                # 🔥 物理总误差合流 (把鞭子加进去)
                phys_loss = phys_loss + (base_phys_loss + repulsion_penalty + t_penalty + t_var_penalty)

                total_base_phys += base_phys_loss.item()
                total_repulsion += repulsion_penalty.item()
                total_t_val += t_vals.mean().item()
                total_t_std += t_vals.std().item() if t_vals.numel() > 1 else 0.0

                valid_batches += 1

        safe_div = valid_batches + 1e-6
        return phys_loss / safe_div, total_base_phys / safe_div, total_repulsion / safe_div, total_t_val / safe_div, total_t_std / safe_div

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(3, device=self.device)

        if isinstance(preds[0], list):
            feats, pred_angle, pred_t = preds
        else:
            feats, pred_angle, pred_t = preds[1]

        batch_size = pred_angle.shape[0]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()
        pred_t = pred_t.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ OBB dataset incorrectly formatted...") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(  # <--- 改为 target_gt_idx
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # 提前声明独立变量接住物理损失
        final_dpr_loss = torch.tensor(0.0, device=self.device)

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

            try:
                if not hasattr(self, '_trainer_ref'):
                    import inspect
                    import weakref
                    self._trainer_ref = None
                    for frame_record in inspect.stack():
                        caller = frame_record[0].f_locals.get('self', None)
                        if caller is not None and hasattr(caller, 'epoch') and hasattr(caller, 'epochs'):
                            self._trainer_ref = weakref.ref(caller)
                            break

                trainer = self._trainer_ref() if self._trainer_ref is not None else None
                current_epoch = trainer.epoch if trainer is not None else 0

                WARMUP_EPOCHS = 0

                if current_epoch >= WARMUP_EPOCHS:
                    scaled_pred_bboxes = pred_bboxes.clone()
                    scaled_pred_bboxes[..., :4] *= stride_tensor
                    input_images = batch['img']

                    # ====================================================================
                    # 🌟 [创新点 7 终极进化：同源去重物理门控 (Physics NMS)]
                    # 解决多框重叠导致渲染极度深色的致命物理悖论！
                    # ====================================================================
                    max_alignment_scores = target_scores.amax(dim=-1)
                    base_physics_mask = fg_mask & (max_alignment_scores > 0.3)

                    physics_mask = torch.zeros_like(base_physics_mask)
                    for b in range(batch_size):
                        valid_idx = torch.nonzero(base_physics_mask[b]).squeeze(-1)
                        if len(valid_idx) == 0:
                            continue

                        # 取出这些有效框分别隶属于哪一根真实的标签导线 (GT)
                        gt_ids = target_gt_idx[b, valid_idx]
                        scores = max_alignment_scores[b, valid_idx]

                        # 遍历当前图中的每一根真实导线
                        unique_gts = gt_ids.unique()
                        for gt in unique_gts:
                            idx_in_valid = torch.nonzero(gt_ids == gt).squeeze(-1)
                            # 在所有预测这根导线的框中，找出对齐得分最高的那一个！
                            best_local_idx = scores[idx_in_valid].argmax()
                            best_idx = idx_in_valid[best_local_idx]

                            # 核心斩断：只允许这个最强、最贴合的代表框进入物理引擎渲染！
                            physics_mask[b, valid_idx[best_idx]] = True

                    if physics_mask.sum() == 0:
                        l_phys = torch.tensor(0.0, device=self.device)
                        # ==========================================
                        # 🚑 修复 3：补上空掩码时的探针默认值
                        # ==========================================
                        base_l, rep_l, t_mean, t_std = 0.0, 0.0, 0.0, 0.0
                        dynamic_stride = 1
                    else:
                        current_imgsz = input_images.shape[-1]
                        dynamic_stride = 2 if current_imgsz > 700 else 1
                        # ==========================================
                        # 🚑 修复 4：改为接收 4 个返回值！
                        # ==========================================
                        l_phys, base_l, rep_l, t_mean, t_std= self.compute_physics_loss_tensorized(
                            scaled_pred_bboxes, pred_t, input_images, physics_mask,
                            render_stride=dynamic_stride
                        )

                    # 🌟 [配套创新 7.1：解耦的外挂式物理威力]
                    # MAX_LAMBDA拉升至 15.0，摆脱原生 box_gain 的牵制，实现绝对物理主权。
                    MAX_LAMBDA = float(os.environ.get('DPR_MAX_LAMBDA', '6.0'))
                    progress = min(1.0, (current_epoch - WARMUP_EPOCHS + 1) / 20.0)
                    lambda_phys = MAX_LAMBDA * progress

                    # ====================================================================
                    # 🌟 [动态梯度天花板放宽]
                    # 由于同源去重门控的加入，总误差现在极其纯净健康。
                    # 将截断阈值提升至 0.5，让光学对齐梯度完美传递，微雕边界！
                    # ====================================================================
                    safe_l_phys = torch.where(
                        l_phys < 0.5,
                        l_phys,
                        0.5 + 0.1 * torch.log(1.0 + (l_phys - 0.5) / 0.1)
                    )

                    final_dpr_loss = safe_l_phys * lambda_phys

                    # 1. 提高打印概率到 2% (约每个 epoch 打印 5 次)
                    import random
                    if random.random() < 0.02 and physics_mask.sum() > 0:
                        # 打印极其详尽的物理状态
                        print(f"\n[🔬 DPR-Loss 探针] Epoch: {current_epoch} | 步长: {dynamic_stride}")
                        print(f"    ├─ 🔍 总核误差: {l_phys.item():.4f} (独立威力: {final_dpr_loss.item():.4f})")
                        print(f"    ├─ 💡 光学透射误差: {base_l:.4f} | 💥 几何斥力: {rep_l:.4f}")
                        print(f"    └─ 👁️ 当前网络预测T值均值: {t_mean:.4f}| 离散度(Std) {t_std:.4f}")

            except Exception as e:
                # 2. 撕开静默防护网：如果发生报错，以黄色警告打印出来，但不终止训练
                import random
                if random.random() < 0.1:  # 限制报错打印频率，防止刷屏
                    print(f"\n[⚠️ DPR-Loss 运算警告] Epoch: {current_epoch} | 错误信息: {e}")
        else:
            final_dpr_loss += (pred_angle * 0).sum() + (pred_t * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        # 核心优化：物理引擎作为独立外挂融合
        loss[0] += final_dpr_loss

        return loss * batch_size, loss.detach()

    def bbox_decode(
            self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1) for xi in feats]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
