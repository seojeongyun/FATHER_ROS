#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigner.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigner.atss_assigner import ATSSAssigner
from yolov6.assigner.tal_assigner import TaskAlignedAssigner
from yolov6.utils.RepulsionLoss import repulsion_loss


class ComputeLoss:
    '''Loss computation func.'''

    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=0,
                 use_dfl=True,
                 reg_max=16,
                 iou_type='giou',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                     'cwd': 10.0,
                     'repgt': 0.5,
                     'repbox': 5.0,
                     'landmark': 0.5},
                 distill_feat=False,
                 distill_weight={
                     'class': 1.0,
                     'dfl': 1.0,
                 },
                 device=torch.device("cpu")
                 ):

        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight
        self.landmarks_loss = LandmarksLoss(1.0)
        self.distill_feat = distill_feat
        self.distill_weight = distill_weight
        self.device = device

    def __call__(
            self,
            outputs,
            t_outputs,
            s_featmaps,
            t_featmaps,
            targets,
            epoch_num,
            max_epoch,
            temperature,
            step_num
    ):

        feats, pred_scores, pred_distri = outputs
        t_feats, t_pred_scores, t_pred_distri = t_outputs[0], t_outputs[-2], t_outputs[-1]
        anchors, anchor_points, n_anchors_list, stride_tensor = \
            generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                             device=feats[0].device)
        t_anchors, t_anchor_points, t_n_anchors_list, t_stride_tensor = \
            generate_anchors(t_feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                             device=feats[0].device)

        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1, 4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # targets
        targets, gt_ldmks = self.preprocess(targets, batch_size, gt_bboxes_scale)
        # Shape of targets: (batch_size, max_len, 5)
        gt_labels = targets[:, :, :1]  # class number
        gt_bboxes = targets[:, :, 1:5]  # xyxy, shape: (batch size, max_len, 4)
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()  # Shape: (batch size, max_len, 1)
        # Think that in self.preprocess(), several dummy lists [-1, 0, 0, 0, 0] are added.
        # The dummy lists become zero, which indicates which list has meaningful x, y, w, and h.

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes, pred_ldmks = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy
        t_anchor_points_s = t_anchor_points / t_stride_tensor
        t_pred_bboxes, t_pred_ldmks = self.bbox_decode(t_anchor_points_s, t_pred_distri)  # xyxy

        try:
            target_labels, target_bboxes, target_ldmks, target_scores, fg_mask = \
                self.formal_assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    gt_labels,
                    gt_bboxes,
                    gt_ldmks,
                    mask_gt)
        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")

            _pred_scores = pred_scores.detach().cpu().float()
            _pred_bboxes = pred_bboxes.detach().cpu().float()
            _anchor_points = anchor_points.cpu().float()
            _gt_labels = gt_labels.cpu().float()
            _gt_bboxes = gt_bboxes.cpu().float()
            _gt_ldmks = gt_ldmks.cpu().float()
            _mask_gt = mask_gt.cpu().float()
            _stride_tensor = stride_tensor.cpu().float()

            target_labels, target_bboxes, target_ldmks, target_scores, fg_mask = \
                self.formal_assigner(
                    _pred_scores,
                    _pred_bboxes * _stride_tensor,
                    _anchor_points,
                    _gt_labels,
                    _gt_bboxes,
                    _gt_ldmks,
                    _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_ldmks = target_ldmks.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()

        # Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor
        target_ldmks /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum

        # bbox loss
        loss_iou, loss_dfl, d_loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, t_pred_distri, t_pred_bboxes,
                                                        temperature, anchor_points_s,
                                                        target_bboxes, target_scores, target_scores_sum, fg_mask)

        logits_student = pred_scores
        logits_teacher = t_pred_scores
        distill_num_classes = self.num_classes
        d_loss_cls = self.distill_loss_cls(logits_student, logits_teacher, distill_num_classes, temperature)
        if self.distill_feat:
            d_loss_cw = self.distill_loss_cw(s_featmaps, t_featmaps)
        else:
            d_loss_cw = torch.tensor(0.).to(feats[0].device)
        import math
        distill_weightdecay = ((1 - math.cos(epoch_num * math.pi / max_epoch)) / 2) * (0.01 - 1) + 1
        d_loss_dfl *= distill_weightdecay
        d_loss_cls *= distill_weightdecay
        d_loss_cw *= distill_weightdecay
        loss_cls_all = loss_cls + d_loss_cls * self.distill_weight['class']
        loss_dfl_all = loss_dfl + d_loss_dfl * self.distill_weight['dfl']

        # repulsion loss
        loss_repGT, loss_repBox = repulsion_loss(
            pred_bboxes, target_bboxes, fg_mask, sigma_repgt=0.9, sigma_repbox=0, pnms=0, gtnms=0)

        # Landmarks Loss
        loss_landmark = self.landmarks_loss(pred_ldmks, target_ldmks, fg_mask)

        loss = self.loss_weight['class'] * loss_cls_all + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl_all + \
               self.loss_weight['repgt'] * loss_repGT + self.loss_weight['repbox'] * loss_repBox + \
               self.loss_weight['cwd'] * d_loss_cw + \
               self.loss_weight['landmark'] * loss_landmark

        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                       (self.loss_weight['dfl'] * loss_dfl_all).unsqueeze(0),
                       (self.loss_weight['class'] * loss_cls_all).unsqueeze(0),
                       (self.loss_weight['repgt'] * loss_repGT).unsqueeze(0),
                       (self.loss_weight['repbox'] * loss_repBox).unsqueeze(0),
                       (self.loss_weight['landmark'] * loss_landmark).unsqueeze(0),
                       (self.loss_weight['cwd'] * d_loss_cw).unsqueeze(0))).detach()

    def distill_loss_cls(self, logits_student, logits_teacher, num_classes, temperature=20):
        logits_student = logits_student.view(-1, num_classes)
        logits_teacher = logits_teacher.view(-1, num_classes)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)

        d_loss_cls = F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        d_loss_cls *= temperature ** 2
        return d_loss_cls

    def distill_loss_cw(self, s_feats, t_feats, temperature=1):
        N, C, H, W = s_feats[0].shape
        # print(N,C,H,W)
        loss_cw = F.kl_div(F.log_softmax(s_feats[0].view(N, C, H * W) / temperature, dim=2),
                           F.log_softmax(t_feats[0].view(N, C, H * W).detach() / temperature, dim=2),
                           reduction='sum',
                           log_target=True) * (temperature * temperature) / (N * C)

        N, C, H, W = s_feats[1].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[1].view(N, C, H * W) / temperature, dim=2),
                            F.log_softmax(t_feats[1].view(N, C, H * W).detach() / temperature, dim=2),
                            reduction='sum',
                            log_target=True) * (temperature * temperature) / (N * C)

        N, C, H, W = s_feats[2].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[2].view(N, C, H * W) / temperature, dim=2),
                            F.log_softmax(t_feats[2].view(N, C, H * W).detach() / temperature, dim=2),
                            reduction='sum',
                            log_target=True) * (temperature * temperature) / (N * C)
        # print(loss_cw)
        return loss_cw

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 15)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            # For i, item == targets[i].
            #             Shape of targets[i]: (6, ) = [image_id, label, x, y, w, h]
            # item[0] == targets[i][0] == image_id
            # item[1:] == [label, x, y, w, h]
            # targets_list[int(item[0])] == targets_list[image_id]
            targets_list[int(item[0])].append(item[1:])
        # max_len means the max number of objects
        # len(l) is the number of objects in a input image.
        max_len = max((len(l) for l in targets_list))
        # For example, len(targets_list) == 2, ==> batch size = 2
        #              len(targets_list[0]) == 1, ==> img_id = 0 has 1 object
        #              len(targets_list[1]) == 3, ==> img_id = 1 has 3 objects
        #              ==> max_len == 3, which means the max number of objects in batch is 3.
        # the result of map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)
        #              len(targets_list[0]) == 3, len(targets_list[1]) == 3.
        # So, the upper code line adds the dummy list [-1,0,0,0,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        # to a list whose length is smaller than max_len.
        targets = torch.from_numpy(
            np.array(list(
                map(lambda l: l + [[-1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]] * (max_len - len(l)),
                    targets_list)))[:, 1:, :]).to(targets.device)
        # [:, 1:, :] --> the first list element [0.0, 0.0, 0.0, 0.0, 0.0 ... 0.0]  is removed
        # scale_tensor has the orignal values of height and width of input
        # The shape of targets is (batch_size, max_len, 5)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:5] = xywh2xyxy(batch_target)

        # Landmarks
        gt_ldmks = targets[:, :, 5:15].mul_(scale_tensor[0, 0])
        return targets, gt_ldmks

    def bbox_decode(self, anchor_points, pred_dist):
        pred_distri = pred_dist.clone()[..., :-10]
        pred_ldmks = pred_dist.clone()[..., -10:]
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_distri.shape
            pred_distri = F.softmax(pred_distri.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(
                self.proj.to(pred_distri.device))
            # shape = (batch_size, n_anchors, 4)
            # Think... Simple example
            # batch_size == 1, n_anchors == 1, self.reg_max == 2
            # pred_distri.view() --> pred_distri's shape: (1, 1, 4, 3)
            # F.softmax(pred_distri.view(), -1)
            #        --> the result's shape is also (1, 1, 4, 3)
            #                                                 |-> softmax function is applied
            # Let's see the lower example.
            #                         A ----------- |
            #                         |             |
            #                         |             |
            #                         |    a----    |
            #                         |    | C |    |
            #                         |    ----b    |
            #                         | ----------- B
            # Note that)
            # the box around C is a grid cell.
            # The box created by A and B is a bbox of an object.
            # A is the left-top point.
            # B is the right-bottom point.
            # C is the center point of an anchor (and the center of the grid cell).
            # A = (Ax, Ay), B = (Bx, By), C = (Cx, Cy)
            # Distance between A and C along x-axis = |Ax-Cx|
            # Distance between A and C along y-axis = |Ay-Cy|
            # Distance between B and C along y-axis = |Bx-Cx|
            # Distance between B and C along y-axis = |By-Cy|
            # For dfl, the yolov6 network predicts the distance, |Ax-Cx|, |Ay-Cy|, |Bx-Cx|, and |By-Cy|
            # The yolov6 network doesn't predict the distance directly.
            # If the network would do, the pred_distri's shape be (1, 1, 4, 1).
            # However, the pred_distri's shape actually is (1, 1, 4, 3) because reg_max = 2.
            # prob_pred_distri = F.softmax(pred_dist.view(), -1), whose shape is (1, 1, 4, 3).
            # prob_pred_distri[0, 0, 0, 0] is the probability that the distance |Ax-Cx| is 0.
            # prob_pred_distri[0, 0, 0, 1] is the probability that the distance |Ax-Cx| is 1.
            # prob_pred_distri[0, 0, 0, 2] is the probability that the distance |Ax-Cx| is 2.
            # Then, prob_pred_distri[0, 0, 0, 0] * 0 + prob_pred_distri[0, 0, 0, 1] * 1 + prob_pred_distri[0, 0, 0, 2] * 2 is
            # the average predicted distance, which is the final prediced distance |Ax-Cx|.
            # prob_pred_distri[0, 0, 0, 0] * 0 + prob_pred_distri[0, 0, 0, 1] * 1 + prob_pred_distri[0, 0, 0, 2] * 2 can be
            # obtained by F.softmax(pred_distri.view(), dim=-1).matmul([0, 1, 2]).
        #
        pred_ldmks[..., 0:9:2] += torch.unsqueeze(anchor_points[..., 0:1], 0).repeat(pred_dist.shape[0], 1, 5)
        pred_ldmks[..., 1:10:2] += torch.unsqueeze(anchor_points[..., 1:2], 0).repeat(pred_dist.shape[0], 1, 5)
        return dist2bbox(pred_distri, anchor_points), pred_ldmks


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, t_pred_dist, t_pred_bboxes, temperature, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            t_pred_bboxes_pos = torch.masked_select(t_pred_bboxes,
                                                    bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist.clone()[..., :-10], dist_mask).reshape([-1, 4, self.reg_max + 1])
                t_pred_dist_pos = torch.masked_select(
                    t_pred_dist.clone()[..., :-10], dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         target_ltrb_pos) * bbox_weight
                d_loss_dfl = self.distill_loss_dfl(pred_dist_pos, t_pred_dist_pos, temperature) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                    d_loss_dfl = d_loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
                    d_loss_dfl = d_loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.
                d_loss_dfl = pred_dist.sum() * 0.

        else:

            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.
            d_loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl, d_loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def distill_loss_dfl(self, logits_student, logits_teacher, temperature=20):

        logits_student = logits_student.view(-1, 17)
        logits_teacher = logits_teacher.view(-1, 17)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)

        d_loss_dfl = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        d_loss_dfl *= temperature ** 2
        return d_loss_dfl


class WingLoss(nn.Module):
    def __init__(self, w=10, e=2):
        super(WingLoss, self).__init__()
        # https://arxiv.org/pdf/1711.06753v4.pdf
        self.w = w
        self.e = e
        self.C = self.w - self.w * np.log(1 + self.w / self.e)

    def forward(self, x, t, sigma=1):
        weight = torch.ones_like(t)
        weight[torch.where(t == -1)] = 0
        diff = weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < self.w).float()
        y = flag * self.w * torch.log(1 + abs_diff / self.e) + (1 - flag) * (abs_diff - self.C)
        return y.sum()


class LandmarksLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=1.0):
        super(LandmarksLoss, self).__init__()
        self.loss_fcn = WingLoss()
        self.alpha = alpha

    def forward(self, pred_ldmks, gt_ldmks, mask):
        mask = mask.unsqueeze(-1).repeat([1, 1, 10])
        pred_ldmks_pos = torch.masked_select(pred_ldmks, mask).reshape([-1, 10])
        gt_ldmks_pos = torch.masked_select(gt_ldmks, mask).reshape([-1, 10])

        lmdk_mask = torch.where(gt_ldmks_pos < 0, torch.full_like(gt_ldmks_pos, 0.), torch.full_like(gt_ldmks_pos, 1.0))
        loss = self.loss_fcn(pred_ldmks_pos * lmdk_mask, gt_ldmks_pos * lmdk_mask)
        return loss / (torch.sum(lmdk_mask) + 10e-14)
