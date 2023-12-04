import torch
import copy
import numpy as np
import os
from yolov6.utils.metrics import ap_per_class


def buffer_statistics(imgs, outputs, targets, shapes, stats):
    #
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()
    # ===== For pr metric =====
    eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])
    # Statistics per image
    # This code is based on
    # https://github.com/ultralytics/yolov5/blob/master/val.py
    for si, pred in enumerate(eval_outputs):
        labels = targets[targets[:, 0] == si, 1:]
        nl = len(labels)
        tcls = labels[:, 0].tolist() if nl else []  # target class
        stats['seen'] += 1

        if len(pred) == 0:
            if nl:
                stats['stats'].append(
                    (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
            continue

        # Predictions
        predn = pred.clone()
        scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

        # Assign all predictions as incorrect
        correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
        if nl:
            from yolov6.utils.nms import xywh2xyxy

            # target boxes
            tbox = xywh2xyxy(labels[:, 1:5])
            tbox[:, [0, 2]] *= imgs[si].shape[1:][1]
            tbox[:, [1, 3]] *= imgs[si].shape[1:][0]

            scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

            labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

            from yolov6.utils.metrics import process_batch

            correct = process_batch(predn, labelsn, iouv)

        # Append statistics (correct, conf, pcls, tcls)
        stats['stats'].append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    '''Rescale coords (xyxy) from img1_shape to img0_shape.'''
    if ratio_pad is None:  # calculate from img0_shape
        gain = [min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])]  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [0, 2]] /= gain[0]  # raw x gain
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [1, 3]] /= gain[0]  # y gain

    if isinstance(coords, torch.Tensor):  # faster individually
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
    else:  # np.array (faster grouped)
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return coords


def calc_pr_metrics(stats_dict, save_dir):
    stats = stats_dict['stats']
    # ===== For pr metric =====
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False)
        AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=1)  # number of targets per class
        # Print results
        results_dict = {
            'num_targets': nt,
            'P_at_0_5iou': mp,
            'R_at_0_5iou': mr,
            'iou50_best_mf1_thres': AP50_F1_max_idx / 1000.0,
            'F1_at_0_5iou': f1.mean(0)[AP50_F1_max_idx],
            'mAP_at_0_5': map50,
            'mAP_at_0_5_0_95': map,
            'precision_all': p,
            'recal_all': r,
            'AP_all': ap,
            'f1_all': f1
        }
        print('map50 :' + str(results_dict['mAP_at_0_5']))
        from scipy.io import savemat
        savemat(os.path.join(save_dir, 'results.mat'), results_dict)
    else:
        print("Calculate metric failed, might check dataset.")
        results_dict = dict()
    return results_dict
