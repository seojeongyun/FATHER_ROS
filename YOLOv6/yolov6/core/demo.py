import cv2
import numpy as np
import torch
import time
from yolov6.utils.events import LOGGER
from yolov6.utils.nms import non_max_suppression_face, non_max_suppression
from copy import deepcopy


class demo:
    def __init__(self, weight, resize_size, device,
                 onnx=False, conf_thres=0.03, iou_thres=0.65, nms_det=300, NET_TYPE='6s'):
        self.device = device
        self.resized_size = resize_size
        #
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.nms_det = nms_det
        #
        self.net_type = NET_TYPE
        self.model = self.get_model(onnx, weight)
        if onnx:
            self.out_name = [i.name for i in self.model.get_outputs()]

    def get_model(self, onnx, weight):
        from yolov6.utils.torch_utils import fuse_model
        LOGGER.info("Loading torch model...")
        ckpt = torch.load(weight, map_location=self.device)
        model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        model = fuse_model(model).eval()
        return model

    @torch.no_grad()
    def run(self, imgs: list):
        bboxes = self.torch_run(imgs)
        return bboxes

    def torch_run(self, imgs: list):
        # 1. imgs is a list including raw images from a camera or dataset.
        # 2. The shape of an image in the list is (W, H, C).
        # 3. Images should be listed in the order of red, green, and blue.
        #
        # ----- Resize the original images to the specific "resized_size".
        resized_imgs, ori_size, resized_size = img_resize(imgs, self.resized_size)

        # ----- Pad zeros to the resized images
        resized_imgs, ratios, pads = letterbox(resized_imgs,
                                               new_shape=(self.resized_size, self.resized_size),
                                               auto=False, scaleup=False)

        # ----- Inference to detect boxes from the resized and padded images.
        resized_imgs = torch.from_numpy(np.stack(resized_imgs, axis=0).transpose(0, 3, 1, 2)) / 255.0
        resized_imgs = resized_imgs.to(self.device, non_blocking=True).float()
        #
        start = time.time()
        outputs, _ = self.model(resized_imgs)
        if self.net_type == '6n':
            outputs = non_max_suppression(outputs,
                                          self.conf_thres,
                                          self.iou_thres,
                                          max_det=self.nms_det)
        else:
            outputs = non_max_suppression_face(outputs,
                                               self.conf_thres,
                                               self.iou_thres,
                                               max_det=self.nms_det)
        bboxes = self.get_bboxes(deepcopy(outputs))
        print(f'\n {len(imgs) / (time.time() - start):.2f} FPS')

        # ----- Scale the coordinate of detected bboxes
        # Get the shape of images
        shapes = []
        for idx in range(len(resized_size)):
            h, w = resized_size[idx]
            h0, w0 = ori_size[idx]
            shapes.append((ori_size[idx], ((h * ratios[idx] / h0, w * ratios[idx] / w0), pads[idx])))
        rescaled_bboxes = self.rescale_bboxes(bboxes, resized_imgs, shapes)
        return rescaled_bboxes, outputs, ori_size, resized_size, ratios, pads

    @staticmethod
    def rescale_bboxes(bboxes, imgs, shapes):
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

        #
        rescaled_bboxes = []
        for i, pred in enumerate(bboxes):
            if len(pred) == 0:
                rescaled_bboxes.append([])
            else:
                scale_coords(imgs[i].shape[1:], pred[:, :4], shapes[i][0], shapes[i][1])
                rescaled_bboxes.append(torch.cat([pred[:, :4].round().int(), bboxes[i][:, 4:6]], dim=1))
        return rescaled_bboxes

    @staticmethod
    def get_bboxes(outputs):
        out_boxes = []
        for batch_id in range(len(outputs)):
            if len(outputs[batch_id]) == 0:
                out_boxes.append([])
            else:
                out_boxes.append(outputs[batch_id][:, 0:6])
        return out_boxes

    @staticmethod
    def post_processing(bboxes, imgs):
        post_ = []
        cropped_face = []
        for im_idx in range(len(imgs)):
            img = imgs[im_idx]
            for bbox_idx in range(len(bboxes[im_idx])):
                bbox = bboxes[im_idx][bbox_idx]
                cropped_face.append(img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                img = cv2.rectangle(img.copy(), (int(bbox[0]), int(bbox[1])),
                                    (int(bbox[2]), int(bbox[3])), (24, 252, 0), 2)
                img = cv2.putText(img, str(np.round(bbox[4].cpu().numpy(), 2)), (int(bbox[0]), int(bbox[1]) - 10),
                                  cv2.FONT_HERSHEY_COMPLEX, 0.5, (24, 252, 0), thickness=1)
            post_.append(img)
        return post_, cropped_face


def img_resize(im: list, resized_size):
    ori_size = []
    resize_size = []
    img_resized = []
    for orig_img in im:
        h0, w0 = orig_img.shape[:2]  # origin shape
        #
        r = resized_size / max(h0, w0)
        if r != 1:
            resized = cv2.resize(
                orig_img,
                (int(w0 * r), int(h0 * r)),
                interpolation=cv2.INTER_AREA
            )
        else:
            resized = orig_img
        #
        ori_size.append((h0, w0))
        resize_size.append(resized.shape[:2])
        img_resized.append(resized.astype(np.float32))
    return img_resized, ori_size, resize_size


def letterbox(ims, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    im_list = []
    ratio_list = []
    pad_list = []
    for im in ims:
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        elif isinstance(new_shape, list) and len(new_shape) == 1:
            new_shape = (new_shape[0], new_shape[0])

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im_dtype = im.dtype
        if im_dtype != 'uint8':
            im = im.astype(np.uint8)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        #
        im_list.append(im.astype(im_dtype))
        ratio_list.append(r)
        pad_list.append((dw, dh))

    return im_list, ratio_list, pad_list
