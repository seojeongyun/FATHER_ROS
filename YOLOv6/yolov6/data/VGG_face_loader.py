import argparse
import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms
from torch.utils.data import Dataset
import torch.nn.functional as F


def get_arg_vgg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--half', action='store_true', help='FP16')
    parser.add_argument('--data_path', default='/storage/sjpark/VGGFace2/', type=str, help='path of dataset')
    parser.add_argument('--batch_size', default=50, help='train or test')
    parser.add_argument('--task', type=str, default='test', help='train or test')
    args = parser.parse_args()
    print(args)
    return args


class data_loader:
    def __init__(self, path, type):
        self.base_path = path
        self.type = type
        self.target_size = 640
        self.img_path = self.get_path()
        self.resize = self.get_resize(self.target_size)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = cv2.imread(self.img_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = torch.tensor(img.transpose())
        # img = self.pad(img)
        # img = img.numpy().transpose()
        return self.img_path[index], img

    @staticmethod
    def resize_img(img_orig:list, half=False):
        batch_img = []
        batch_ratio = []
        batch_dwdh = []
        for idx in range(len(img_orig)):
            image = img_orig[idx].copy()
            #
            img, ratio, dwdh = data_loader.letterbox(image)
            #
            img = img.transpose((2,0,1))
            image = np.ascontiguousarray(img)
            #
            if half:
                img = image.astype(np.float16)
            else:
                img = image.astype(np.float32)

            img = torch.Tensor(img)
            batch_img.append(img)
            batch_ratio.append(ratio)
            batch_dwdh.append(dwdh)
        resized_img = torch.stack(batch_img)

        return resized_img, batch_ratio, batch_dwdh

    def get_path(self):
        path = glob.glob(self.base_path + self.type + '/*' + '/*.jpg')
        path = sorted(path)
        return path

    def get_resize(self, target_size):
        resize = torchvision.transforms.Resize((target_size, target_size))
        return resize

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Preprocess image. For details, see:
        https://github.com/meituan/YOLOv6/issues/613
        """

        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def pad(self, image):
        c, h, w = image.shape
        dw = (self.target_size - w) / 2
        dh = (self.target_size - h) / 2

        if dw is not int:
            left = round(dw)
            right = round(dw)
        else:
            left = dw
            right = dw

        if dh is not int:
            up = round(dh)
            down = round(dh)
        else:
            up = dh
            down = dh

        image = F.pad(image, (left, right, up, down), "constant", 0)
        image = self.resize(image)
        return image

    @staticmethod
    def collate_fn(batch):
        path, inputs = zip(*batch)
        # inputs = torch.stack(inputs, dim=0)
        inputs = list(inputs)
        path = list(path)
        return inputs, path


if __name__ == '__main__':
    args = get_arg_vgg()
    train = data_loader(args.data_path, type='train')
    test = data_loader(args.data_path, type='test')

    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=5,
        shuffle=True,
        collate_fn=data_loader.collate_fn
    )

    for batch_id, data in enumerate(train_loader):
        image, path = data[0], data[1]
        print(path, image.shape)
