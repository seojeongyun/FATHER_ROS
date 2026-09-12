import pickle
import sys, os

import cv2
import torch
import numpy as np
import torch.utils.data.dataset
from tqdm import tqdm
#
import random
from utilss.augmentation import ArgumentationSchedule

import torch.nn.functional as F


def pad_to_square(image, pad_value=0):
    _, h, w = image.shape

    difference = abs(h - w)

    # (top, bottom) padding or (left, right) padding
    if h <= w:
        top = difference // 2
        bottom = difference - difference // 2
        pad = [0, 0, top, bottom]
    else:
        left = difference // 2
        right = difference - difference // 2
        pad = [left, right, 0, 0]

    # Add padding
    image = F.pad(image, pad, mode='constant', value=pad_value)
    return image, pad


def resize(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)


def default_loader(bgrImg224):
    input = torch.zeros(1, 3, 224, 224)
    img = bgrImg224
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img = torch.from_numpy(img).float()
    input[0, :, :, :] = img
    return input


def torch_loader(bgrImg224, dimension=224, device=torch.device('cpu')):
    if bgrImg224.shape[0] != bgrImg224.shape[1]:
        raise ("input picture not a square")

    if bgrImg224.shape[0] != dimension:
        bgrImg224 = cv2.resize(bgrImg224, (dimension, dimension))

    img = torch.from_numpy(bgrImg224).float().to(device)
    img = img.transpose(2, 0).transpose(1, 2) / 255.
    return img


class IDDataSet():
    def __init__(self, img_size, root_dir, cache_file="", augmentation=False):
        self.root_dir = root_dir
        self.file_paths = []
        self.file_IDs = []
        self.file_labels = []
        self.img_size = img_size

        self.IDs = []
        self.IDsLabels = {}

        self.augmentation = augmentation

        if os.path.exists(cache_file) is False:
            for dir in tqdm(os.listdir(self.root_dir)):
                if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                    raise ("DIR Error")
                self.IDs.append(dir)

                label_idx = 0
                for ID in self.IDs:
                    self.IDsLabels[ID] = label_idx
                    label_idx += 1

            for dir in tqdm(os.listdir(self.root_dir)):
                if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                    raise ("DIR Error")
                for file in os.listdir(os.path.join(self.root_dir, dir)):
                    if os.path.splitext(file)[1] in [".jpg", ".bmp", ".png"]:
                        self.file_paths.append(os.path.join(self.root_dir, dir, file))
                        self.file_IDs.append(dir)
                        self.file_labels.append(self.IDsLabels[dir])
            print("data set loaded from scratch len: {}".format(len(self.file_paths)))
            sys.stdout.flush()
            with open(cache_file, "wb") as f:
                pickle.dump([self.file_paths,
                             self.file_IDs,
                             self.file_labels,
                             self.IDs,
                             self.IDsLabels], f)
        else:
            print("start loading from cache")
            sys.stdout.flush()
            with open(cache_file, "rb") as f:
                self.file_paths, self.file_IDs, self.file_labels, self.IDs, self.IDsLabels = pickle.load(f)
            print("data set loaded from cache len: {}".format(len(self.file_paths)))
            sys.stdout.flush()

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_id = self.file_IDs[idx]
        file_label = self.file_labels[idx]

        bgrIm = cv2.imread(file_path)
        if self.augmentation:
            bgrIm = ArgumentationSchedule(bgrIm, random.randint(0, 9))

        # image shape != self.img_size
        # Resize
        # --- Debugging Codes ----
        # import matplotlib.pyplot as plt
        # plt.imshow()
        # --------------------------

        return torch_loader(bgrIm, dimension=self.img_size), file_path, file_id, file_label

    def __len__(self):
        return len(self.file_paths)


class totaldata():
    def __init__(self, img_size, root_dir, cache_file="", augmentation=False):
        self.root_dir = root_dir
        self.file_paths = []
        self.file_IDs = []
        self.file_labels = []
        self.img_size = img_size

        self.IDs = []
        self.IDsLabels = {}

        self.augmentation = augmentation

        if os.path.exists(cache_file) is False:
            for dir in tqdm(os.listdir(self.root_dir)):
                if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                    raise ("DIR Error")
                self.IDs.append(dir)

                label_idx = 0
                for ID in self.IDs:
                    self.IDsLabels[ID] = label_idx
                    label_idx += 1

            for dir in tqdm(os.listdir(self.root_dir)):
                if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                    raise ("DIR Error")
                for file in os.listdir(os.path.join(self.root_dir, dir)):
                    if os.path.splitext(file)[1] in [".jpg", ".bmp", ".png"]:
                        self.file_paths.append(os.path.join(self.root_dir, dir, file))
                        self.file_IDs.append(dir)
                        self.file_labels.append(self.IDsLabels[dir])
            print("data set loaded from scratch len: {}".format(len(self.file_paths)))
            sys.stdout.flush()
            with open(cache_file, "wb") as f:
                pickle.dump([self.file_paths,
                             self.file_IDs,
                             self.file_labels,
                             self.IDs,
                             self.IDsLabels], f)
        else:
            print("start loading from cache")
            sys.stdout.flush()
            with open(cache_file, "rb") as f:
                self.file_paths, self.file_IDs, self.file_labels, self.IDs, self.IDsLabels = pickle.load(f)
            print("data set loaded from cache len: {}".format(len(self.file_paths)))
            sys.stdout.flush()

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_id = self.file_IDs[idx]
        file_label = self.file_labels[idx]

        bgrIm = cv2.imread(file_path)
        if self.augmentation:
            bgrIm = ArgumentationSchedule(bgrIm, random.randint(0, 9))

        if bgrIm.shape[0] != bgrIm.shape[1]:
            pad_to_square(torch.tensor(bgrIm).permute(2, 0, 1).type(torch.float32))
            bgrIm = resize(torch.tensor(bgrIm).permute(2, 0, 1).type(torch.float32), self.img_size)
            bgrIm = np.array(bgrIm.permute(1, 2, 0)).astype(np.uint8)

        # image shape != self.img_size
        # Resize
        # --- Debugging Codes ----
        # import matplotlib.pyplot as plt
        # plt.imshow()
        # --------------------------

        return torch_loader(bgrIm, dimension=self.img_size), file_path, file_id, file_label

    def __len__(self):
        return len(self.file_paths)


class VGGdataset():
    def __init__(self, img_size, root_dir, cache_file="", augmentation=False):
        self.root_dir = root_dir
        self.file_paths = []
        self.file_IDs = []
        self.file_labels = []
        self.img_size = img_size

        self.IDs = []
        self.IDsLabels = {}

        self.augmentation = augmentation

        if os.path.exists(cache_file) is False:
            for dir in tqdm(os.listdir(self.root_dir)):
                if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                    raise ("DIR Error")
                self.IDs.append(dir)

                label_idx = 0
                for ID in self.IDs:
                    self.IDsLabels[ID] = label_idx
                    label_idx += 1

            for dir in tqdm(os.listdir(self.root_dir)):
                if os.path.isdir(os.path.join(self.root_dir, dir)) is False:
                    raise ("DIR Error")
                for file in os.listdir(os.path.join(self.root_dir, dir)):
                    if os.path.splitext(file)[1] in [".jpg", ".bmp", ".png"]:
                        self.file_paths.append(os.path.join(self.root_dir, dir, file))
                        self.file_IDs.append(dir)
                        self.file_labels.append(self.IDsLabels[dir])
            print("data set loaded from scratch len: {}".format(len(self.file_paths)))
            sys.stdout.flush()
            with open(cache_file, "wb") as f:
                pickle.dump([self.file_paths,
                             self.file_IDs,
                             self.file_labels,
                             self.IDs,
                             self.IDsLabels], f)
        else:
            print("start loading from cache")
            sys.stdout.flush()
            with open(cache_file, "rb") as f:
                self.file_paths, self.file_IDs, self.file_labels, self.IDs, self.IDsLabels = pickle.load(f)
            print("data set loaded from cache len: {}".format(len(self.file_paths)))
            sys.stdout.flush()

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        file_id = self.file_IDs[idx]
        file_label = self.file_labels[idx]

        bgrIm = cv2.imread(file_path)
        if self.augmentation:
            bgrIm = ArgumentationSchedule(bgrIm, random.randint(0, 9))

        if bgrIm.shape[0] != bgrIm.shape[1]:
            pad_to_square(torch.tensor(bgrIm).permute(2, 0, 1).type(torch.float32))
            bgrIm = resize(torch.tensor(bgrIm).permute(2, 0, 1).type(torch.float32), self.img_size)
            bgrIm = np.array(bgrIm.permute(1, 2, 0)).astype(np.uint8)

        # image shape != self.img_size
        # Resize
        # --- Debugging Codes ----
        # import matplotlib.pyplot as plt
        # plt.imshow()
        # --------------------------

        return torch_loader(bgrIm, dimension=self.img_size), file_path, file_id, file_label

    def __len__(self):
        return len(self.file_paths)


if __name__ == '__main__':
    dataset_object = VGGdataset(img_size=112,
                                root_dir="/storage/sjpark/VGGFace2/vgg_train_6s_hs_CBAM_9601",
                                cache_file='/storage/sjpark/VGGFace2/cache/vggface2_train_img_6s_hs_CBAM_9601.pickle')
    # dataset_object = totaldata(img_size=112, root_dir="/storage/sjpark/total_data/train",cache_file="/storage/sjpark/total_data/cache/total_train_img.pickle"
    #                            )
    data_loader = torch.utils.data.DataLoader(dataset_object, batch_size=8, shuffle=True, num_workers=0)

    for img, file_path, id, label in tqdm(data_loader):
        bgrIm = (img[0] * 255.0).numpy().transpose(1, 2, 0).astype(np.uint8)
        cv2.imshow("x", bgrIm)
        cv2.waitKey(0)
        pass
    pass
