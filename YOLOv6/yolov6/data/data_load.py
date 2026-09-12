import os

from torch.utils.data.dataloader import DataLoader
from yolov6.data.face_datasets import TrainValDataset
from yolov6.utils.events import NCOLS
# from yolov6.data.coco_datasets import TrainValDataset

def create_dataloader(
        path,
        img_size,
        batch_size,
        stride,
        hyp=None,
        augment=False,
        check_images=False,
        check_labels=False,
        pad=0.0,
        workers=8,
        shuffle=False,
        data_dict=None,
        task="Train",

):
    """Create general dataloader.
        This is slightly different from the original data_load in https://github.com/meituan/YOLOv6/blob/main/yolov6/data/data_load.py
        This version doesn't consider the rectangular input image.
        In other words, the shape of input images always are square..!
        For a beginner, i remove the code parts for rectangular input images.
        In addition, No distribution.
    Returns dataloader and dataset
    """
    dataset = TrainValDataset(
        path,
        img_size,
        batch_size,
        augment=augment,
        hyp=hyp,
        check_images=check_images,
        check_labels=check_labels,
        stride=int(stride),
        pad=pad,
        data_dict=data_dict,
        task=task,
    )
    batch_size = min(batch_size, len(dataset))
    workers = min(
        [
            os.cpu_count(),
            batch_size if batch_size > 1 else 0,
            workers,
        ]
    )  # number of workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        collate_fn=TrainValDataset.collate_fn
    )


if __name__ == '__main__':
    print('Debugging')

    # Data class and Loader
    from yolov6.utils.events import load_yaml
    from yolov6.utils.config import Config

    cfg = Config.fromfile('../../configs/base/yolov6l_base.py')
    data_dict = load_yaml('../../data/WIDER_FACE.yaml')
    path = data_dict['train']
    nc = int(data_dict['nc'])
    class_names = data_dict['names']
    assert len(class_names) == nc, f'the length of class_names should match the number of classes defined'
    grid_size = max(int(max(cfg.model.head.strides)), 32)

    img_dir = path
    img_size = 640
    batch_size = 16
    stride = 32
    hyp = dict(cfg.data_aug)
    augment = True
    workers = 8
    shuffle = True
    check_images = True
    check_labels = True
    task = 'train'

    loader = create_dataloader(
        path=img_dir,
        img_size=img_size,
        batch_size=batch_size,
        stride=grid_size,
        hyp=hyp,
        augment=augment,
        workers=8,
        shuffle=shuffle,
        check_images=check_images,
        check_labels=check_labels,
        data_dict=data_dict,
        task='train'
    )

    from tqdm import tqdm

    pbar = enumerate(loader)
    pbar = tqdm(pbar, total=len(loader), ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    for step, batch_data in pbar:
        print('load data...')
