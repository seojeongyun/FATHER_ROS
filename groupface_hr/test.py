import sys, os
import torch, cv2
import numpy as np
import time
import torch.nn.functional as F
import dataset.dataset
import random
import matplotlib.pyplot as plt

from models.GroupFace import GroupFace
from loss.loss import AddMarginProduct, ArcMarginProduct, SphereProduct
from loss.focal_loss import FocalLoss

from dataset.dataset import VGGdataset, torch_loader
from Config.config_test import Config

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity


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


def load_gallery_probe(parameters):
    gallery_path = parameters['options'].test_path
    #
    all_path = []
    p_files = []
    p_ids = []
    g_files = []
    g_ids = []

    # make gallery_probe slicing index
    num_thres = parameters['options'].num_thres
    for i, dir in enumerate(os.listdir(gallery_path)):
        num = 0
        files_in_dir = os.listdir(os.path.join(gallery_path, dir))
        random.shuffle(files_in_dir)
        for file in files_in_dir:
            num += 1
            if num <= num_thres:
                file_path = os.path.join(gallery_path, dir, file)
                all_path.append(file_path)
    # split gallery_probe
    for j in range(0, len(os.listdir(gallery_path))):
        tmp = random.choice(all_path[j * num_thres: (1 + j) * num_thres])
        p_files.append(tmp)
        p_ids.append(tmp.split("/")[-2])
        for i in all_path[j * num_thres: (1 + j) * num_thres]:
            if i == tmp:
                pass
            else:
                g_files.append(i)
                g_ids.append(i.split("/")[-2])

    return g_files, g_ids, p_files, p_ids


def load_gallery_probe_dict(parameters):
    gallery_path = parameters['options'].test_path
    #
    selected_dict = dict()
    # make gallery_probe slicing index
    num_thres = parameters['options'].num_thres
    for i, dir in enumerate(os.listdir(gallery_path)):
        num = 0
        files_in_dir = os.listdir(os.path.join(gallery_path, dir))
        random.shuffle(files_in_dir)
        selected_dict[dir] = {}
        for file in files_in_dir:
            num += 1
            if num <= num_thres:
                selected_dict[dir].update({file: {'path': os.path.join(gallery_path, dir, file)}})

    # split gallery_probe
    gallery_dict = dict()
    prob_dict = dict()
    gallery_dir_name = list(selected_dict.keys())
    for j in range(0, len(os.listdir(gallery_path))):
        selected_gallery = gallery_dir_name[j]
        # Select prob
        gallery_files = selected_dict[selected_gallery]
        prob_dict[selected_gallery] = {list(gallery_files.keys())[0]: gallery_files[list(gallery_files.keys())[0]]}
        gallery_files.pop(list(gallery_files.keys())[0])
        # Select gallery
        gallery_dict[selected_gallery] = gallery_files
        assert not list(prob_dict[selected_gallery].keys()) in list(gallery_dict[selected_gallery].keys())
    return gallery_dict, prob_dict


def setup_device(option):
    if len(opt.gpu_id) != 0:
        device = torch.device("cuda:{}".format(option.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def setup_dataloader(option):
    dataset = VGGdataset(option['img_size'], option['path'], option['cache_file'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=option['batch_size'], shuffle=True, num_workers=0)
    return loader


def setup_network(option, device):
    model = GroupFace(resnet=option.resnet, feature_dim=option.feature_dim, groups=option.groups).to(device)
    if option.checkpoints_file is not None:
        print("Loading saved weights {}".format(option.checkpoints_file))
        file_path = os.path.join(option.checkpoints_save_path, option.checkpoints_file)
        if os.path.exists(file_path):
            weights = torch.load(file_path, map_location=device)
            model.load_state_dict(weights['model'])
        else:
            ValueError("There is no weight file!!")
    return model


def setup_loss(option, device):
    if option.loss == 'focal_loss':
        loss = FocalLoss(gamma=2)
    else:
        loss = torch.nn.CrossEntropyLoss()

    if option.fc_metric == 'arc':
        fc_metric = ArcMarginProduct(in_features=option.feature_dim, out_features=option.num_classes, s=30, m=0.5,
                                     device=device, easy_margin=option.easy_margin)
    elif option.fc_metric == 'add':
        fc_metric = AddMarginProduct(in_features=option.feature_dim, out_features=option.num_classes, s=30, m=0.35)
    elif option.fc_metric == 'sphere':
        fc_metric = SphereProduct(in_features=option.feature_dim, out_features=option.num_classes, m=4)
    else:
        fc_metric = torch.nn.Linear(512, opt.num_classes)

    return loss, fc_metric


def setup_optimizer(option, model, fc_metric):
    if option.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': fc_metric.parameters()}],
                                    lr=option.lr, weight_decay=option.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': fc_metric.parameters()}],
                                     lr=option.lr, weight_decay=option.weight_decay)
    return optimizer


def setup_scheduler(option, optimizer):
    if option.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
    else:
        NotImplementedError('Not Implemented...{}'.format(option.scheduler))
    return scheduler


def save_model(model, save_path, dataset_name, iter_cnt):
    save_name = os.path.join(save_path, dataset_name + '_' + str(iter_cnt) + '.pth')
    torch.save({'model': model.state_dict()}, save_name)


def eval(parameters):
    writer = parameters['writer']
    device = parameters['device']
    #
    model = parameters['model']

    #
    model.eval()
    #
    gallery_files, gallery_ids, probe_files, probe_ids = load_gallery_probe(parameters)
    #
    gallery_feats = []
    for gallery_file in gallery_files:
        # Obtain a feature vector of a gallery face

        gallery_file, pad = pad_to_square(torch.Tensor(cv2.imread(gallery_file)).permute(2, 0, 1))
        gallery_file = resize(gallery_file, parameters["options"].img_size)
        gallery_file = np.array(gallery_file.permute(1, 2, 0)).astype(np.uint8)
        _, final, _, _ = model(
            torch_loader(gallery_file, dimension=parameters['options'].img_size).unsqueeze(0).to(device))
        gallery_feat = final / torch.norm(final, p=2, keepdim=False)
        gallery_feat = gallery_feat.detach().cpu().reshape(1, parameters['options'].feature_dim).numpy()
        gallery_feats.append(gallery_feat)
    gallery_feats = np.stack(gallery_feats, axis=0).squeeze(axis=1)
    #
    evaled_cnt = 0.0
    right_cnt = 0.0
    false_cnt = 0.0
    sum_right_top1_score = []
    sum_false_top1_socre = []

    for probe_file, GT_id in zip(probe_files, probe_ids):
        evaled_cnt += 1
        probe_file, pad = pad_to_square(torch.Tensor(cv2.imread(probe_file)).permute(2, 0, 1))
        probe_file = resize(probe_file, parameters["options"].img_size)
        probe_file = np.array(probe_file.permute(1, 2, 0)).astype(np.uint8)
        # Obtain a feature vector of a probe face
        _, final, _, _ = model(
            torch_loader(probe_file, dimension=parameters['options'].img_size).unsqueeze(0).to(device))
        probe_feat = final / torch.norm(final, p=2, keepdim=False)
        probe_feat = probe_feat.detach().cpu().reshape(1, parameters['options'].feature_dim).numpy()

        scores = cosine_similarity(probe_feat, gallery_feats)
        max_idx = np.argmax(scores)
        predicted_id = gallery_ids[max_idx]
        if predicted_id == GT_id:
            right_cnt += 1.0
            sum_right_top1_score.append(scores[0][gallery_ids.index(GT_id)])

        else:
            false_cnt += 1.0
            sum_false_top1_socre.append(scores[0][gallery_ids.index(GT_id)])

        if evaled_cnt > 100:
            break

    plt.hist(sum_right_top1_score, bins=50, range=(-1, 1), alpha=0.5, label='sum_right_top1_score')
    plt.legend()
    plt.show()

    plt.hist(sum_false_top1_socre, bins=50, range=(-1, 1), alpha=0.5, label='sum_false_top1_score')
    plt.legend()
    plt.show()

    writer.add_scalar(tag='eval/acc', scalar_value=right_cnt / evaled_cnt, global_step=parameters['global_step'])
    writer.add_scalar(tag='eval/avgRscore', scalar_value=np.mean(np.array(sum_right_top1_score)),
                      global_step=parameters['global_step'])


if __name__ == '__main__':
    parms = dict()
    opt = Config()
    parms['options'] = opt
    parms['global_step'] = 0

    # ----- Device -----
    parms['device'] = setup_device(opt)

    # ----- Model -----
    parms['model'] = setup_network(opt, parms['device'])

    # ----- Logging -----
    parms['writer'] = SummaryWriter(log_dir='logging/')

    # ----- Load on device -----
    parms['model'].to(parms['device'])

    # ----- Test -----
    eval(parms)
    print('End')
