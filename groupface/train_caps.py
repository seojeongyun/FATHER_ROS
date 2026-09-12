import sys, os
import torch, cv2
import numpy as np
import time
from tqdm import tqdm

from models.GroupFace_caps import GroupFace_caps
from loss.loss import ArcMarginProduct
from loss.focal_loss import FocalLoss

from dataset.dataset import VGGdataset, totaldata, torch_loader
from Config.config import Config

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity

from test import resize, pad_to_square, load_gallery_probe
import matplotlib.pyplot as plt
from Config import config_test

from utilss.ema_single import ModelEMA
from utilss.ema_arc import ModelEMA_arc
from utilss.general import increment_name
from copy import deepcopy
import math
import random

torch.set_float32_matmul_precision('highest')
NCOLS = 100


def load_gallery(parameters):
    gallery_path = parameters['options'].gallery_path
    #
    gallery_files = []
    gallery_ids = []
    gallery_len = len(os.listdir(gallery_path))
    #
    for i, dir in enumerate(os.listdir(gallery_path)):
        for file in os.listdir(os.path.join(gallery_path, dir)):
            file_path = os.path.join(gallery_path, dir, file)
            gallery_files.append(file_path)
            gallery_ids.append(dir)
        sys.stdout.write("\r>> LoadGallery[{}/{}] ".format(i, gallery_len))
        sys.stdout.flush()
    print('\n')
    return gallery_files, gallery_ids


def load_probe(parameters):
    probe_path = parameters['options'].probe_path
    probe_files = []
    probe_ids = []

    probe_len = len(os.listdir(probe_path))
    for i, dir in enumerate(os.listdir(probe_path)):
        for file in os.listdir(os.path.join(probe_path, dir)):
            file_path = os.path.join(probe_path, dir, file)
            probe_files.append(file_path)
            probe_ids.append(dir)
        sys.stdout.write("\r>> LoadProbe[{}/{}] ".format(i, probe_len))
        sys.stdout.flush()
    print('\n')
    return probe_files, probe_ids


def setup_device(option):
    if len(opt.gpu_id) != 0:
        device = torch.device("cuda:{}".format(option.gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def setup_dataloader(option):
    if option['dataset_name'] == 'VGGFace2':
        dataset = VGGdataset(option['img_size'], option['path'], option['cache_file'])
    elif option['dataset_name'] == 'total_data':
        dataset = totaldata(option['img_size'], option['path'], option['cache_file'])

    loader = torch.utils.data.DataLoader(dataset, batch_size=option['batch_size'], shuffle=True,
                                         num_workers=option["num_workers"])
    return loader


def setup_network(option, device):
    model = GroupFace_caps(data_h=option.img_size, data_w=option.img_size,
                           capdimen=48, predcapdimen=64, numpricap=512,
                           num_final_cap=64,
                           feature_dim=option.feature_dim, groups=option.groups
                           ).to(device)
    return model


def setup_loss(option, device):
    if option.loss == 'focal_loss':
        loss = FocalLoss(gamma=2)
    else:
        loss = torch.nn.CrossEntropyLoss()
    return loss.to(device)


def setup_fc_metric(option, device):
    if option.fc_metric == 'arc':
        fc_metric = ArcMarginProduct(in_features=option.feature_dim, out_features=option.num_classes, s=30, m=0.5,
                                     device=device, easy_margin=option.easy_margin)
    else:
        raise NotImplementedError("[setup_fc_metric] NOT IMPLEMENTED..")
    return fc_metric.to(device)


def setup_optimizer(option, model, fc_metric):
    if option.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': fc_metric.parameters()}],
                                     lr=option.lr, weight_decay=option.weight_decay)
    else:
        raise NotImplementedError("Not Implemented {}".format(option.optimizer))
    return optimizer


def setup_scheduler(option, optimizer):
    if option.scheduler == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / option.epoch)) / 2) * (option.lrf - 1) + 1
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif option.scheduler == 'constant':
        lf = lambda x: 1.0
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif option.scheduler == 'cyclelr':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=option.lr_base, max_lr=option.lr_max,
                                                      step_size_up=option.T_up,
                                                      step_size_down=option.T_down,
                                                      gamma=option.lr_gamma, cycle_momentum=False,
                                                      mode='triangular2')
    elif option.scheduler == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=option.T_down, gamma=0.9)
    else:
        NotImplementedError('Not Implemented...{}'.format(option.scheduler))
    return scheduler


def load_weights(option, model, ema_model, fc_metric, ema_fc_metric, optimizer, best_top1_acc, device):
    if option.checkpoints_file is not None:
        file_path = os.path.join(option.checkpoints_save_path, option.checkpoints_file)
        assert os.path.exists(file_path), f'There is no weight file!!'

        print("Loading saved weights {}".format(option.checkpoints_file))
        ckpt = torch.load(file_path, map_location=device)
        #
        resume_state_dict = ckpt['model'].state_dict()  # checkpoint state_dict as FP32
        model.load_state_dict(resume_state_dict, strict=True)  # load
        #
        ema_model.ema.load_state_dict(ckpt['ema_model'].state_dict(), strict=True)
        #
        fc_metric.load_weights(ckpt['fc_metric'])
        #
        ema_fc_metric.load_weights(ckpt['ema_fc_metric'])
        #
        # optimizer.load_state_dict(ckpt['optimizer'])
        #
        # best_top1_acc = ckpt['best_top1_ac']

    return model, ema_model, fc_metric, ema_fc_metric, optimizer, best_top1_acc


def save_model(model, ema_model, fc_metric, ema_fc_metric,
               save_path, optimizer, option,
               best_top1_ac=0.0, best_top5_acc=0.0, best_harmonic_acc=0.0, prefix='last'):
    save_file = '{}res{}_group{}_featdim{}_top1_{}_harmonic_{}.pth'.format(prefix, option.resnet, option.groups,
                                                                           option.feature_dim,
                                                                           str(round(best_top1_ac, 3)).replace('.',
                                                                                                               '_'),
                                                                           str(round(best_harmonic_acc, 3)).replace('.', '_'),
                                                                           option.img_size)
    path = os.path.join(save_path, save_file)
    torch.save({'model': deepcopy(model),
                'ema_model': deepcopy(ema_model.ema),
                'fc_metric': {
                    'weight': fc_metric.weight,
                    'in_features': fc_metric.in_features,
                    'out_features': fc_metric.out_features,
                    'm': fc_metric.m,
                    's': fc_metric.s,
                    'easy_margin': fc_metric.easy_margin,
                },
                'best_top1_ac': best_top1_ac,
                'best_top5_acc': best_top5_acc,
                'best_harmonic_acc': best_harmonic_acc,
                'ema_fc_metric': deepcopy(ema_fc_metric),
                'optimizer': optimizer.state_dict()}, path)


def training(parameters):
    # --------------training parameter-------------------------
    option = parameters['options']
    device = parameters['device']
    scheduler = parameters['scheduler']
    model = parameters['model']
    ema_model = parameters['ema_model']
    fc_metric = parameters['fc_metric']
    ema_fc_metric = parameters['ema_fc_metric']
    optimizer = parameters['optimizer']
    train_loader = parameters['train_loader']
    criterion = parameters['criterion']
    writer = parameters['writer']
    save_path = parameters['save_path']
    accumulate = max(1, round(64 / parameters['options'].batch_size))
    last_opt_step = 0

    # --------------test parameter-------------------------
    t_parmeter = dict()

    t_parmeter['device'] = device
    t_parmeter['model'] = model.eval()
    t_parmeter['options'] = config_test.Config
    t_parmeter['writer'] = writer
    t_parmeter['global_step'] = 0
    t_parmeter['ema_model'] = ema_model

    # -------
    t_parmeter['best_eval'] = 0.0

    #
    best_top1_acc = parms['best_top1_acc']
    best_top5_acc = 0
    best_harmonic_mean = 0.0
    #
    eval(t_parmeter)
    #
    for i in range(option.epoch):
        model.train()
        loss_sum = 0.0
        cnt_right = 0.0
        cnt_total = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, ascii=True, ncols=NCOLS)
        for curr_step, data in pbar:
            parameters['global_step'] = (i * len(train_loader) + (curr_step + 1))
            t_parmeter['global_step'] = (i * len(train_loader) + (curr_step + 1))
            #
            img, file_path, id, label = data
            img = img.to(device)
            label = label.to(device).long()

            group_inter, final, group_prob, group_label = model(img)
            output = fc_metric(final, label)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            if parameters['global_step'] - last_opt_step >= accumulate:
                optimizer.step()
                if ema_model:
                    ema_model.update(model)
                    ema_fc_metric.update(fc_metric)
                last_opt_step = parameters['global_step']

            # Accumulate losses
            loss_sum += float(loss)

            # For accuracy
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            cnt_right += np.sum((output == label).astype(int))
            cnt_total += label.shape[0]

            # Print log data
            if parameters['global_step'] % option.print_freq == 0:
                writer.add_scalar(tag='train/loss', scalar_value=loss, global_step=parameters['global_step'])
                # pbar.set_postfix(loss=loss.detach().cpu().numpy().item(), acc=-1)

            if parameters['global_step'] % option.eval_interval == 0:
                curr_top1_acc, curr_top5_acc = eval(t_parmeter)
                model.train()
                #
                writer.add_scalar(tag='train/classification_acc', scalar_value=cnt_right / cnt_total,
                                  global_step=parameters['global_step'])
                pbar.set_postfix(loss=loss.detach().cpu().numpy().item(), acc=(cnt_right / cnt_total))
                cnt_right = 0.0
                cnt_total = 0.0

                curr_harmonic_mean = 2 * (curr_top1_acc * curr_top5_acc) / (curr_top1_acc + curr_top5_acc)
                if best_harmonic_mean < curr_harmonic_mean:
                    best_harmonic_mean = curr_harmonic_mean
                    best_top1_acc = curr_top1_acc
                    best_top5_acc = curr_top5_acc
                    save_model(model, ema_model, fc_metric, ema_fc_metric,
                               save_path, optimizer, option,
                               best_top1_acc, best_top5_acc, best_harmonic_mean, 'best_')
                    print('\n [save] top1_acc: {}, top5_acc: {}', curr_top1_acc, curr_top5_acc)
        #
        scheduler.step()
        save_model(model, ema_model, fc_metric, ema_fc_metric,
                   save_path, optimizer, option, prefix='last_')
        writer.add_scalar(tag='train/lr', scalar_value=scheduler.get_lr()[0], global_step=i)
        print("\nEpoch {} Trained Loss_Sum:{:.5f}\n".format(i, loss_sum / float(len(train_loader))))


def eval(parameters):
    writer = parameters['writer']
    device = parameters['device']
    #
    model = parameters['ema_model'].ema
    #
    model.eval().to(device)
    #
    print('\n[eval] Get gallery and probe images...')
    gallery_files, gallery_ids, probe_files, probe_ids = load_gallery_probe(parameters)
    g_id_unique = np.unique(gallery_ids)
    #
    set_g_index = list(range(len(gallery_files)))
    split_index = []
    tmp = []
    for i in set_g_index:
        if i != 0 and i % 32 == 0:
            split_index.append(tmp)
            tmp = []
        tmp.append(i)
        if i == set_g_index[-1]:
            split_index.append(tmp)

    gallery_feats = []
    print('\n[eval] Obtain feature vectors of gallery images...')
    for idxes_batch in tqdm(split_index, total=len(split_index)):
        batch = []
        for i in idxes_batch:
            gallery_file, _ = pad_to_square(torch.Tensor(cv2.imread(gallery_files[i])).permute(2, 0, 1))
            gallery_file = resize(gallery_file, parameters["options"].img_size)
            gallery_file = np.array(gallery_file.permute(1, 2, 0)).astype(np.uint8)
            batch.append(torch_loader(gallery_file, dimension=parameters['options'].img_size).unsqueeze(0).to(device))
        batch = torch.cat(batch, dim=0)
        _, final, _, _ = model(batch)
        gallery_feat = final / torch.norm(final, p=2, dim=1, keepdim=True)
        gallery_feat = gallery_feat.detach().cpu().numpy()
        gallery_feats.append(gallery_feat)
    gallery_feats = np.concatenate(gallery_feats, axis=0)
    #
    evaled_cnt = 0.0
    top1_cnt = dict()
    top3_cnt = dict()
    top5_cnt = dict()
    top1_mean_cnt = dict()
    top3_mean_cnt = dict()
    top5_mean_cnt = dict()
    #
    gallery_ids_np = np.expand_dims(np.array(gallery_ids), axis=0)
    num_gallery_imgs = np.linspace(10, 120, 12, dtype=np.int16)
    for num_imgs in num_gallery_imgs:
        top1_cnt[num_imgs] = 0.0
        top3_cnt[num_imgs] = 0.0
        top5_cnt[num_imgs] = 0.0
        top1_mean_cnt[num_imgs] = 0.0
        top3_mean_cnt[num_imgs] = 0.0
        top5_mean_cnt[num_imgs] = 0.0
    #
    print('\n[eval] Calculate accuracy...')
    pbar = tqdm(zip(probe_files, probe_ids), total=len(probe_ids), leave=True, mininterval=0.1, ascii=True, ncols=NCOLS)
    for probe_file, GT_id in pbar:
        evaled_cnt += 1
        probe_file, pad = pad_to_square(torch.Tensor(cv2.imread(probe_file)).permute(2, 0, 1))
        probe_file = resize(probe_file, parameters["options"].img_size)
        probe_file = np.array(probe_file.permute(1, 2, 0)).astype(np.uint8)
        # Obtain a feature vector of a probe face
        _, final, _, _ = model(
            torch_loader(probe_file, dimension=parameters['options'].img_size).unsqueeze(0).to(device))
        probe_feat = final / torch.norm(final, p=2, keepdim=False)
        probe_feat = probe_feat.detach().cpu().reshape(1, parameters['options'].feature_dim).numpy()
        #
        scores = cosine_similarity(probe_feat, gallery_feats)
        #
        for num_imgs in num_gallery_imgs:
            #
            selected_idxes = []
            selected_g_ids = []
            mean_score_per_g_id = []
            mean_g_id = []
            for g_id in g_id_unique:
                idx_g_id = np.where(gallery_ids_np[0] == g_id)[0]
                try:
                    idxes_ = random.sample(list(idx_g_id), num_imgs)
                    actual_num_imgs = len(idxes_)
                except:
                    idxes_ = list(idx_g_id)
                    actual_num_imgs = len(idxes_)
                #
                selected_idxes.append(idxes_)
                selected_g_ids.append([g_id] * actual_num_imgs)
                #
                mean_score_per_g_id.append(np.average(scores[0, idxes_]))
                mean_g_id.append(g_id)
            #
            selected_idxes = list(np.concatenate(selected_idxes, axis=0))
            selected_g_ids = list(np.concatenate(selected_g_ids, axis=0))
            selected_scores = scores[0, selected_idxes]
            #
            sort_scores_idx = np.argsort(selected_scores)
            sorted_gallery_ids = np.array(selected_g_ids)[sort_scores_idx]
            #
            top1_id = [sorted_gallery_ids[-1]]
            top3_ids = list(sorted_gallery_ids[-3:])
            top5_ids = list(sorted_gallery_ids[-5:])
            top1_cnt[num_imgs] += 1 if GT_id in top1_id else 0
            top3_cnt[num_imgs] += 1 if GT_id in top3_ids else 0
            top5_cnt[num_imgs] += 1 if GT_id in top5_ids else 0
            #
            sort_mean_scores_idx = np.argsort(mean_score_per_g_id)
            sort_mean_g_id = np.array(mean_g_id)[sort_mean_scores_idx]
            #
            top1_id = [sort_mean_g_id[-1]]
            top3_ids = list(sort_mean_g_id[-3:])
            top5_ids = list(sort_mean_g_id[-5:])
            top1_mean_cnt[num_imgs] += 1 if GT_id in top1_id else 0
            top3_mean_cnt[num_imgs] += 1 if GT_id in top3_ids else 0
            top5_mean_cnt[num_imgs] += 1 if GT_id in top5_ids else 0

    for num_imgs in num_gallery_imgs:
        writer.add_scalar(tag='eval/top1_acc/{}'.format(num_imgs),
                          scalar_value=top1_cnt[num_imgs] / evaled_cnt,
                          global_step=parameters['global_step'])
        writer.add_scalar(tag='eval/top3_acc/{}'.format(num_imgs),
                          scalar_value=top3_cnt[num_imgs] / evaled_cnt,
                          global_step=parameters['global_step'])
        writer.add_scalar(tag='eval/top5_acc/{}'.format(num_imgs),
                          scalar_value=top5_cnt[num_imgs] / evaled_cnt,
                          global_step=parameters['global_step'])
        writer.add_scalar(tag='eval/top1_acc (mean)/{}'.format(num_imgs),
                          scalar_value=top1_mean_cnt[num_imgs] / evaled_cnt,
                          global_step=parameters['global_step'])
        writer.add_scalar(tag='eval/top3_acc (mean)/{}'.format(num_imgs),
                          scalar_value=top3_mean_cnt[num_imgs] / evaled_cnt,
                          global_step=parameters['global_step'])
        writer.add_scalar(tag='eval/top5_acc (mean)/{}'.format(num_imgs),
                          scalar_value=top5_mean_cnt[num_imgs] / evaled_cnt,
                          global_step=parameters['global_step'])
    return top1_cnt[120] / evaled_cnt, top5_cnt[120] / evaled_cnt


if __name__ == '__main__':
    from setproctitle import *

    setproctitle('HRLEE: GROUPFACE-YOLO')

    parms = dict()
    opt = Config()
    parms['options'] = opt
    parms['global_step'] = 0

    # ----- Device -----
    parms['device'] = setup_device(opt)

    # ----- Dataset -----
    parms['train_loader'] = setup_dataloader(
        {'path': opt.train_path, 'cache_file': opt.cache_file, 'batch_size': opt.batch_size, 'img_size': opt.img_size,
         'num_workers': opt.num_workers, 'dataset_name': opt.dataset_name})

    # ----- Model -----
    parms['model'] = setup_network(opt, parms['device']).to(parms['device'])
    parms['ema_model'] = ModelEMA(parms['model'])
    parms['fc_metric'] = setup_fc_metric(opt, parms['device']).to(parms['device'])
    parms['ema_fc_metric'] = ModelEMA_arc(parms['fc_metric'])

    # ----- Criterion -----
    parms['criterion'] = setup_loss(opt, parms['device'])

    # ----- Logging -----
    log_path = str(increment_name('logging/' + opt.log_dir_name))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    parms['writer'] = SummaryWriter(log_dir=log_path)

    save_path = str(increment_name(os.path.join(opt.checkpoints_save_path + opt.dataset_name.lower())))
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    parms['save_path'] = save_path
    parms['best_top1_acc'] = 0.0

    # ----- Optimizer -----
    parms['optimizer'] = setup_optimizer(opt, parms['model'], parms['fc_metric'])
    parms['scheduler'] = setup_scheduler(opt, parms['optimizer'])

    # ----- Load pre-trained parameters -----
    (parms['model'], parms['ema_model'],
     parms['fc_metric'], parms['ema_fc_metric'],
     parms['optimizer'], parms['best_top1_acc']) = \
        load_weights(opt,
                     parms['model'], parms['ema_model'],
                     parms['fc_metric'], parms['ema_fc_metric'],
                     parms['optimizer'], parms['best_top1_acc'], parms['device'])

    # ----- Training -----
    training(parms)
    print('End')
