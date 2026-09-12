import sys, os
import torch, cv2
import numpy as np
import time
from tqdm import tqdm

from models.GroupFace import GroupFace
from loss.loss import ArcMarginProduct
from loss.focal_loss import FocalLoss

from dataset.dataset import VGGdataset, totaldata, torch_loader
from Config.config import Config

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics.pairwise import cosine_similarity

from test import resize, pad_to_square, load_gallery_probe_dict
from Config import config_test

from utilss.ema_single import ModelEMA
from utilss.ema_arc import ModelEMA_arc
from utilss.general import increment_name
from copy import deepcopy
import math

torch.set_float32_matmul_precision('high')
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
    model = GroupFace(resnet=option.resnet, feature_dim=option.feature_dim, groups=option.groups).to(device)
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
    if option.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': fc_metric.parameters()}],
                                    lr=option.lr, weight_decay=option.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': fc_metric.parameters()}],
                                     lr=option.lr, weight_decay=option.weight_decay)
    return optimizer


def setup_scheduler(option, optimizer):
    if option.scheduler == 'cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / option.epoch)) / 2) * (option.lrf - 1) + 1
    elif option.scheduler == 'constant':
        lf = lambda x: 1.0
    else:
        NotImplementedError('Not Implemented...{}'.format(option.scheduler))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
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
        ema_model.ema.load_state_dict(ckpt['ema_model'].state_dict())
        #
        fc_metric.load_weights(ckpt['fc_metric'])
        #
        ema_fc_metric.load_weights(ckpt['ema_fc_metric'])
        #
        optimizer.load_state_dict(ckpt['optimizer'])
        #
        best_top1_acc = ckpt['best_top1_acc']

    return model, ema_model, fc_metric, ema_fc_metric, optimizer, best_top1_acc


def save_model(model, ema_model, fc_metric, ema_fc_metric,
               save_path, optimizer, option, best_top1_ac, prefix):
    save_file = '{}res{}_group{}_featdim{}_top1_{}.pth'.format(prefix, option.resnet, option.groups,
                                                               option.feature_dim, str(best_top1_ac).replace('.', '_'))
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
    for i in range(option.epoch):
        model.train()
        loss_sum = 0.0
        cnt_right = 0.0
        cnt_total = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, mininterval=0.1, ascii=True,
                    ncols=NCOLS)
        for curr_step, data in pbar:
            pbar.set_description(f'Epoch: {i + 1}')
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
                pbar.set_postfix(loss=loss.detach().cpu().numpy().item(), acc=-1)

            if parameters['global_step'] % option.eval_interval == 0:
                curr_top1_acc, curr_top5_acc = eval_various_num_thres(t_parmeter)
                model.train()
                #
                writer.add_scalar(tag='train/classification_acc', scalar_value=cnt_right / cnt_total,
                                  global_step=parameters['global_step'])
                pbar.set_postfix(loss=loss.detach().cpu().numpy().item(), acc=(cnt_right / cnt_total))
                cnt_right = 0.0
                cnt_total = 0.0

                if best_top1_acc < curr_top1_acc:
                    best_top1_acc = curr_top5_acc
                    save_model(model, ema_model, fc_metric, ema_fc_metric,
                               save_path, optimizer, option, best_top1_acc, 'best_')
                    print('\n [save] top1_acc: {}, top5_acc: {}', curr_top1_acc, curr_top5_acc)
        #
        scheduler.step()
        save_model(model, ema_model, fc_metric, ema_fc_metric,
                   save_path, optimizer, option, 0.0, 'last_')
        writer.add_scalar(tag='train/lr', scalar_value=scheduler.get_last_lr()[0], global_step=i)
        print("\nEpoch {} Trained Loss_Sum:{:.5f}\n".format(i, loss_sum / float(len(train_loader))))


def eval_various_num_thres(parameters):
    writer = parameters['writer']
    device = parameters['device']
    #
    model = parameters['ema_model'].ema
    #
    model.eval().to(device)
    #
    num_thres_set = np.linspace(10, 120, 12)
    #
    scores_per_thres = dict()
    for num_thres in tqdm(num_thres_set, total=len(num_thres_set)):
        parameters['options'].num_thres = num_thres
        scores_per_thres[num_thres] = dict()
        #
        print('======== num_thres: {} ========'.format(num_thres))
        print('\n[eval] Get gallery and probe images...')
        gallery_folders, probe_folders = load_gallery_probe_dict(parameters)
        #
        gallery_feats = dict()
        print('\n[eval] Obtain feature vectors of gallery images...')
        for gallery_name in tqdm(gallery_folders, total=len(gallery_folders), leave=True, mininterval=0.1, ascii=True,
                                 ncols=NCOLS):
            # Obtain a feature vector of a gallery face
            gallery_files = gallery_folders[gallery_name]
            gallery_feats[gallery_name] = []
            for img_name in gallery_files:
                img_path = gallery_files[img_name]['path']
                img, pad = pad_to_square(torch.Tensor(cv2.imread(img_path)).permute(2, 0, 1))
                img = resize(img, parameters["options"].img_size)
                img = np.array(img.permute(1, 2, 0)).astype(np.uint8)
                _, final, _, _ = model(
                    torch_loader(img, dimension=parameters['options'].img_size).unsqueeze(0).to(device))
                gallery_feat = final / torch.norm(final, p=2, keepdim=False)
                gallery_feat = gallery_feat.detach().cpu().reshape(1, parameters['options'].feature_dim).numpy()
                gallery_feats[gallery_name].append(gallery_feat)
            gallery_feats[gallery_name] = np.stack(gallery_feats[gallery_name], axis=0).squeeze(axis=1)
        #
        evaled_cnt = 0.0
        top1_cnt = 0.0
        top5_cnt = 0.0
        print('\n[eval] Calculate accuracy...\n\n')
        for probe_id in tqdm(probe_folders, total=len(probe_folders), leave=True, mininterval=0.1, ascii=True,
                             ncols=NCOLS):
            probe_files = probe_folders[probe_id]
            #
            for img_name in probe_files:
                evaled_cnt += 1
                img_path = probe_files[img_name]['path']
                img, pad = pad_to_square(torch.Tensor(cv2.imread(img_path)).permute(2, 0, 1))
                img = resize(img, parameters["options"].img_size)
                img = np.array(img.permute(1, 2, 0)).astype(np.uint8)
                # Obtain a feature vector of a probe face
                _, final, _, _ = model(
                    torch_loader(img, dimension=parameters['options'].img_size).unsqueeze(0).to(device))
                probe_feat = final / torch.norm(final, p=2, keepdim=False)
                probe_feat = probe_feat.detach().cpu().reshape(1, parameters['options'].feature_dim).numpy()
                #
                scores = []
                g_ids = []
                for feat_per_gallery in gallery_feats:
                    g_feats = gallery_feats[feat_per_gallery]
                    scores.append(cosine_similarity(probe_feat, g_feats).mean())
                    g_ids.append(feat_per_gallery)
                #
                sort_scores_idx = np.argsort(scores)
                #
                gallery_ids_np = np.array(g_ids)
                sorted_gallery_ids = gallery_ids_np[sort_scores_idx]
                #
                top1_id = [sorted_gallery_ids[-1]]
                top5_ids = list(sorted_gallery_ids[-5:])
                #
                top1_cnt += 1 if probe_id in top1_id else 0
                top5_cnt += 1 if probe_id in top5_ids else 0

        scores_per_thres[num_thres] = {'top1_acc': top1_cnt / evaled_cnt, 'top5_acc': top5_cnt / evaled_cnt}

    for num_thres in list(scores_per_thres.keys()):
        writer.add_scalar(tag='eval/top1_acc@{}'.format(num_thres),
                          scalar_value=scores_per_thres[num_thres]['top1_acc'], global_step=parameters['global_step'])
        writer.add_scalar(tag='eval/top5_acc@{}'.format(num_thres),
                          scalar_value=scores_per_thres[num_thres]['top5_acc'], global_step=parameters['global_step'])

    return scores_per_thres


if __name__ == '__main__':
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
