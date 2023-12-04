import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import dot
from numpy.linalg import norm

import torch
import torch.nn.functional as F


def load_gallery_probe(gallery_config, probe_path=None, split=True):
    import os
    import random
    #
    gallery_path = gallery_config['path']
    num_thres = gallery_config['max_num_imgs']
    max_num_gallery_imgs = gallery_config['max_num_gallery_imgs']
    #
    g_files = []
    p_files = []
    #
    for i, dir_name in enumerate(os.listdir(gallery_path)):
        files_in_dir = os.listdir(os.path.join(gallery_path, dir_name))
        random.shuffle(files_in_dir)
        #
        if split:
            if len(files_in_dir) > num_thres + max_num_gallery_imgs:
                for _ in range(num_thres):
                    p_files.append((dir_name, os.path.join(gallery_path, dir_name, files_in_dir.pop())))
                for _ in range(max_num_gallery_imgs):
                    g_files.append((dir_name, os.path.join(gallery_path, dir_name, files_in_dir.pop())))
            else:
                if len(files_in_dir) > max_num_gallery_imgs:
                    for _ in range(max_num_gallery_imgs):
                        g_files.append((dir_name, os.path.join(gallery_path, dir_name, files_in_dir.pop())))
                    for _ in range(len(files_in_dir)):
                        p_files.append((dir_name, os.path.join(gallery_path, dir_name, files_in_dir.pop())))
                else:
                    for _ in range(int(0.8 * len(files_in_dir))):
                        g_files.append((dir_name, os.path.join(gallery_path, dir_name, files_in_dir.pop())))
                    for _ in range(len(files_in_dir)):
                        p_files.append((dir_name, os.path.join(gallery_path, dir_name, files_in_dir.pop())))
        else:
            for _ in range(len(files_in_dir)):
                g_files.append((dir_name, os.path.join(gallery_path, dir_name, files_in_dir.pop())))
    #
    if split:
        print('\033[92m' + 'LOADING GALLERY IMAGES COMPLETED!!')
        return g_files, p_files
    else:
        assert probe_path is not None
        for i, dir_name in enumerate(os.listdir(probe_path)):
            files_in_dir = os.listdir(os.path.join(probe_path, dir_name))
            for _ in range(len(files_in_dir)):
                p_files.append((dir_name, os.path.join(probe_path, dir_name, files_in_dir.pop())))
        print('\033[92m' + 'LOADING GALLERY IMAGES COMPLETED!!')
        return g_files, p_files


class face_similarity():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['network_cfg']['device']
        # LOAD NETWORK
        self.model = self.get_model(cfg['network_cfg'])
        #
        self.g_vectors = None
        self.g_ids = None
        self.g_files = None
    	

    def find_top5_face_ids(self, probe_img, NUM_TOP_IMAGE=5, verbose=False):
        assert len(probe_img.shape) == 3
        #
        if verbose:
            print('\033[92m' + 'CALCULATING COSINE SIMILARITY BETWEEN PROBE and GALLERIES')
        #
        probe_img = (self.resize(self.pad_to_square(probe_img)[0], self.cfg['imgsize']).float() / 255.).unsqueeze(dim=0)
        probe_feat = self.calc_feat_vec(probe_img.to(self.device))
        #
        scores = torch.tensor(cosine_similarity(probe_feat, self.g_vectors))
        #
        _, sorted_idx = torch.sort(scores, descending=True)
        #
        top5_g_imgs = []
        for idx in list(sorted_idx[0][:NUM_TOP_IMAGE].numpy()):
            img_ = torch.Tensor(cv2.imread(self.g_files[idx][1])).permute(2, 0, 1)
            img_ = self.resize(self.pad_to_square(img_)[0], int((self.cfg['imgsize'] * 2) / 5)).float()
            top5_g_imgs.append(img_)
        top5_g_imgs = torch.cat(top5_g_imgs, dim=2)
        top5_g_imgs = torch.cat([
            torch.zeros(3, top5_g_imgs.shape[1], (self.cfg['imgsize'] * 3) - top5_g_imgs.shape[2]),
            top5_g_imgs],
            dim=2)
        #
        return ([self.g_ids[k] for k in list(sorted_idx[0, :NUM_TOP_IMAGE].numpy())],
                [scores[0, k] for k in list(sorted_idx[0, :NUM_TOP_IMAGE].numpy())], probe_img, top5_g_imgs)

    def calc_feat_vec(self, inputs):
        _, final, _, _ = self.model(inputs)
        feat_vec = (final / torch.norm(final, p=2, dim=1, keepdim=True)).detach().cpu()
        return feat_vec

    def get_gallery_feat_vecs(self, files):
        from tqdm import tqdm
        from copy import deepcopy
        #
        files_idxes = list(range(len(files)))
        remaining_idxes = []
        if len(files) % 32 != 0:
            for _ in range(len(files_idxes[-(len(files) % 32):])):
                remaining_idxes.append(files_idxes.pop())
        files_idxes = list(torch.tensor(files_idxes).view(-1, 32))
        if len(files) % 32 != 0:
            files_idxes.append(deepcopy(remaining_idxes))
        del remaining_idxes
        #
        gallery_feats = []
        gallery_ids = []
        for idxes in tqdm(files_idxes, total=len(files_idxes), leave=True):
            batch_imgs = []
            for idx in list(idxes):
                gallery_ids.append(files[idx][0])
                img = torch.Tensor(cv2.cvtColor(cv2.imread(files[idx][1]), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
                img = self.resize(self.pad_to_square(img)[0], self.cfg['imgsize'])
                img = img.float() / 255.
                batch_imgs.append(img.unsqueeze(0).to(self.device))
            batch = torch.cat(batch_imgs, dim=0)

            gallery_feat = self.calc_feat_vec(batch)
            gallery_feats.append(gallery_feat.detach().cpu())
        gallery_feats = torch.cat(gallery_feats, dim=0)
        print('\033[92m' + 'CALCULATION of IMAGES in GALLERY COMPLETED!!')
        return gallery_feats, gallery_ids

    def make_gallery_feats_database(self, g_files):
        self.g_files = g_files
        self.g_vectors, self.g_ids = self.get_gallery_feat_vecs(g_files)

    @staticmethod
    def get_model(cfg_net):
        import os
        #
        assert cfg_net['weight'] is not None, f'SHOULD SET THE WEIGHT PATH...'
        assert os.path.exists(cfg_net['weight']), f'THERE IS NO WEIGHT...'
        #
        ckpt = torch.load(cfg_net['weight'], map_location=cfg_net['device'])
        model = ckpt['ema_model'].to(cfg_net['device'])
        print('\033[92m' + 'LOADING MODEL COMPLETED!!')
        return model

    @staticmethod
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

    @staticmethod
    def resize(image, size):
        return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

    @staticmethod
    def view_result(img, probe_id, top5_ids, top5_scores, top5_g_imgs):
        resized_probe_img = img.squeeze(dim=0).numpy().transpose(1, 2, 0)
        result_area = torch.zeros(resized_probe_img.shape[0],
                                  resized_probe_img.shape[1] * 2,
                                  resized_probe_img.shape[2]).numpy()
        #
        result_area = cv2.putText(result_area, "Target ID: " + str(probe_id), (20, 40),
                                  cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 213, 0), thickness=1)
        #
        bar_height = 80
        bar_start_h = 160
        bar_end_h = bar_start_h - bar_height
        bar_width = 20
        bar_start_w = 10
        bar_end_w = 10 + bar_width
        left_margin = 50
        for top5_i in range(len(top5_ids)):
            #if top5_ids[top5_i] == probe_id:
            color = (115, 254, 255)
            #else:
                #color = (255, 0, 0)
            cv2.putText(result_area, top5_ids[top5_i],
                        (left_margin + 80 * top5_i, bar_start_h + 20), cv2.FONT_HERSHEY_COMPLEX,
                        0.3, color, thickness=1)
            cv2.rectangle(result_area,
                          (left_margin + 80 * top5_i + bar_start_w, bar_end_h),
                          (left_margin + 80 * top5_i + bar_end_w, bar_start_h), color, thickness=2)
            score_bar_h = int(torch.ceil(bar_height * top5_scores[top5_i]).numpy())
            cv2.rectangle(result_area,
                          (left_margin + 80 * top5_i + bar_start_w, bar_start_h - score_bar_h),
                          (left_margin + 80 * top5_i + bar_end_w, bar_start_h), color, thickness=-1)
            cv2.putText(result_area, str(torch.round(top5_scores[top5_i], decimals=2).numpy()),
                        (left_margin + 80 * top5_i, bar_start_h + 40), cv2.FONT_HERSHEY_COMPLEX,
                        0.3, color, thickness=1)
        #
        cv2.putText(result_area, "Pred ID",
                    (0, bar_start_h + 20), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (0, 255, 0), thickness=1)
        cv2.putText(result_area, "Score",
                    (0, bar_start_h + 40), cv2.FONT_HERSHEY_COMPLEX,
                    0.3, (0, 255, 0), thickness=1)
        result_area = cv2.putText(result_area, "Predicted IDs within top 5",
                                  (30, bar_end_h - 10), cv2.FONT_HERSHEY_COMPLEX,
                                  0.4, (0, 255, 0), thickness=1)
        result_area = cv2.cvtColor(result_area, cv2.COLOR_RGB2BGR)
        #
        result_img = torch.cat([torch.tensor(resized_probe_img * 255).type(torch.uint8),
                                torch.tensor(result_area).type(torch.uint8)], dim=1)
        result_img = torch.cat([result_img, top5_g_imgs.permute(1, 2, 0).type(torch.uint8)], dim=0)
        return result_img.numpy()
