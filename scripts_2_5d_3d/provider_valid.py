import os
import cv2
import h5py
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.seg_util import genSegMalis
from utils.gen_affs import gen_affs, gen_affs_3d
from utils.weights import weight_binary_ratio

class Provider_valid(Dataset):
    def __init__(self, cfg, valid_data=None, test=False, test_split=None, direc='x'):
        # basic settings
        self.cfg = cfg
        self.mode = cfg.TRAIN.mode
        self.shift = cfg.DATA.shift
        self.if_dilate = cfg.DATA.if_dilate
        self.if_bg = cfg.DATA.if_bg
        self.test = test
        self.direc = direc
        if valid_data is not None:
            valid_dataset_name = valid_data
        else:
            try:
                valid_dataset_name = cfg.DATA.valid_dataset
                print('valid on valid dataset!')
            except:
                valid_dataset_name = cfg.DATA.dataset_name
                print('valid on train dataset!')

        # training dataset files (h5), may contain many datasets
        if valid_dataset_name == 'cremiA':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs_interp.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif valid_dataset_name == 'cremiB':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs_interp.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif valid_dataset_name == 'cremiC':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs_interp.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif valid_dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs_interp.h5']
            self.train_labels = ['cremiC_labels.h5']
            # self.sub_path = 'cremi'
            # self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
            # self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif valid_dataset_name == 'isbi':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_inputs.h5']
            self.train_labels = ['isbi_labels.h5']
        elif valid_dataset_name == 'ac3':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']
        elif valid_dataset_name == 'ac4':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
        elif valid_dataset_name == 'fib':
            self.sub_path = 'fib'
            self.train_datasets = ['fib_inputs.h5']
            self.train_labels = ['fib_labels.h5']
        else:
            raise AttributeError('No this dataset type!')

        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        if test_split is None:
            self.test_split = cfg.DATA.test_split
        else:
            self.test_split = test_split
        if valid_dataset_name == 'isbi_test' or valid_dataset_name == 'ac3':
            self.test_split = 100
        print('the number of valid(test) = %d' % self.test_split)

        # load dataset
        self.dataset = []
        self.labels = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()
            data = data[-self.test_split:]
            self.dataset.append(data)

            # load labels
            f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
            label = f_label['main'][:]
            f_label.close()
            label = label[-self.test_split:]
            self.labels.append(label)
        self.origin_data_shape = list(self.dataset[0].shape)

        # generate gt affinity
        self.gt_affs = []
        for k in range(len(self.labels)):
            temp = self.labels[k].copy()
            if self.if_dilate:
                temp = genSegMalis(temp, 1)
            # self.gt_affs.append(seg_to_affgraph(temp, mknhood3d(1), pad='replicate').astype(np.float32))
            self.gt_affs.append(gen_affs_3d(temp, shift=1, padding=True, background=self.if_bg))

        self.num_per_dataset = self.test_split
        self.iters_num = self.num_per_dataset * len(self.dataset)

    def __getitem__(self, index):
        pos_data = index // self.num_per_dataset

        if self.mode == 'x-y-z' or self.mode == 'x-y-z-2' or self.mode == 'z':
            if index == 0:
                img1 = self.dataset[pos_data][index].copy()
                lb1 = self.labels[pos_data][index].copy()
            else:
                img1 = self.dataset[pos_data][index-1].copy()
                lb1 = self.labels[pos_data][index-1].copy()
            img2 = self.dataset[pos_data][index].copy()
            lb2 = self.labels[pos_data][index].copy()
            affs0 = gen_affs(lb1, lb2, dir=0, shift=self.shift, padding=True, background=self.if_bg)
            if self.mode == 'x-y-z':
                affs1_2 = gen_affs(lb1, None, dir=2, shift=self.shift)
                affs1_1 = gen_affs(lb1, None, dir=1, shift=self.shift)
                affs2_2 = gen_affs(lb2, None, dir=2, shift=self.shift)
                affs2_1 = gen_affs(lb2, None, dir=1, shift=self.shift)
                affs = np.stack([affs1_2, affs1_1, affs2_2, affs2_1, affs0], axis=0)
            elif self.mode == 'x-y-z-2':
                affs1 = gen_affs(lb2, None, dir=1, shift=self.shift, padding=True, background=self.if_bg)
                affs2 = gen_affs(lb2, None, dir=2, shift=self.shift, padding=True, background=self.if_bg)
                affs = np.stack([affs0, affs1, affs2], axis=0)
            else:
                affs = affs0[np.newaxis, ...]
        elif self.mode == 'x-y':
            img1 = self.dataset[pos_data][index].copy()
            lb1 = self.labels[pos_data][index].copy()
            h, w = img1.shape
            img2 = img1.copy()
            if self.direc == 'x':
                img2[:, self.shift:] = img1[:, :w-self.shift]
                affs = gen_affs(lb1, None, dir=2, shift=self.shift)
            elif self.direc == 'y':
                img2[self.shift:, :] = img1[:h-self.shift, :]
                affs = gen_affs(lb1, None, dir=1, shift=self.shift)
            else:
                raise NotImplementedError
            affs = affs[np.newaxis, ...]
        else:
            raise NotImplementedError
        imgs = np.stack([img1, img2], axis=0)
        imgs = imgs.astype(np.float32) / 255.0
        weight0 = weight_binary_ratio(affs[0])
        weight1 = weight_binary_ratio(affs[1])
        weight2 = weight_binary_ratio(affs[2])
        weightmap = np.stack([weight0, weight1, weight2], axis=0)
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        affs = np.ascontiguousarray(affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        return imgs, affs, weightmap

    def __len__(self):
        return self.iters_num

    def get_gt_affs(self, num_data=0):
        return self.gt_affs[num_data].copy()

    def get_gt_lb(self, num_data=0):
        lbs = self.labels[num_data].copy()
        return lbs

    def get_raw_data(self, num_data=0):
        out = self.dataset[num_data].copy()
        return out


if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    import time
    import torch
    from utils.show import show_one
    from sklearn.metrics import f1_score

    seed = 555
    np.random.seed(seed)
    random.seed(seed)
    cfg_file = 'seg_onlylb_suhu_wbce_lr01_snemi3d_data25.yaml'
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict( yaml.load(f) )
    
    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    data = Provider_valid(cfg)
    dataloader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True)
    
    gt_affs = data.get_gt_affs()
    pred = np.random.random(tuple(gt_affs.shape)).astype(np.float32)
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    gt_affs = gt_affs.astype(np.uint8)
    pred = pred.astype(np.uint8)
    gt_affs = gt_affs.flatten()
    pred = pred.flatten()
    f1 = f1_score(1 - gt_affs, 1- pred)
    print(f1)

    # t = time.time()
    # for k, batch in enumerate(dataloader, 0):
    #     inputs, target, wrightmap = batch
    #     target = target.data.numpy()
    #     data.add_vol(target[0])
    # out_affs = data.get_results()
    # for k in range(out_affs.shape[1]):
    #     affs_xy = out_affs[2, k]
    #     affs_xy = (affs_xy * 255).astype(np.uint8)
    #     Image.fromarray(affs_xy).save(os.path.join(out_path, str(k).zfill(4)+'.png'))
    #     # print('single cost time: ', time.time()-t1)
    #     # tmp_data = np.squeeze(tmp_data)
    #     # if cfg.MODEL.model_type == 'mala':
    #     #     tmp_data = tmp_data[14:-14,106:-106,106:-106]
    #     # affs_xy = affs[2]
    #     # weightmap_xy = weightmap[2]

    #     # img_data = show_one(tmp_data)
    #     # img_affs = show_one(affs_xy)
    #     # img_weight = show_one(weightmap_xy)
    #     # im_cat = np.concatenate([img_data, img_affs, img_weight], axis=1)
    #     # Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
    # print(time.time() - t)