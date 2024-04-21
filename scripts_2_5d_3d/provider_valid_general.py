import os
import cv2
import h5py
import math
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.seg_util import genSegMalis
from utils.gen_affs import gen_affs
from data.data_affinity import seg_to_aff
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from utils.affinity_ours import gen_affs_mutex_multi, gen_affs_mutex_3d
from utils.weights import weight_binary_ratio

class Provider_valid(Dataset):
    def __init__(self, cfg, valid_data=None, test=False, test_split=None, direc='x'):
        # basic settings
        self.cfg = cfg
        self.mode = cfg.TRAIN.mode
        self.shift_channels = cfg.shift
        self.if_dilate = cfg.DATA.if_dilate
        self.if_bg = cfg.DATA.if_bg
        self.test = test
        self.direc = direc
        if valid_data is not None:
            valid_dataset_name = valid_data
            print('valid on valid dataset!',valid_dataset_name)
        else:
            valid_dataset_name = cfg.DATA.dataset_name
            print('valid on train dataset!',valid_dataset_name)
		# split training data
        self.train_split = cfg.DATA.train_split
        # training dataset files (h5), may contain many datasets
        if valid_dataset_name == 'cremiA':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif valid_dataset_name == 'cremiB':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif valid_dataset_name == 'cremiC':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif valid_dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs.h5']
            self.train_labels = ['cremiC_labels.h5']
            # self.sub_path = 'cremi'
            # self.train_datasets = ['cremiA_inputs.h5', 'cremiB_inputs.h5', 'cremiC_inputs.h5']
            # self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif valid_dataset_name == 'wafer4':
            self.sub_path = 'wafer4'
            self.train_datasets = ['wafer4_inputs.h5']
            self.train_labels = ['wafer4_labels.h5']
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
        # if valid_dataset_name == 'isbi_test' or valid_dataset_name == 'ac3':
        #     self.test_split = 100
        print('the number of valid(test) = %d' % self.test_split)

        # load dataset
        self.dataset = []
        self.labels = []
        self.labels_origin = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()
            
            if valid_dataset_name == 'ac3':
                data = data[:self.test_split]
            else:
                data = data[-self.test_split:]
            self.dataset.append(data)

            # load labels
            f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
            label = f_label['main'][:]
            f_label.close()
            if valid_dataset_name == 'ac3':
                label = label[:self.test_split]
            else:
                label = label[-self.test_split:]  
            self.labels_origin.append(label.copy())
            if self.if_dilate:
                if cfg.DATA.widen_way:
                    label = seg_widen_border(label, tsz_h=1)
                else:
                    label = genSegMalis(label, 1)
            self.labels.append(label)
        self.origin_data_shape = list(self.dataset[0].shape)

        # generate gt affinity
        self.gt_affs = []
        for k in range(len(self.labels)):
            temp = self.labels[k].copy()
            # self.gt_affs.append(seg_to_affgraph(temp, mknhood3d(1), pad='replicate').astype(np.float32))
            # self.gt_affs.append(gen_affs_mutex_3d(temp, shift=self.shift, padding=True, background=self.if_bg))
            self.gt_affs.append(seg_to_aff(temp).astype(np.float32))

        self.num_per_dataset = self.test_split
        self.iters_num = self.num_per_dataset * len(self.dataset)

    def __getitem__(self, index):
        pos_data = index // self.num_per_dataset

        if self.mode == 'x-y-z-2':
            if index == 0:
                img1 = self.dataset[pos_data][index].copy()
                lb1 = self.labels[pos_data][index].copy()
            else:
                img1 = self.dataset[pos_data][index-1].copy()
                lb1 = self.labels[pos_data][index-1].copy()
            img2 = self.dataset[pos_data][index].copy()
            lb2 = self.labels[pos_data][index].copy()
            if self.shift_channels is None:
                lbs = np.stack([lb1, lb2], axis=0)
                affs = seg_to_aff(lbs).astype(np.float32)
                affs = affs[:, 1]
            else:
                # affs = gen_affs_mutex_multi(lb1, lb2, shift=self.shift_channels, padding=True, background=self.if_bg)
                affs2 = gen_affs(lb2, None, dir=2, shift=1, padding=True, background=self.if_bg)
                affs1 = gen_affs(lb2, None, dir=1, shift=1, padding=True, background=self.if_bg)
                affs0 = gen_affs(lb1, lb2, dir=0, padding=True, background=self.if_bg)
                affs = np.stack([affs0, affs1, affs2], axis=0)
        else:
            raise NotImplementedError
        imgs = np.stack([img1, img2], axis=0)
        imgs = imgs.astype(np.float32) / 255.0
        weightmap = np.zeros_like(affs, dtype=np.float32)
        for i in range(affs.shape[0]):
            weightmap[i] = weight_binary_ratio(affs[i])
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        affs = np.ascontiguousarray(affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        return imgs, affs, weightmap

    def __len__(self):
        return self.iters_num

    def get_gt_affs(self, num_data=0):
        return self.gt_affs[num_data]

    def get_gt_lb(self, num_data=0):
        return self.labels_origin[num_data]

    def get_raw_data(self, num_data=0):
        return self.dataset[num_data]


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
