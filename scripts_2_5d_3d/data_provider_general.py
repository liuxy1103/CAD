from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import h5py
import math
import time
import torch
import random
import numpy as np
from PIL import Image
import multiprocessing
from joblib import Parallel
from joblib import delayed
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from augmentation import Flip
from augmentation import Elastic
from augmentation import Grayscale
from augmentation import Rotate
from augmentation import Rescale
from utils.gen_affs import gen_affs
from utils.affinity_ours import gen_affs_mutex_multi
from utils.seg_util import genSegMalis
from utils.weights import weight_binary_ratio
from data.data_segmentation import seg_widen_border
from data.data_affinity import seg_to_aff

def center_crop(image, det_shape=[160, 160]):
	src_shape = image.shape
	shift1 = (src_shape[1] - det_shape[0]) // 2
	shift2 = (src_shape[2] - det_shape[1]) // 2
	assert shift1 >= 0 or shift2 >= 0, "overflow in center-crop"
	image = image[:, shift1:shift1+det_shape[0], shift2:shift2+det_shape[1]]
	return image

class Train(Dataset):
	def __init__(self, cfg):
		super(Train, self).__init__()
		# multiprocess settings
		# num_cores = multiprocessing.cpu_count()
		# self.parallel = Parallel(n_jobs=num_cores, backend='threading')
		self.cfg = cfg

		self.data_folder = cfg.DATA.data_folder
		self.dataset_name = cfg.DATA.dataset_name
		self.patch_size = list(cfg.DATA.patch_size)
		self.shift_channels = cfg.shift
		self.padding = cfg.DATA.padding
		self.crop_size = [self.patch_size[i]+2*self.padding for i in range(len(self.patch_size))]
		# self.patch_size = [2] + self.patch_size
		self.crop_size = [2] + self.crop_size
		self.mode = cfg.TRAIN.mode
		self.if_dilate = cfg.DATA.if_dilate
		self.if_bg = cfg.DATA.if_bg

		# training dataset files (h5), may contain many datasets
		if cfg.DATA.dataset_name == 'cremiA':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiA_inputs.h5']
			self.train_labels = ['cremiA_labels.h5']
		elif cfg.DATA.dataset_name == 'cremiB':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiB_inputs.h5']
			self.train_labels = ['cremiB_labels.h5']
		elif cfg.DATA.dataset_name == 'cremiC':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiC_inputs.h5']
			self.train_labels = ['cremiC_labels.h5']
		elif cfg.DATA.dataset_name == 'cremi-all':
			self.sub_path = 'cremi'
			self.train_datasets = ['cremiA_inputs.h5', 'cremiB_inputs.h5', 'cremiC_inputs.h5']
			self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
		elif cfg.DATA.dataset_name == 'wafer4':
			self.sub_path = 'wafer4'
			self.train_datasets = ['wafer4_inputs.h5']
			self.train_labels = ['wafer4_labels.h5']
		elif cfg.DATA.dataset_name == 'isbi':
			self.sub_path = 'snemi3d'
			self.train_datasets = ['isbi_inputs.h5']
			self.train_labels = ['isbi_labels.h5']
		elif cfg.DATA.dataset_name == 'ac3':
			self.sub_path = 'ac3_ac4'
			self.train_datasets = ['AC3_inputs.h5']
			if cfg.DATA.if_sparse:
				if cfg.DATA.sparse_id == 5:
					self.train_labels = ['AC3_labels5.h5']
				elif cfg.DATA.sparse_id == 10:
					self.train_labels = ['AC3_labels10.h5']
				elif cfg.DATA.sparse_id == 20:
					self.train_labels = ['AC3_labels20.h5']
			else:
				self.train_labels = ['AC3_labels.h5']
		elif cfg.DATA.dataset_name == 'ac4':
			self.sub_path = 'ac3_ac4'
			self.train_datasets = ['AC4_inputs.h5']
			self.train_labels = ['AC4_labels.h5']
		elif cfg.DATA.dataset_name == 'fib':
			self.sub_path = 'fib'
			self.train_datasets = ['fib_inputs.h5']
			self.train_labels = ['fib_labels.h5']
		else:
			raise AttributeError('No this dataset type!')

		# the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
		self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
		assert len(self.train_datasets) == len(self.train_labels)

		# split training data
		self.train_split = cfg.DATA.train_split

		# augmentation
		self.if_scale_aug = cfg.DATA.if_scale_aug
		self.if_filp_aug = cfg.DATA.if_filp_aug
		self.if_elastic_aug = cfg.DATA.if_elastic_aug
		self.if_intensity_aug = cfg.DATA.if_intensity_aug
		self.if_rotation_aug = cfg.DATA.if_rotation_aug

		# augmentation initoalization
		self.augs_init()

		# load dataset
		self.dataset = []
		self.labels = []
		for k in range(len(self.train_datasets)):
			print('load ' + self.train_datasets[k] + ' ...')
			# load raw data
			f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
			data = f_raw['main'][:]
			f_raw.close()
			data = data[:self.train_split]
			self.dataset.append(data)

			# load labels
			f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
			label = f_label['main'][:]
			f_label.close()
			label = label[:self.train_split]
			if self.if_dilate:
				if cfg.DATA.widen_way:
					label = seg_widen_border(label, tsz_h=1)
				else:
					label = genSegMalis(label, 1)
			self.labels.append(label)

		# the training dataset size
		self.raw_data_shape = list(self.dataset[0].shape)
		print('raw data shape: ', self.raw_data_shape)


	def __getitem__(self, index):
		# random select one dataset if contain many datasets
		k = random.randint(0, len(self.train_datasets)-1)
		used_data = self.dataset[k] 
		used_label = self.labels[k]

		random_z = random.randint(0, self.raw_data_shape[0]-self.crop_size[0])
		# random_z = random.randint(0, self.raw_data_shape[0]-self.shift-1)
		random_y = random.randint(0, self.raw_data_shape[1]-self.crop_size[1])
		random_x = random.randint(0, self.raw_data_shape[2]-self.crop_size[2])

		imgs = used_data[random_z:random_z+self.crop_size[0], \
						random_y:random_y+self.crop_size[1], \
						random_x:random_x+self.crop_size[2]].copy()
		lbs = used_label[random_z:random_z+self.crop_size[0], \
						random_y:random_y+self.crop_size[1], \
						random_x:random_x+self.crop_size[2]].copy()
		# img1 = used_data[random_z, \
		# 				random_y:random_y+self.crop_size[1], \
		# 				random_x:random_x+self.crop_size[2]].copy()
		# img2 = used_data[random_z+self.shift, \
		# 				random_y:random_y+self.crop_size[1], \
		# 				random_x:random_x+self.crop_size[2]].copy()
		# imgs = np.stack([img1, img2], axis=0)
		# lb1 = used_label[random_z, \
		# 				random_y:random_y+self.crop_size[1], \
		# 				random_x:random_x+self.crop_size[2]].copy()
		# lb2 = used_label[random_z+self.shift, \
		# 				random_y:random_y+self.crop_size[1], \
		# 				random_x:random_x+self.crop_size[2]].copy()
		# lbs = np.stack([lb1, lb2], axis=0)

		imgs = imgs.astype(np.float32) / 255.0
		data = {'image': imgs, 'label': lbs}
		# p=0.5 for augmentation
		if np.random.rand() < 0.5:
			data = self.augs_mix(data)
		imgs = data['image']
		lbs = data['label']
		imgs = center_crop(imgs, det_shape=self.patch_size)
		lbs = center_crop(lbs, det_shape=self.patch_size)

		img1 = imgs[0]
		lb1 = lbs[0]
		h, w = img1.shape
		if self.mode == 'x-y-z-2':
			img2 = imgs[1]
			lb2 = lbs[1]
			if self.shift_channels is None:
				affs = seg_to_aff(lbs)
				affs = affs[:, 1]
			else:
				# affs = gen_affs_mutex_multi(lb1, lb2, shift=self.shift_channels, padding=True, background=self.if_bg)
				affs2 = gen_affs(lb2, None, dir=2, shift=1, padding=True, background=self.if_bg)
				affs1 = gen_affs(lb2, None, dir=1, shift=1, padding=True, background=self.if_bg)
				affs0 = gen_affs(lb1, lb2, dir=0, padding=True, background=self.if_bg)
				affs = np.stack([affs0, affs1, affs2], axis=0)
		else:
			raise NotImplementedError

		# extend dimension
		imgs = np.stack([img1, img2], axis=0)
		# imgs = center_crop(imgs, det_shape=self.patch_size)
		# affs = center_crop(affs, det_shape=self.patch_size)
		weightmap = np.zeros_like(affs, dtype=np.float32)
		for i in range(affs.shape[0]):
			weightmap[i] = weight_binary_ratio(affs[i])

		imgs = np.ascontiguousarray(imgs, dtype=np.float32)
		affs = np.ascontiguousarray(affs, dtype=np.float32)
		weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
		lbs = lbs.astype(np.float32)
		return imgs, lbs, affs, weightmap

	def __len__(self):
		return int(sys.maxsize)

	def augs_init(self):
		# https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
		self.aug_rotation = Rotate(p=0.5)
		self.aug_rescale = Rescale(p=0.5)
		self.aug_flip = Flip(p=1.0, do_ztrans=0)
		self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
		self.aug_grayscale = Grayscale(p=0.75)

	# TO DO
	def augs_single(self, data):
		random_id = np.random.randint(1, 5+1)
		if random_id == 1:
			data = self.aug_rotation(data)
		elif random_id == 2:
			data = self.aug_rescale(data)
		elif random_id == 3:
			data = self.aug_flip(data)
		elif random_id == 4:
			data = self.aug_elastic(data)
		elif random_id == 5:
			data = self.aug_grayscale(data)
		else:
			raise NotImplementedError
		return data

	def augs_mix(self, data):
		if self.if_filp_aug and random.random() > 0.5:
			data = self.aug_flip(data)
		if self.if_rotation_aug and random.random() > 0.5:
			data = self.aug_rotation(data)
		if self.if_scale_aug and random.random() > 0.5:
			data = self.aug_rescale(data)
		if self.if_elastic_aug and random.random() > 0.5:
			data = self.aug_elastic(data)
		if self.if_intensity_aug and random.random() > 0.5:
			data = self.aug_grayscale(data)
		return data

def collate_fn(batchs):
	out_input = []
	for batch in batchs:
		out_input.append(torch.from_numpy(batch['image']))
	
	out_input = torch.stack(out_input, 0)
	return {'image':out_input}

class Provider(object):
	def __init__(self, stage, cfg):
			#patch_size, batch_size, num_workers, is_cuda=True):
		self.stage = stage
		if self.stage == 'train':
			self.data = Train(cfg)
			self.batch_size = cfg.TRAIN.batch_size
			self.num_workers = cfg.TRAIN.num_workers
		elif self.stage == 'valid':
			# return valid(folder_name, kwargs['data_list'])
			pass
		else:
			raise AttributeError('Stage must be train/valid')
		self.is_cuda = cfg.TRAIN.if_cuda
		self.data_iter = None
		self.iteration = 0
		self.epoch = 1
	
	def __len__(self):
		return self.data.num_per_epoch
	
	def build(self):
		if self.stage == 'train':
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                                             shuffle=False, drop_last=False, pin_memory=True))
		else:
			self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))
	
	def next(self):
		if self.data_iter is None:
			self.build()
		try:
			batch = self.data_iter.next()
			self.iteration += 1
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
				batch[3] = batch[3].cuda()
			return batch
		except StopIteration:
			self.epoch += 1
			self.build()
			self.iteration += 1
			batch = self.data_iter.next()
			if self.is_cuda:
				batch[0] = batch[0].cuda()
				batch[1] = batch[1].cuda()
				batch[2] = batch[2].cuda()
				batch[3] = batch[3].cuda()
			return batch

def show(img3d):
	# only used for image with shape [18, 160, 160]
	num = img3d.shape[0]
	column = 5
	row = math.ceil(num / float(column))
	size = img3d.shape[1]
	img_all = np.zeros((size*row, size*column), dtype=np.uint8)
	for i in range(row):
		for j in range(column):
			index = i*column + j
			if index >= num:
				img = np.zeros_like(img3d[0], dtype=np.uint8)
			else:
				img = (img3d[index] * 255).astype(np.uint8)
			img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
	return img_all


if __name__ == '__main__':
	import yaml
	from attrdict import AttrDict
	from utils.shift_channels import shift_func
	from utils.show import show_twoImage
	""""""
	seed = 555
	np.random.seed(seed)
	random.seed(seed)
	cfg_file = 'seg_general_ac4_data80_c16_norm_wz2_noaug.yaml'
	with open('./config/' + cfg_file, 'r') as f:
		cfg = AttrDict( yaml.load(f) )
	
	if cfg.DATA.shift_channels is not None:
		cfg.shift = shift_func(cfg.DATA.shift_channels)
	else:
		cfg.shift = None
	
	out_path = os.path.join('./', 'data_temp')
	if not os.path.exists(out_path):
		os.mkdir(out_path)
	data = Train(cfg)
	t = time.time()
	for i in range(0, 50):
		t1 = time.time()
		tmp_data, affs, weightmap = iter(data).__next__()
		print('single cost time: ', time.time()-t1)
		# tmp_data = np.squeeze(tmp_data, axis=0)
		# affs = np.squeeze(affs, axis=0)
		im_cat = show_twoImage(tmp_data, affs[:3])
		Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4)+'.png'))
	print(time.time() - t)