import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from loss.loss import WeightedMSE, WeightedBCE
from provider_valid_general import Provider_valid
from loss.loss import BCELoss, WeightedBCE, MSELoss
from CoDetectionCNN import CoDetectionCNN
from utils.shift_channels import shift_func
from loss.embedding2affs import embedding_loss
from loss.embedding_norm import embedding_loss_norm, embedding_loss_norm_abs
from loss.embedding_norm import embedding_loss_norm_trunc

import waterz
from utils.show import draw_fragments_3d
from utils.fragment import watershed, randomlabel, relabel
from data.data_segmentation import seg_widen_border
from utils.lmc import mc_baseline
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings
warnings.filterwarnings("ignore")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='2d_bs16_ps256_loss1.0_slice1.0_cross1.0', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, default='2023-10-23--15-01-45_2d_bs16_ps256_loss0.0_slice1.0_cross1.0_interaction10.0_3d_z16_h160_noft_lr_ratio1_ft10000')
    parser.add_argument('-id', '--model_id', type=int, default=121000)
    parser.add_argument('-m', '--mode', type=str, default='wafer4')
    parser.add_argument('-ts', '--test_split', type=int, default=20)
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-sw', '--show', action='store_true', default=True)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))
    
    if cfg.DATA.shift_channels is None:
        cfg.shift = None
    else:
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('../inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)

    device = torch.device('cuda:0')
    model = CoDetectionCNN(n_channels=cfg.MODEL.input_nc,
                        n_classes=cfg.MODEL.output_nc,
                        filter_channel=cfg.MODEL.filter_channel,
                        sig=cfg.MODEL.if_sigmoid).to(device)

    ckpt_path = os.path.join('../models', trained_model, 'model-%06d.ckpt' % args.model_id)
    checkpoint = torch.load(ckpt_path)

    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                            shuffle=False, drop_last=False, pin_memory=True)

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")

    model.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    output_affs = []
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        inference_size = inputs.shape[-1]
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        # inputs = F.pad(inputs, (48, 48, 48, 48), mode='reflect')
        with torch.no_grad():
            if inputs.shape[-1] == 1250:
                inputs = F.pad(inputs, (7, 7, 7, 7), mode='reflect')
                embedding1, embedding2 = model(inputs)
                embedding1 = embedding1[:,:,7:-7,7:-7]
                embedding2 = embedding2[:,:,7:-7,7:-7]
            else:
                embedding1, embedding2 = model(inputs)
        # embedding1 = F.pad(embedding1, (-48, -48, -48, -48))
        # embedding2 = F.pad(embedding2, (-48, -48, -48, -48))
        if cfg.TRAIN.loss_mode == 'nn_cos':
            tmp_loss, pred = embedding_loss(embedding1, embedding2, target, weightmap, criterion, shift=cfg.shift)
        elif cfg.TRAIN.loss_mode == 'norm':
            tmp_loss, pred = embedding_loss_norm(embedding1, embedding2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=inference_size)
        elif cfg.TRAIN.loss_mode == 'abs':
            tmp_loss, pred = embedding_loss_norm_abs(embedding1, embedding2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=inference_size)
        elif cfg.TRAIN.loss_mode == 'trunc':
            tmp_loss, pred = embedding_loss_norm_trunc(embedding1, embedding2, target, weightmap, criterion,
                                            affs0_weight=cfg.TRAIN.affs0_weight, shift=1, size=inference_size)
        else:
            raise NotImplementedError
        losses_valid.append(tmp_loss.item())
        if cfg.TRAIN.if_verse:
            pred = pred * 2 - 1
            pred = torch.clamp(pred, 0.0, 1.0)
        output_affs.append(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    epoch_loss = sum(losses_valid) / len(losses_valid)
    output_affs = np.asarray(output_affs, dtype=np.float32)
    output_affs = np.transpose(output_affs, (1, 0, 2, 3))
    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()

    # save
    print('save affs...')
    print('the shape of affs:', output_affs.shape)
    f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
    f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
    f.close()


    # save
    print('save gt affs...')
    print('the shape of affs:', gt_affs.shape)
    f = h5py.File(os.path.join(out_affs, 'affs_gt.hdf'), 'w')
    f.create_dataset('main', data=gt_affs, dtype=np.float32, compression='gzip')
    f.close()


    # for mutex
    output_affs = output_affs[:3]

    print('segmentation...')
    fragments = watershed(output_affs, 'maxima_distance')
    #sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    seg_waterz = list(waterz.agglomerate(output_affs, [0.50],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]
    seg_waterz = relabel(seg_waterz).astype(np.uint16)
    print('the max id = %d' % np.max(seg_waterz))
    f = h5py.File(os.path.join(out_affs, 'seg_waterz.hdf'), 'w')
    f.create_dataset('main', data=seg_waterz, dtype=seg_waterz.dtype, compression='gzip')
    f.close()


    print('save gt segmentation')
    f = h5py.File(os.path.join(out_affs, 'seg_gt.hdf'), 'w')
    f.create_dataset('main', data=gt_seg, dtype=gt_seg.dtype, compression='gzip')
    f.close()



    arand = adapted_rand_ref(gt_seg, seg_waterz, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, seg_waterz, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('waterz: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')

    seg_lmc = mc_baseline(output_affs)
    seg_lmc = relabel(seg_lmc).astype(np.uint16)
    print('the max id = %d' % np.max(seg_lmc))
    f = h5py.File(os.path.join(out_affs, 'seg_lmc.hdf'), 'w')
    f.create_dataset('main', data=seg_lmc, dtype=seg_lmc.dtype, compression='gzip')
    f.close()

    arand = adapted_rand_ref(gt_seg, seg_lmc, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, seg_lmc, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
        (voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')

    # compute MSE
    if args.pixel_metric:
        print('MSE...')
        output_affs_prop = output_affs.copy()
        whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
        print('BCE...')
        output_affs = np.clip(output_affs, 0.000001, 0.999999)
        bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
        whole_bce = np.sum(bce) / np.size(gt_affs)
        output_affs[output_affs <= 0.5] = 0
        output_affs[output_affs > 0.5] = 1
        print('F1...')
        whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
        # whole_arand = 0.0
        # new
        print('F1 boundary...')
        whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
        # whole_arand_bound = 0.0
        print('mAP...')
        # whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        whole_map = 0.0
        print('AUC...')
        # whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        whole_auc = 0.0
        ###################################################
        malis = 0.0
        ###################################################
        print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
            (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
                    (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('\n')
    else:
        output_affs_prop = output_affs
    f_txt.close()

    # show
    if args.show:
        affs_img_path = os.path.join(out_affs, 'affs_img')
        seg_img_path = os.path.join(out_affs, 'seg_img')
        if not os.path.exists(affs_img_path):
            os.makedirs(affs_img_path)
        if not os.path.exists(seg_img_path):
            os.makedirs(seg_img_path)

        print('show affs...')
        output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        for i in range(output_affs_prop.shape[1]):
            cat1 = np.concatenate([output_affs_prop[0,i], output_affs_prop[1,i], output_affs_prop[2,i]], axis=1)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), cat1)
        
        print('show seg...')
        # seg_waterz[gt_seg==0] = 0
        # seg_lmc[gt_seg==0] = 0
        color_seg_waterz = draw_fragments_3d(seg_waterz)
        color_seg_lmc = draw_fragments_3d(seg_lmc)
        color_gt = draw_fragments_3d(gt_seg)
        for i in range(color_gt.shape[0]):
            im_cat = np.concatenate([color_seg_waterz[i], color_seg_lmc[i], color_gt[i]], axis=1)
            cv2.imwrite(os.path.join(seg_img_path, str(i).zfill(4)+'.png'), im_cat)
    print('Done')
