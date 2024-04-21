import os
import h5py
import tifffile
import numpy as np
from PIL import Image

def gen_colormap(seed=555):
    np.random.seed(seed)
    color_val = np.random.randint(0, 255, (65535, 3))
    # print('the shape of color map:', color_val.shape)
    return color_val

def sort_gt(path='../data/snemi3d/AC3_labels.h5', out='ac3.h5'):
    f_gt = h5py.File(path, 'r')
    labels = f_gt['main'][:]
    f_gt.close()
    labels = labels[-50:]

    value_0 = labels == 0
    print('min value=%d, max value=%d' % (np.min(labels), np.max(labels)))
    ids, count = np.unique(labels, return_counts=True)
    id_dict = {}
    for k,v in zip(ids, count):
        id_dict.setdefault(k, v)
    sorted_results = sorted(id_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    print(sorted_results[:10])
    num = 1
    new_labels = np.zeros_like(labels)
    for k in sorted_results:
        temp_id = k[0]
        if temp_id == 0:
            continue
        new_labels[labels==temp_id] = num
        num += 1

    new_labels[value_0] = 0
    new_labels = new_labels.astype(np.uint16)
    print('min value=%d, max value=%d' % (np.min(new_labels), np.max(new_labels)))

    f_out = h5py.File('./'+out, 'w')
    f_out.create_dataset('main', data=new_labels, dtype=new_labels.dtype, compression='gzip')
    f_out.close()

    return new_labels

def find_id(label, mask):
    label = label * mask
    ids, count = np.unique(label, return_counts=True)
    id_dict = {}
    for k,v in zip(ids, count):
        id_dict.setdefault(k, v)
    sorted_results = sorted(id_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    new_id = sorted_results[0][0]
    if new_id == 0:
        new_id = sorted_results[1][0]
    return new_id

def show_color(label, color_val):
    h, w = label.shape
    gt_color = np.zeros((h,w,3), dtype=np.int8)
    ids = np.unique(label)
    for i in ids:
        if i == 0:
            continue
        temp = np.zeros_like(label, dtype=np.uint8)
        temp[label==i] = 1
        tmp_color = color_val[i]
        gt_color[:,:,0] += temp * tmp_color[0]
        gt_color[:,:,1] += temp * tmp_color[1]
        gt_color[:,:,2] += temp * tmp_color[2]
    gt_color = gt_color.astype(np.uint8)
    return gt_color

def match_id(seg, label, mask=True, seed=0):
    if mask:
        seg[label==0] = 0
    ids, count = np.unique(seg, return_counts=True)
    id_dict = {}
    for k,v in zip(ids, count):
        id_dict.setdefault(k, v)
    sorted_results = sorted(id_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    # print(sorted_results[:10])

    max_id = np.max(seg) + 1
    used_id = []
    new_seg = np.zeros_like(seg)
    for k in sorted_results:
        tmp_id = k[0]
        if tmp_id == 0:
            continue
        mask = np.zeros_like(seg)
        mask[seg==tmp_id] = 1
        new_id = find_id(label.copy(), mask)
        if new_id not in used_id:
            used_id.append(new_id)
            new_seg[seg==tmp_id] = new_id
        else:
            new_seg[seg==tmp_id] = max_id
            max_id += 1
    new_seg[label==0] = 0

    color_val = gen_colormap(seed=seed)
    label_color = show_color(label, color_val)
    seg_color = show_color(new_seg, color_val)
    im_cat = np.concatenate([seg_color, label_color], axis=1)
    return im_cat


if __name__ == '__main__':
    seg = np.random.randint(0, 100, size=(512, 512))
    label = np.random.randint(0, 100, size=(512, 512))
    out = match_id(seg, label) # 2D
    Image.fromarray(out).save('./out.png')