import numpy as np

def gen_affs(map1, map2=None, dir=0, shift=1, padding=True, background=False):
    if dir == 0 and map2 is None:
        raise AttributeError('map2 is none')
    map1 = map1.astype(np.float32)
    h, w = map1.shape
    if dir == 0:
        map2 = map2.astype(np.float32)
    elif dir == 1:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[shift:, :] = map1[:h-shift, :]
    elif dir == 2:
        map2 = np.zeros_like(map1, dtype=np.float32)
        map2[:, shift:] = map1[:, :w-shift]
    else:
        raise AttributeError('dir must be 0, 1 or 2')
    dif = map2 - map1
    out = dif.copy()
    out[dif == 0] = 1
    out[dif != 0] = 0
    if background:
        out[map1 == 0] = 0
        out[map2 == 0] = 0
    if padding:
        if dir == 1:
            out[0, :] = (map1[0, :] > 0).astype(np.float32)
        if dir == 2:
            out[:, 0] = (map1[:, 0] > 0).astype(np.float32)
    return out


def gen_affs_3d(labels, shift=1, padding=True, background=False):
    assert len(labels.shape) == 3, '3D input'
    out = []
    for i in range(labels.shape[0]):
        if i == 0:
            if padding:
                affs0 = (labels[0] > 0).astype(np.float32)
            else:
                affs0 = np.zeros_like(labels[0], dtype=np.float32)
        else:
            affs0 = gen_affs(labels[i-1], labels[i], dir=0, shift=shift, padding=padding, background=background)
        affs1 = gen_affs(labels[i], None, dir=1, shift=shift, padding=padding, background=background)
        affs2 = gen_affs(labels[i], None, dir=2, shift=shift, padding=padding, background=background)
        affs = np.stack([affs0, affs1, affs2], axis=0)
        out.append(affs)
    out = np.asarray(out, dtype=np.float32)
    out = np.transpose(out, (1, 0, 2, 3))
    return out
