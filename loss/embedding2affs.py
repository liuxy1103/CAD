import torch
import numpy as np
import torch.nn as nn

def embedding2affs_single(embedding1, embedding2, shift=[0,0,0], dis=nn.CosineSimilarity(dim=1, eps=1e-6)):
    assert len(shift) == 3, 'the len(shift) must be 3'
    b, c, h, w = embedding1.shape
    if shift[0] == 0:
        embedding2 = embedding1.clone()
    # embedding3 = embedding1.detach().clone()
    embedding3 = torch.zeros_like(embedding1)
    
    if shift[1] <= 0 and shift[2] <= 0:
        embedding3[:, :, -shift[1]:, -shift[2]:] = embedding1[:, :, :h+shift[1], :w+shift[2]]
    elif shift[1] <= 0 and shift[2] > 0:
        embedding3[:, :, -shift[1]:, :w-shift[2]] = embedding1[:, :, :h+shift[1], shift[2]:]
    elif shift[1] > 0 and shift[2] <= 0:
        embedding3[:, :, :h-shift[1], -shift[2]:] = embedding1[:, :, shift[1]:, :w+shift[2]]
    elif shift[1] > 0 and shift[2] > 0:
        embedding3[:, :, :h-shift[1], :w-shift[2]] = embedding1[:, :, shift[1]:, shift[2]:]
    else:
        pass
    
    out = torch.abs(dis(embedding3, embedding2))
    out[out < 0.0] = 0.0
    out[out > 1.0] = 1.0
    return out

def embedding2affs_multi(embedding1, embedding2, shift=[[0,0,0]]):
    dis = nn.CosineSimilarity(dim=1, eps=1e-6)
    b, c, h, w = embedding1.shape
    pred = torch.zeros((b, len(shift), h, w), device=embedding1.device)
    for i, k in enumerate(shift):
        pred_affs = embedding2affs_single(embedding1, embedding2, shift=k, dis=dis)
        pred[:, i] = pred_affs
    return pred

def embedding_loss(embedding1, embedding2, target, weightmap, criterion, shift=[[0,0,0]]):
    dis = nn.CosineSimilarity(dim=1, eps=1e-6)
    pred = torch.zeros_like(target)
    # losses = []
    for i, k in enumerate(shift):
        pred_affs = embedding2affs_single(embedding1, embedding2, shift=k, dis=dis)
        pred[:, i] = pred_affs
        # losses.append(criterion(pred, target[:, i], weightmap[:, i]))
    loss = criterion(pred, target, weightmap)
    # loss = sum(losses)
    return loss, pred