import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def embedding2affs_norm(embedding1, embedding2, shift=1, size=256):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
    affs0 = torch.abs(affs0)
    affs0[affs0 < 0.0] = 0.0
    affs0[affs0 > 1.0] = 1.0

    affs1 = torch.sum(embedding2[:, :, shift:, :]*embedding2[:, :, :size-shift, :], dim=1, keepdim=True)
    affs1 = torch.abs(affs1)
    affs1[affs1 < 0.0] = 0.0
    affs1[affs1 > 1.0] = 1.0
    affs1 = F.pad(affs1, (0,0,1,0), mode='reflect')

    affs2 = torch.sum(embedding2[:, :, :, shift:]*embedding2[:, :, :, :size-shift], dim=1, keepdim=True)
    affs2 = torch.abs(affs2)
    affs2[affs2 < 0.0] = 0.0
    affs2[affs2 > 1.0] = 1.0
    affs2 = F.pad(affs2, (1,0,0,0), mode='reflect')

    affs = torch.cat([affs0, affs1, affs2], dim=1)
    return affs

def embedding_loss_norm(embedding1, embedding2, target, weightmap, criterion, affs0_weight=1, shift=1, size=256):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
    affs0 = (affs0 + 1) / 2
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1], weightmap[:, 0:1])

    affs1 = torch.sum(embedding2[:, :, shift:, :]*embedding2[:, :, :size-shift, :], dim=1, keepdim=True)
    affs1 = (affs1 + 1) / 2
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, shift:, :], weightmap[:, 1:2, shift:, :])
    affs1 = F.pad(affs1, (0,0,1,0), mode='reflect')

    affs2 = torch.sum(embedding2[:, :, :, shift:]*embedding2[:, :, :, :size-shift], dim=1, keepdim=True)
    affs2 = (affs2 + 1) / 2
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, shift:], weightmap[:, 2:3, :, shift:])
    affs2 = F.pad(affs2, (1,0,0,0), mode='reflect')

    loss = affs0_weight * loss0 + loss1 + loss2
    affs = torch.cat([affs0, affs1, affs2], dim=1)
    return loss, affs


def embedding_loss_norm_abs(embedding1, embedding2, target, weightmap, criterion, affs0_weight=1, shift=1, size=256):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
    affs0 = torch.abs(affs0)
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1], weightmap[:, 0:1])

    affs1 = torch.sum(embedding2[:, :, shift:, :]*embedding2[:, :, :size-shift, :], dim=1, keepdim=True)
    affs1 = torch.abs(affs1)
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, shift:, :], weightmap[:, 1:2, shift:, :])
    affs1 = F.pad(affs1, (0,0,1,0), mode='reflect')

    affs2 = torch.sum(embedding2[:, :, :, shift:]*embedding2[:, :, :, :size-shift], dim=1, keepdim=True)
    affs2 = torch.abs(affs2)
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, shift:], weightmap[:, 2:3, :, shift:])
    affs2 = F.pad(affs2, (1,0,0,0), mode='reflect')

    loss = affs0_weight * loss0 + loss1 + loss2
    affs = torch.cat([affs0, affs1, affs2], dim=1)
    return loss, affs


def embedding_loss_norm_trunc(embedding1, embedding2, target, weightmap, criterion, affs0_weight=1, shift=1, size=256):
    embedding1 = F.normalize(embedding1, p=2, dim=1)
    embedding2 = F.normalize(embedding2, p=2, dim=1)
    affs0 = torch.sum(embedding1*embedding2, dim=1, keepdim=True)
    affs0 = torch.clamp(affs0, 0.0, 1.0)
    loss0 = criterion(affs0, target[:, 0:1], weightmap[:, 0:1])

    affs1 = torch.sum(embedding2[:, :, shift:, :]*embedding2[:, :, :size-shift, :], dim=1, keepdim=True)
    affs1 = torch.clamp(affs1, 0.0, 1.0)
    loss1 = criterion(affs1, target[:, 1:2, shift:, :], weightmap[:, 1:2, shift:, :])
    affs1 = F.pad(affs1, (0,0,1,0), mode='reflect')

    affs2 = torch.sum(embedding2[:, :, :, shift:]*embedding2[:, :, :, :size-shift], dim=1, keepdim=True)
    affs2 = torch.clamp(affs2, 0.0, 1.0)
    loss2 = criterion(affs2, target[:, 2:3, :, shift:], weightmap[:, 2:3, :, shift:])
    affs2 = F.pad(affs2, (1,0,0,0), mode='reflect')

    loss = affs0_weight * loss0 + loss1 + loss2
    affs = torch.cat([affs0, affs1, affs2], dim=1)
    return loss, affs


def embedding_loss_l2(embedding1, embedding2, target, weightmap, criterion, affs0_weight=1, shift=1, size=256):
    dist_affs0 = (embedding1 - embedding2) ** 2
    affs0 = torch.sum(dist_affs0, dim=1, keepdim=True)
    affs0 = (affs0 - torch.min(affs0)) / (torch.max(affs0) - torch.min(affs0))
    # affs0 = torch.abs(affs0)
    # affs0 = (affs0 + 1) / 2
    # affs0[affs0 < 0.0] = 0.0
    # affs0[affs0 > 1.0] = 1.0
    loss0 = criterion(affs0, target[:, 0:1], weightmap[:, 0:1])

    dist_affs1 = (embedding2[:, :, shift:, :] - embedding2[:, :, :size-shift, :]) ** 2
    affs1 = torch.sum(dist_affs1, dim=1, keepdim=True)
    affs1 = (affs1 - torch.min(affs1)) / (torch.max(affs1) - torch.min(affs1))
    # affs1 = torch.abs(affs1)
    # affs1 = (affs1 + 1) / 2
    # affs1[affs1 < 0.0] = 0.0
    # affs1[affs1 > 1.0] = 1.0
    loss1 = criterion(affs1, target[:, 1:2, shift:, :], weightmap[:, 1:2, shift:, :])
    affs1 = F.pad(affs1, (0,0,1,0), mode='reflect')

    dist_affs2 = (embedding2[:, :, :, shift:] - embedding2[:, :, :, :size-shift]) ** 2
    affs2 = torch.sum(dist_affs2, dim=1, keepdim=True)
    affs2 = (affs2 - torch.min(affs2)) / (torch.max(affs2) - torch.min(affs2))
    # affs2 = torch.abs(affs2)
    # affs2 = (affs2 + 1) / 2
    # affs2[affs2 < 0.0] = 0.0
    # affs2[affs2 > 1.0] = 1.0
    loss2 = criterion(affs2, target[:, 2:3, :, shift:], weightmap[:, 2:3, :, shift:])
    affs2 = F.pad(affs2, (1,0,0,0), mode='reflect')

    loss = affs0_weight * loss0 + loss1 + loss2
    affs = torch.cat([affs0, affs1, affs2], dim=1)
    return loss, affs
