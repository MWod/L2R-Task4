import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

def ncc_loss(sources, targets, device="cpu", **params):
    # Global NCC
    size = sources.size(2)*sources.size(3)*sources.size(4)
    sources_mean = torch.mean(sources, dim=(1, 2, 3, 4)).view(sources.size(0), 1, 1, 1, 1)
    targets_mean = torch.mean(targets, dim=(1, 2, 3, 4)).view(targets.size(0), 1, 1, 1, 1)
    sources_std = torch.std(sources, dim=(1, 2, 3, 4)).view(sources.size(0), 1, 1, 1, 1)
    targets_std = torch.std(targets, dim=(1, 2, 3, 4)).view(targets.size(0), 1, 1, 1, 1)
    ncc = (1/size)*torch.sum((sources - sources_mean)*(targets-targets_mean) / (sources_std * targets_std), dim=(1, 2, 3, 4))
    return -torch.mean(ncc)

def mind_loss(sources, targets, device="cpu", **params):
    try:
        dilation = params['dilation']
        radius = params['radius']
        return torch.mean((MINDSSC(sources, device=device, dilation=dilation, radius=radius) - MINDSSC(targets, device=device, dilation=dilation, radius=radius))**2)
    except:
        return torch.mean((MINDSSC(sources, device=device) - MINDSSC(targets, device=device))**2)

def curvature_regularization(displacement_fields, device="cpu", **params):
    u_x = displacement_fields[:, 0, :, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3), displacement_fields.size(4))
    u_y = displacement_fields[:, 1, :, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3), displacement_fields.size(4))
    u_z = displacement_fields[:, 2, :, :, :].view(-1, 1, displacement_fields.size(2), displacement_fields.size(3), displacement_fields.size(4))
    x_laplacian = utils.tensor_laplacian(u_x, device)[:, :, 1:-1, 1:-1, 1:-1]
    y_laplacian = utils.tensor_laplacian(u_y, device)[:, :, 1:-1, 1:-1, 1:-1]
    z_laplacian = utils.tensor_laplacian(u_z, device)[:, :, 1:-1, 1:-1, 1:-1]
    x_term = x_laplacian**2
    y_term = y_laplacian**2
    z_term = z_laplacian**2
    curvature = 1/3*torch.mean(x_term + y_term + z_term)
    return curvature

def simple_regularization(displacement_fields, device="cpu", **params):
    penalty_y = ((displacement_fields[:, :, 1:, :, :] - displacement_fields[:, :, :-1, :, :])**2).mean()
    penalty_x = ((displacement_fields[:, :, :, 1:, :] - displacement_fields[:, :, :, :-1, :])**2).mean()
    penalty_z = ((displacement_fields[:, :, :, :, 1:] - displacement_fields[:, :, :, :, :-1])**2).mean()
    penalty = 1/3*(penalty_x + penalty_y + penalty_z)
    return penalty

def dice(m1, m2):
    smooth = 1
    m1 = m1.float().contiguous().view(-1)
    m2 = m2.float().contiguous().view(-1)
    m3 = torch.sum(m1 * m2)
    result = 1 - ((2 * m3 + smooth) / (m1.sum() + m2.sum() + smooth))
    return result

def dice_loss(sources_masks, targets_masks, device="cpu", **params):
    unique_values = torch.unique(sources_masks)
    if len(unique_values) == 1:
        return torch.autograd.Variable(torch.Tensor([0]), requires_grad=True).to(device)    
    loss = torch.autograd.Variable(torch.Tensor([0]), requires_grad=True).to(device)
    for i in range(1, len(unique_values)):
        c_dice = dice(sources_masks == unique_values[i], targets_masks == unique_values[i])
        loss = loss + c_dice
    loss = loss / (len(unique_values) - 1)
    return loss

def mse_loss(sources_masks, targets_masks, device="cpu", **params):
    return torch.mean((sources_masks-targets_masks)**2)

def pdist_squared(x):
    xx = (x**2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    dist[dist != dist] = 0
    dist = torch.clamp(dist.float(), 0.0, np.inf)
    return dist

def MINDSSC(img, radius=2, dilation=2, device="cpu"):
    # Code from: https://github.com/voxelmorph/voxelmorph/pull/145
    kernel_size = radius * 2 + 1
    six_neighbourhood = torch.Tensor([[0,1,1],
                                      [1,1,0],
                                      [1,0,1],
                                      [1,1,2],
                                      [2,1,1],
                                      [1,2,1]]).long()
    dist = pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
    mask = ((x > y).view(-1) & (dist == 2).view(-1))
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask,:]
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask,:]
    mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
    mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad1 = nn.ReplicationPad3d(dilation)
    rpad2 = nn.ReplicationPad3d(radius)
    ssd = F.avg_pool3d(rpad2((F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2), kernel_size, stride=1)
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean().item()*0.001, mind_var.mean().item()*1000)
    mind /= mind_var
    mind = torch.exp(-mind)
    mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
    return mind


def ncc_loss_local(y_true, y_pred, device="cpu", **params):
    # Local NCC - Code from VoxelMorph framework
    win = None
    I = y_true
    J = y_pred

    # get dimension of volume
    # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

    # set window size
    win = [9] * ndims if win is None else win

    # compute filters
    sum_filt = torch.ones([1, 1, *win]).to("cuda")

    pad_no = math.floor(win[0]/2)

    if ndims == 1:
        stride = (1)
        padding = (pad_no)
    elif ndims == 2:
        stride = (1,1)
        padding = (pad_no, pad_no)
    else:
        stride = (1,1,1)
        padding = (pad_no, pad_no, pad_no)

    # get convolution function
    conv_fn = getattr(F, 'conv%dd' % ndims)

    # compute CC squares
    I2 = I * I
    J2 = J * J
    IJ = I * J

    I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
    J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
    I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
    J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
    IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = cross * cross / (I_var * J_var + 1e-5)

    return -torch.mean(cc)

