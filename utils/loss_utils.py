#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def zero_one_loss(img):
    zero_epsilon = 1e-3
    val = torch.clamp(img, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def get_tv_loss(
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    pad: int = 1,
    step: int = 1,
) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss


def calculate_Lplane_with_normals(normals, num_groups):
    """
    计算 Lplane，其中输入的 xyz 表示点的法向量。
    Args:
        normals (torch.Tensor): 输入张量，形状为 (N, 3)，表示每个点的法向量。
        num_groups (int): 随机抽取的小组数量。
    Returns:
        float: Lplane 的值。
    """
    N = normals.shape[0]
    assert normals.shape[1] == 3, "normals 的形状应为 (N, 3)"
    assert N >= 3, "必须至少有 3 个点"

    # 随机生成 num_groups 组，每组 3 个点，组内点不重复
    indices = torch.randint(0, N, (num_groups, 3))  # 形状 (num_groups, 3)

    # 确保组内点不重复
    while (indices[:, 0] == indices[:, 1]).any() or (indices[:, 0] == indices[:, 2]).any() or (
            indices[:, 1] == indices[:, 2]).any():
        mask = (indices[:, 0] == indices[:, 1]) | (indices[:, 0] == indices[:, 2]) | (indices[:, 1] == indices[:, 2])
        indices[mask] = torch.randint(0, N, (mask.sum(), 3))

    # 提取法向量
    selected_points = normals[indices]  # 形状 (num_groups, 3, 3)

    # 提取组内的三个法向量
    na = selected_points[:, 0]  # 第一个点的法向量
    nb = selected_points[:, 1]  # 第二个点的法向量
    nc = selected_points[:, 2]  # 第三个点的法向量

    # 计算 na × nb
    cross_product = torch.cross(na, nb, dim=1)  # 形状 (num_groups, 3)

    # 计算 |na × nb · nc|
    dot_product = torch.abs(torch.sum(cross_product * nc, dim=1))  # 点积并取绝对值

    # 计算 Lplane
    Lplane = dot_product.mean().item()
    return Lplane
