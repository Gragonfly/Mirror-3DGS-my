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
import sys
from datetime import datetime
import numpy as np
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from PIL import Image

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def PILtoTorch2(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL.convert('L'))) / 255.0
    return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)


# 定义反射函数 F
def reflect_vector(v, n, d):
    n_norm_sq = torch.dot(n, n)  # 法向量的平方模
    return v - 2 * (torch.sum(v * n, dim=-1, keepdim=True) + d) * n / n_norm_sq


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def plot_plane_and_points(A, B, C, D, points):
    """
    绘制三维平面和散点图。

    参数：
        A, B, C, D : 平面 Ax + By + Cz + D = 0 的参数。
        points : ndarray, shape (N, 3)，散点的坐标数组。
    """
    # 创建一个三维坐标系
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 定义网格范围
    x_range = np.linspace(-10, 10, 50)
    y_range = np.linspace(-10, 10, 50)
    x, y = np.meshgrid(x_range, y_range)

    # 计算平面上的 z 值
    if C != 0:  # 确保 C 不为零
        z = (-A * x - B * y - D) / C
    else:
        raise ValueError("参数 C 不能为零，这会导致无法计算 z 值。")

    # 绘制平面
    ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color='lightblue')

    # 绘制散点
    points = np.asarray(points)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red', label='Points_mirror')

    # 设置轴标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 显示图例
    ax.legend()

    # 显示图像
    plt.title("Plane and Points")
    plt.show()
def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

def get_minimum_axis(scales, rotations):
    sorted_idx = torch.argsort(scales, descending=False, dim=-1)
    R = build_rotation(rotations)
    R_sorted = torch.gather(R, dim=2, index=sorted_idx[:,None,:].repeat(1, 3, 1)).squeeze()
    x_axis = R_sorted[:,0,:] # normalized by defaut

    return x_axis



def filter_far_points_adaptive_torch(points, method="percentile", std_factor=2.0, percentile=90):
    """
    自适应过滤离中心太远的点，支持 PyTorch 张量输入。

    :param points: 输入点集 (N, 3)，PyTorch 张量。
    :param method: 自适应阈值的计算方法，"std" 或 "percentile"。
    :param std_factor: 如果使用标准差法，阈值为 mean + std_factor * std。
    :param percentile: 如果使用百分位法，阈值为距离的 percentile。
    :return: 过滤后的点集，PyTorch 张量。
    """
    # 计算点的中心
    center = points.mean(dim=0)

    # 计算每个点到中心的欧几里得距离
    distances = torch.norm(points - center, dim=1)

    # 根据方法计算自适应阈值
    if method == "std":
        mean_distance = distances.mean()
        std_distance = distances.std()
        threshold = mean_distance + std_factor * std_distance
    elif method == "percentile":
        threshold = torch.quantile(distances, percentile / 100.0)
    else:
        raise ValueError("Unsupported method. Choose 'std' or 'percentile'.")

    # 保留距离小于等于阈值的点
    mask = distances <= threshold
    filtered_points = points[mask]
    print(f"Adaptive threshold: {threshold.item():.2f} (method: {method})")
    return filtered_points



import torch
import numpy as np
from PIL import Image

def flip_image_horizontally(image_tensor):
    """
    Flip a single-channel image horizontally (mirror effect).

    :param image_tensor: Input image as a PyTorch tensor (1, H, W).
    :return: Horizontally flipped image as a PyTorch tensor (1, H, W).
    """
    # Ensure the input tensor is 3D with shape (1, H, W)
    if len(image_tensor.shape) != 3 or image_tensor.shape[0] != 1:
        raise ValueError("Input tensor must have shape (1, H, W).")

    # Remove the channel dimension to work with (H, W)
    image_np = image_tensor[0].cpu().numpy()

    # Convert NumPy array to PIL Image
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode='L')

    # Flip the image horizontally
    flipped_image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # Convert flipped PIL image back to NumPy array
    flipped_image_np = np.array(flipped_image_pil) / 255.0

    # Convert NumPy array back to PyTorch tensor and add the channel dimension
    flipped_image_tensor = torch.from_numpy(flipped_image_np).unsqueeze(0)

    return flipped_image_tensor


# 转换旋转矩阵为四元数
def matrix_to_quaternion(R):
    q = torch.zeros((R.size(0), 4), dtype=R.dtype, device=R.device)
    q[:, 0] = 0.5 * torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0])
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0])
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0])
    assert not torch.isnan(q).any(), "Quaternion contains NaN values"
    return q

def rot2quaternion(rotation_matrix):
    # 将 PyTorch tensor 转换为 NumPy 数组
    rotation_matrix = rotation_matrix.cpu().detach().numpy()

    # 使用 scipy 的 Rotation.from_matrix 计算四元数
    r3 = Rotation.from_matrix(rotation_matrix)
    qua = r3.as_quat()  # 返回四元数 [x, y, z, w]
    q = torch.tensor(qua, dtype=torch.float32, device="cuda")[:, [3,0,1,2]]
    return q# 转回 torch.tensor 并返回


def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img