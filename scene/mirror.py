import torch
import torch.nn as nn
from torch import no_grad


from utils import ransac
import numpy as np

from utils.general_utils import get_linear_noise_func


mirror_stage = 20_000
s2_stage = 5_000

class Mirror:

    def __init__(self):
        self.best_eq = None
        self.optimizer = None

    def train_setting(self):
        l = [
            {'params': [self.best_eq],
             'lr': 0.00016,
             "name": "eq"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.01, eps=1e-15)

        self.specular_scheduler_args = get_linear_noise_func(lr_init=1.6e-4,
                                                             lr_final=1.6e-6,
                                                             lr_delay_mult=0.01,
                                                             max_steps=25000)
        self.specular_scheduler_args2 = get_linear_noise_func(lr_init=1.6e-6,
                                                             lr_final=1.6e-8,
                                                             lr_delay_mult=0.01,
                                                             max_steps=60000)



    @torch.no_grad()
    def compute_mirror_plane(self, mirror_point, mirror_opacity_threshold=0.7, sansac_threshold=0.05):

        # select point
        mirror_points = mirror_point.detach().cpu().numpy()  # (N, 3)
        print("chosen points: ", mirror_points.shape[0])
        best_eq, _ = ransac.Plane(mirror_points, sansac_threshold)

        a, b, c, d = best_eq[0], best_eq[1], best_eq[2], best_eq[3]  # 平面参数
        print("train plain~ ", best_eq)

        # 调用函数绘图
        # plot_plane_and_points(a,  b, c, d, mirror_points)

        self.best_eq = nn.Parameter(torch.tensor([a, b, c, d]).cuda().float().requires_grad_(True))

        # mirror_transform
        mirror_transform = np.array([
            1 - 2 * a * a, -2 * a * b, -2 * a * c, -2 * a * d,
            -2 * a * b, 1 - 2 * b * b, -2 * b * c, -2 * b * d,
            -2 * a * c, -2 * b * c, 1 - 2 * c * c, -2 * c * d,
            0, 0, 0, 1
        ]).reshape(4, 4)
        mirror_transform = torch.as_tensor(mirror_transform, dtype=torch.float, device="cuda")

        return mirror_transform

    @torch.no_grad()
    def set_mirror_pos(self, _best_eq):
        a, b, c, d = _best_eq[0], _best_eq[1], _best_eq[2], _best_eq[3]
        self.best_eq = nn.Parameter(torch.tensor([a, b, c, d]).cuda().float().requires_grad_(True))

        # mirror_transform
        mirror_transform = np.array([
            1 - 2 * a * a, -2 * a * b, -2 * a * c, -2 * a * d,
            -2 * a * b, 1 - 2 * b * b, -2 * b * c, -2 * b * d,
            -2 * a * c, -2 * b * c, 1 - 2 * c * c, -2 * c * d,
            0, 0, 0, 1
        ]).reshape(4, 4)
        mirror_transform = torch.as_tensor(mirror_transform, dtype=torch.float, device="cuda")

        return mirror_transform

    @torch.no_grad()
    def fit_plane_least_squares(self, points):
        """
        使用最小二乘法拟合一组 3D 点的平面。

        Args:
            points (torch.Tensor): 点的坐标张量，形状为 (N, 3)，每行表示一个点 (x, y, z)。

        Returns:
            torch.Tensor: 平面反射变换矩阵，形状为 (4, 4)。
        """
        assert points.shape[1] == 3, "输入点的形状应为 (N, 3)"

        # 提取坐标
        X = points[:, 0]  # x 坐标
        Y = points[:, 1]  # y 坐标
        Z = points[:, 2]  # z 坐标

        # 构造矩阵 A 和向量 b
        A = torch.stack([X, Y, torch.ones_like(X)], dim=1)  # (N, 3)
        b = Z  # (N,)

        # 使用 torch.linalg.lstsq 求解 [a, b, d']，其中 c = 1
        solution = torch.linalg.lstsq(A, b).solution
        a, b, d = solution.squeeze()  # 解是一个张量，解包为 a, b, d
        c = -1.0

        # # 保存平面参数
        self.best_eq = nn.Parameter(
            torch.tensor([a, b, c, d], device=points.device, dtype=points.dtype).requires_grad_(True)
        )
        # print("get fitted plane is ",(a, b, c, d))
        #
        # # 计算平面反射变换矩阵
        # mirror_transform = torch.tensor([
        #     [1 - 2 * a * a, -2 * a * b, -2 * a * c, -2 * a * d],
        #     [-2 * a * b, 1 - 2 * b * b, -2 * b * c, -2 * b * d],
        #     [-2 * a * c, -2 * b * c, 1 - 2 * c * c, -2 * c * d],
        #     [0, 0, 0, 1]
        # ], dtype=points.dtype, device=points.device)

    def get_plane_error(self, mirror_points, min_opacity=0.5):
        """enforcing the mirror points close to the plane"""

        a, b, c, d = self.best_eq[0].detach(), self.best_eq[1].detach(), self.best_eq[2].detach(), self.best_eq[3].detach()  # 平面参数
        dist = ((mirror_points[:, 0] * a + mirror_points[:, 1] * b + mirror_points[:, 2] * c + d
                ) / torch.sqrt(a ** 2 + b ** 2 + c ** 2)).abs().mean()

        return dist

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "eq":
                if iteration <= mirror_stage + s2_stage:
                    lr = self.specular_scheduler_args(iteration)
                else:
                    lr = self.specular_scheduler_args2(iteration)
                param_group['lr'] = lr
                return lr

