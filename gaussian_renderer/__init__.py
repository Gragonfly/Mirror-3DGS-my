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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.general_utils import build_rotation, reflect_vector, matrix_to_quaternion, rot2quaternion
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, mirror_transform=None, stage=1):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity


    # 让镜面透明，这是在第三阶段 优化完整图像采用得到的，第二阶段只优化mask和plane，但是第二阶段在优化场景的同时需要镜面部分的场景，所以第二阶段也用得到，第一阶段只优化非镜面场景和mask，用不到
    if mirror_transform is not None:

        mirror_opacity_threshold = 0.2
        # mirror_attr = self.get_mirror.reshape(-1, 3)
        # print("all points: ", mirror_attr.shape[0])


        mirror_attri = pc.get_mirror.reshape(-1, 3)
        mask = (mirror_attri > mirror_opacity_threshold).any(axis=1)
        mask = mask.reshape(-1, 1) #(N, 1)

        # mirror_attri2 = torch.where(mask, torch.ones_like(mask), torch.zeros_like(mask))  # (N, 1)
        mirror_attri2 = mask.float()

        opacity = opacity * (1 - mirror_attri2)
        a, b, c, d = mirror_transform[0], mirror_transform[1], mirror_transform[2], mirror_transform[3]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if stage == 1:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    out_extra = {}

    if stage == 1:
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        p_hom = torch.cat([pc.get_xyz, torch.ones_like(pc.get_xyz[..., :1])], -1).unsqueeze(-1)
        p_view = torch.matmul(viewpoint_camera.world_view_transform.transpose(0, 1), p_hom)
        p_view = p_view[..., :3, :]
        depth = p_view.squeeze()[..., 2:3]
        depth = depth.repeat(1, 3)

        depth_render, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=depth,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        out_extra.update({"depth_render": depth_render})

    elif stage == 2:
        # mirror_attri = pc.get_mirror  #(N, 1)
        # mirror_opacity_threshold = 0.7
        # # mirror_attr = gaussians.get_mirror.reshape(-1, 1)
        # # print("all points: ", mirror_attr.shape[0])
        # mask = (mirror_attri > mirror_opacity_threshold)
        # mask = mask.view(-1).detach()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # get mirror point
        means3d = means3D.detach()
        means3d = means3d
        # n = torch.tensor([a, b, c], dtype=mirror_transform.dtype, device=mirror_transform.device)  # 法向量
        # n_norm_sq = torch.dot(n, n)  # 法向量的平方模
        # n_normalize = n / n_norm_sq
        # dot_products = torch.mm(means3d, n.view(-1, 1)).squeeze(-1)  # 计算 means 与 n 的点积
        # reflected_means = means3d - 2 * (dot_products + d).unsqueeze(-1) * n_normalize
        n = mirror_transform[:3]
        reflected_means = reflect_vector(means3d, n, d)

        # get mirror rotation

        # 计算反射后的旋转矩阵
        R = build_rotation(rotations.detach())  # (N, 3, 3)
        R_reflected = torch.zeros_like(R)
        for i in range(3):  # 遍历 R 的每一列向量
            Ri = means3d + R[:, :, i]  # 计算偏移后的主轴方向
            # Fi = Ri - 2 * ((Ri @ n_normalize + d) / (n_normalize @ n_normalize))[:, None] * n_normalize  # 镜像变换
            Fi = reflect_vector(Ri, n, d)
            R_reflected[:, :, i] = Fi - reflected_means  # 减去镜像中心

        # 转换为四元数形式
        reflected_rotations = rot2quaternion(R_reflected)

        #get mirror sh
        # 计算 F(mu_cam)
        mu_cam = viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        # dot_products = torch.sum(mu_cam * n, dim=-1, keepdim=True)  # 摄像机与法向量点积
        # F_mu_cam = mu_cam - 2 * (dot_products + d) * n / n_norm_sq
        F_mu_cam = reflect_vector(mu_cam, n, d)

        # 计算 d_hat
        d_hat = means3d - F_mu_cam
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2).detach()
        dir_pp = d_hat
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)


        rendered_image, radii = rasterizer(
            means3D = reflected_means,
            means2D = means2D.detach(),
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity.detach(),
            scales = scales.detach(),
            rotations = reflected_rotations,
            cov3D_precomp = None)
    elif stage == 3:
        # mirror_attri = pc.get_mirror  #(N, 1)
        # mirror_opacity_threshold = 0.7
        # # mirror_attr = gaussians.get_mirror.reshape(-1, 1)
        # # print("all points: ", mirror_attr.shape[0])
        # mask = (mirror_attri > mirror_opacity_threshold)
        # mask = mask.view(-1).detach()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        # get mirror point
        means3d = means3D
        # n = torch.tensor([a, b, c], dtype=mirror_transform.dtype, device=mirror_transform.device)  # 法向量
        n = mirror_transform[:3]
        # n_norm_sq = torch.dot(n, n)  # 法向量的平方模
        # n_normalize = n / n_norm_sq
        # dot_products = torch.mm(means3d, n.view(-1, 1)).squeeze(-1)  # 计算 means 与 n 的点积
        # reflected_means = means3d - 2 * (dot_products + d).unsqueeze(-1) * n_normalize
        reflected_means = reflect_vector(means3d, n, d)

        # get mirror rotation

        # 计算反射后的旋转矩阵
        R = build_rotation(rotations)  # (N, 3, 3)
        R_reflected = torch.zeros_like(R)
        for i in range(3):  # 遍历 R 的每一列向量
            Ri = means3d + R[:, :, i]  # 计算偏移后的主轴方向
            # Fi = Ri - 2 * ((Ri @ n_normalize + d) / (n_normalize @ n_normalize))[:, None] * n_normalize  # 镜像变换
            Fi = reflect_vector(Ri, n, d)
            R_reflected[:, :, i] = Fi - reflected_means  # 减去镜像中心

        # 转换为四元数形式
        reflected_rotations = rot2quaternion(R_reflected)

        #get mirror sh
        # 计算 F(mu_cam)
        mu_cam = viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)
        # dot_products = torch.sum(mu_cam * n, dim=-1, keepdim=True)  # 摄像机与法向量点积
        # F_mu_cam = mu_cam - 2 * (dot_products + d) * n / n_norm_sq
        F_mu_cam = reflect_vector(mu_cam, n, d)

        # 计算 d_hat
        d_hat = means3d - F_mu_cam
        shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = d_hat
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        rendered_image, radii = rasterizer(
            means3D = reflected_means,
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = reflected_rotations,
            cov3D_precomp = None)

    # mirror mask
    mirror_opacity = pc.get_mirror
    # mirror_color = torch.ones_like(means3D).detach()
    mirror_mask_pre = rasterizer(
        means3D = means3D,  #(N, 3)
        means2D = means2D,  #(N, 3)
        shs = None,
        colors_precomp = mirror_opacity, #(N, 3)
        opacities = opacity,  #(N, 1)
        scales = scales,  #(N, 3)
        rotations = rotations,  #(N, 4)
        cov3D_precomp = cov3D_precomp)[0]

    out_extra.update({"render": rendered_image,
            "mirror_mask_pre": mirror_mask_pre,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii})


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return out_extra
