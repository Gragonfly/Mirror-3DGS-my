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

import os
import numpy as np
import torch
from random import randint

import torchvision
from triton.language import dtype

from metrics import evaluate
from scene.mirror import Mirror
from utils.loss_utils import l1_loss, ssim, zero_one_loss, get_tv_loss, calculate_Lplane_with_normals
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, filter_far_points_adaptive_torch
import uuid
import torchviz
from utils.general_utils import colormap
from tqdm import tqdm
from utils.image_utils import psnr, psnr2
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import sys
import os


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

mirror_stage = 20_000
s2_stage = 5_000


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    mirror_plane = Mirror()

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        mirror_pre = render_pkg["mirror_mask_pre"]
        gt_image = viewpoint_cam.original_image.cuda() #(3, H, W)
        gt_mirror_mask = viewpoint_cam.mirror_mask.cuda().repeat(3, 1, 1) #(3, H, W)
        # test pre_mask
        gt_mirror_mask_pre = viewpoint_cam.mirror_mask_pre.cuda() #(3, H, W)


        # 计算 mask 区域：
        # 设定阈值，假设 mask 的红色或白色区域像素值较高（接近 1）
        mask_threshold = 0.5  # 适用于归一化的图像（0-1）
        mask_region = gt_mirror_mask_pre.sum(dim=0) > mask_threshold  # (H, W)，找到非黑色区域

        # 使用 torch.where 选择对应区域
        gt_image_s1 = torch.where(mask_region.unsqueeze(0).repeat(3,1,1), gt_mirror_mask_pre, gt_image)  # (3, H, W)

        # render
        # gt_mirror_render = torch.where(mask_region.unsqueeze(0).repeat(3, 1, 1), 1, 0)
        # gt_image_s1 = torch.clamp(gt_image + gt_mirror_mask, 0.0, 1.0) #(3, H, W)
        # gt_fh_mask = viewpoint_cam.flip_horizon_mask.cuda().repeat(3, 1, 1) #(3, H, W)

        # Loss
        if iteration < mirror_stage:
            mirror_l1 = l1_loss(mirror_pre, gt_mirror_mask_pre)
            zore_one_loss = zero_one_loss(mirror_pre)
            Ll1 = l1_loss(image, gt_image_s1)
            # depth_render = render_pkg["depth_render"]
            # tv_loss = get_tv_loss(gt_image_s1, depth_render)

            # mirror_opacity_threshold = 0.7
            # mirror_attr = gaussians.get_mirror.reshape(-1, 1)
            # mask = (mirror_attr > mirror_opacity_threshold)
            # mask = mask.view(-1)
            #
            # mirror_shortest_axis = gaussians.get_minimum_scaling[mask]
            # normal = gaussians.get_minimum_axis[mask]
            # shortest_axis_loss = 0
            # normal_consistency_loss = 0
            # if mirror_shortest_axis.shape[0] > 10:
            #     shortest_axis_loss = torch.abs(mirror_shortest_axis).mean()
            #     num_group = round((iteration - 100) / 2) + 100
            #
            #     normal_consistency_loss = calculate_Lplane_with_normals(normal, num_group)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image_s1)) + 0.4 * mirror_l1 + 0.001 * zore_one_loss

        elif iteration < mirror_stage + s2_stage:
            # get mirror render result
            if iteration == mirror_stage:

                mirror_opacity_threshold = 0.7
                mirror_attr = gaussians.get_mirror.reshape(-1, 3)
                # print("all points: ", mirror_attr.shape[0])
                mask = (mirror_attr > mirror_opacity_threshold).all(axis=1)
                mask = mask.view(-1)

                # select point
                xyz = gaussians.get_xyz
                mirror_points = xyz[mask]  # (N, 3)

                plane_loss = 0

                mirror_points_filter = filter_far_points_adaptive_torch(mirror_points, "percentile", percentile=90)
                mirror_plane.fit_plane_least_squares(mirror_points_filter)

                # mirror_transform = mirror_plane.set_mirror_pos(scene.best_eq)
                mirror_l1 = l1_loss(mirror_pre, gt_mirror_mask)
                zore_one_loss = zero_one_loss(mirror_pre)
                # plane_loss = 0
                # shortest_axis_loss = 0
                # normal_consistency_loss = 0
                mirror_plane.train_setting()
            else:
                # best_eq = mirror_plane.best_eq
                mirror_opacity_threshold = 0.7
                mirror_attr = gaussians.get_mirror.reshape(-1, 3)
                # print("all points: ", mirror_attr.shape[0])
                mask = (mirror_attr > mirror_opacity_threshold).all(axis=1)
                mask = mask.view(-1)

                # select point
                xyz = gaussians.get_xyz
                mirror_points = xyz[mask]  # (N, 3)

                plane_loss = mirror_plane.get_plane_error(mirror_points)

                mirror_l1 = l1_loss(mirror_pre, gt_mirror_mask)
                zore_one_loss = zero_one_loss(mirror_pre)


                # mirror_shortest_axis = gaussians.get_minimum_scaling[mask]
                # normal = gaussians.get_minimum_axis[mask]
                # shortest_axis_loss = 0
                # normal_consistency_loss = 0
                # if mirror_shortest_axis.shape[0] > 10:
                #     shortest_axis_loss = torch.abs(mirror_shortest_axis).mean()
                #     num_group = round((iteration - 100) / 2) + 100
                #
                #     normal_consistency_loss = calculate_Lplane_with_normals(normal, num_group)


            # if iteration % 50 ==0:
            #     print("mirror_plane: ",mirror_plane.best_eq)

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, mirror_transform=mirror_plane.best_eq, stage=2)
            mirror_image = render_pkg["render"]
            # mirror_image_pre_inverse = render_pkg["mirror_mask_pre"]
            # inverse_mirror_l1 = l1_loss(mirror_image_pre_inverse, gt_fh_mask)
            final_image = image * (1 - gt_mirror_mask) + mirror_image * gt_mirror_mask

            Ll1_mirror = l1_loss(final_image, gt_image)
            # Ll1 = l1_loss(image, gt_image_s1)
            loss = (1.0 - opt.lambda_dssim) * Ll1_mirror + opt.lambda_dssim * (1.0 - ssim(final_image, gt_image)) + 0.01 * mirror_l1 + 0.001 * zore_one_loss + 0.2 * plane_loss
            # loss = (1.0 - opt.lambda_dssim) * Ll1_mirror + opt.lambda_dssim * (1.0 - ssim(final_image, gt_image)) + 0.2 * plane_loss


        else:  #STAGE 3  train for mask
            # if mirror_plane.best_eq.requires_grad is True:
            #     mirror_plane.best_eq.requires_grad_(False)
        #         print(gaussians.get_xyz.requires_grad)
        #         print(mirror_plane.best_eq.requires_grad)

            best_eq = mirror_plane.best_eq
            mirror_opacity_threshold = 0.7
            mirror_attr = gaussians.get_mirror.reshape(-1, 3)
            # print("all points: ", mirror_attr.shape[0])
            mask = (mirror_attr > mirror_opacity_threshold).all(axis=1)
            mask = mask.view(-1)

            # select point
            xyz = gaussians.get_xyz
            mirror_points = xyz[mask]  # (N, 3)

            plane_loss = mirror_plane.get_plane_error(mirror_points)
            # a, b, c, d = best_eq[0], best_eq[1], best_eq[2], best_eq[3]  # 平面参数
            # if iteration % 50 == 0:
            # #     print("mirror_point_size ", mirror_points.shape[0])
            #     print("train plain~ ", mirror_plane.best_eq)
            #     print("plane loss ", plane_loss)
            # mirror_transform
            # mirror_transform = torch.tensor([
            #     [1 - 2 * a * a, -2 * a * b, -2 * a * c, -2 * a * d],
            #     [-2 * a * b, 1 - 2 * b * b, -2 * b * c, -2 * b * d],
            #     [-2 * a * c, -2 * b * c, 1 - 2 * c * c, -2 * c * d],
            #     [0, 0, 0, 1]
            # ], dtype=torch.float, device="cuda")


            mirror_l1 = l1_loss(mirror_pre, gt_mirror_mask)
            zore_one_loss = zero_one_loss(mirror_pre)

            render_pkg = render(viewpoint_cam, gaussians, pipe, bg, mirror_transform=best_eq, stage=3)
            mirror_image = render_pkg["render"]

            final_image = image * (1 - mirror_pre) + mirror_image * mirror_pre

            Ll1 = l1_loss(final_image, gt_image)

            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(final_image, gt_image)) + 0.4 * mirror_l1 + 0.001 * zore_one_loss + 0.2 * plane_loss

        if iteration == opt.iterations:
            print("best eq: ", mirror_plane.best_eq)

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, mirror_plane, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # if iteration < mirror_stage:
                    #     gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    # else:
                    #     gaussians.densify_and_prune(opt.densify_grad_threshold/2, 0.005, scene.cameras_extent,size_threshold/2)
                    # 修改2个阈值，让地面上的裂纹能生成（可能
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration > mirror_stage:
                    mirror_plane.optimizer.step()
                    mirror_plane.optimizer.zero_grad(set_to_none = True)
                    mirror_plane.update_learning_rate(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

    return mirror_plane.best_eq

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    tb_runs_path = os.path.join(args.model_path, "runs")
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(tb_runs_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, mirror_plane:Mirror, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        # tb_writer.add_scalar('train_loss_patches/mirror_l1', mirror_l1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                l1_fused_test = 0.0
                psnr_fused_test =0.0
                l1_mirror_part_test = 0.0
                psnr_mirror_part_test = 0.0
                count_mirror = 0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    mirror_mask_pre = torch.clamp(render_pkg["mirror_mask_pre"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    gt_mirror_mask = torch.clamp(viewpoint.mirror_mask.to("cuda"), 0.0, 1.0)
                    gt_mirror_mask_pre = torch.clamp(viewpoint.mirror_mask_pre.to("cuda"), 0.0, 1.0)
                    gt_mirror_part = gt_mirror_mask * gt_image
                    # gt_image_s1 = torch.clamp(gt_image + gt_mirror_mask, 0.0, 1.0)

                    # 计算 mask 区域：
                    # 设定阈值，假设 mask 的红色或白色区域像素值较高（接近 1）
                    mask_threshold = 0.5  # 适用于归一化的图像（0-1）
                    mask_region = gt_mirror_mask_pre.sum(dim=0) > mask_threshold  # (H, W)，找到非黑色区域

                    # 使用 torch.where 选择对应区域
                    gt_image_s1 = torch.where(mask_region.unsqueeze(0).repeat(3, 1, 1), gt_mirror_mask_pre, gt_image)  # (3, H, W)

                    # gt_mirror_part = torch.where(mask_region.unsqueeze(0).repeat(3, 1, 1), gt_image, torch.zeros_like(gt_image))

                    # gt_mirror_render = torch.where(mask_region.unsqueeze(0).repeat(3, 1, 1), torch.ones_like(gt_mirror_mask), torch.zeros_like(gt_mirror_mask))

                    # gt_fh_mask = torch.clamp(viewpoint.flip_horizon_mask.to("cuda"), 0.0, 1.0)
                    if iteration < mirror_stage or iteration > mirror_stage + s2_stage:

                        depth = render_pkg["depth_render"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')

                    if iteration >= mirror_stage and iteration < mirror_stage + s2_stage:
                        mirror_render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                                       mirror_transform=mirror_plane.best_eq, stage=2)
                        mirror_image = mirror_render_pkg["render"]
                        # flip_horizon_mask_pre = mirror_render_pkg["mirror_mask_pre"]
                        mirror_part = mirror_image * gt_mirror_mask
                        image_fused = image * (1 - gt_mirror_mask) + mirror_image * gt_mirror_mask

                    elif iteration >= mirror_stage + s2_stage:
                        mirror_render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs,
                                                       mirror_transform=mirror_plane.best_eq, stage=2)
                        mirror_image = mirror_render_pkg["render"]
                        # flip_horizon_mask_pre = mirror_render_pkg["mirror_mask_pre"]
                        mirror_part = mirror_image * mirror_mask_pre
                        image_fused = image * (1 - mirror_mask_pre) + mirror_image * mirror_mask_pre

                    if tb_writer and (idx < 30):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mirror_mask_pre".format(viewpoint.image_name), mirror_mask_pre[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image_s1[None] - image[None]).abs(), global_step=iteration)
                        if iteration < mirror_stage or iteration > mirror_stage + s2_stage:
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        if iteration >= mirror_stage:
                            tb_writer.add_images(config['name'] + "_view_{}/mirror_part".format(viewpoint.image_name), mirror_part[None], global_step=iteration)
                            # tb_writer.add_images(config['name'] + "_view_{}/flip_horizon_mask_pre".format(viewpoint.image_name), flip_horizon_mask_pre[None], global_step=iteration)
                            # tb_writer.add_images(config['name'] + "_view_{}/mirror_image".format(viewpoint.image_name), mirror_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/image_fused".format(viewpoint.image_name), image_fused[None], global_step=iteration)
                            # tb_writer.add_images(config['name'] + "_view_{}/errormap_fused".format(viewpoint.image_name), (gt_image[None] - image_fused[None]).abs(), global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth_s1".format(viewpoint.image_name), gt_image_s1[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/gt_mirror_mask".format(viewpoint.image_name), gt_mirror_mask[None], global_step=iteration)
                            # tb_writer.add_images(config['name'] + "_view_{}/gt_fh_mask".format(viewpoint.image_name), gt_fh_mask[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image_s1).mean().double()
                    psnr_test += psnr(image, gt_image_s1).mean().double()
                    if iteration >= mirror_stage:
                        l1_fused_test += l1_loss(image_fused, gt_image).mean().double()
                        psnr_fused_test += psnr(image_fused, gt_image).mean().double()

                        is_nonzero = torch.any(gt_mirror_mask != 0)
                        if is_nonzero:
                            # l1 loss的计算可能还是存在问题，但是无关紧要
                            l1_mirror_part_test += l1_loss(mirror_part, gt_mirror_mask).mean().double()
                            psnr_mirror_part_test += psnr2(mirror_part, gt_mirror_part, gt_mirror_mask).mean().double()
                            count_mirror += 1
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                if iteration >= mirror_stage:
                    l1_fused_test /= len(config['cameras'])
                    psnr_fused_test /= len(config['cameras'])
                    if count_mirror != 0:
                        l1_mirror_part_test /= count_mirror
                        psnr_mirror_part_test /= count_mirror
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                print("\n[ITER {}] Evaluating {}:fused L1 {} PSNR {}".format(iteration, config['name'], l1_fused_test, psnr_fused_test))
                print("\n[ITER {}] Evaluating {}:mirror part L1 {} PSNR {}".format(iteration, config['name'], l1_mirror_part_test, psnr_mirror_part_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    if iteration >= mirror_stage:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint -fused l1_loss', l1_fused_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint -fused psnr', psnr_fused_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint -mirror part l1_loss', l1_mirror_part_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint -mirror part psnr', psnr_mirror_part_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, best_eq):
    import os

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    mirror_part_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mirror_part")
    mirror_part_gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mirror_part_gt")
    mask_gt_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)
    os.makedirs(mirror_part_path, exist_ok=True)
    os.makedirs(mirror_part_gt_path, exist_ok=True)
    os.makedirs(mask_gt_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        mirror_mask_pre = render_pkg["mirror_mask_pre"]
        gt = view.original_image[0:3, :, :]
        gt_mask = view.mirror_mask.repeat(3, 1, 1)
        render_pkg2 = render(view, gaussians, pipeline, background, mirror_transform=best_eq, stage=3)
        mirror_image = render_pkg2["render"]
        mirror_part = mirror_image * mirror_mask_pre
        mirror_part_gt = gt * gt_mask
        image_fused = rendering * (1 - mirror_mask_pre) + mirror_image * mirror_mask_pre

        torchvision.utils.save_image(image_fused, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mirror_part, os.path.join(mirror_part_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(mirror_part_gt, os.path.join(mirror_part_gt_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_mask, os.path.join(mask_gt_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, best_eq):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # if not skip_train:
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, best_eq)

        # if not skip_test:
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, best_eq)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3000, 10_000, 15000, 20_000, 25000, 30_000, 35000, 40_000, 45000, 50000, 55000, 60000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3000, 10_000, 15000, 20_000, 25000, 30_000, 35000, 40_000, 45000, 50000, 55000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    log_file_path = "log.txt"
    log_file = os.path.join(args.model_path, log_file_path)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    type = sys.getfilesystemencoding()
    sys.stdout = Logger(log_file)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    best_eq = training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")

    # rendering
    print(f'\nStarting Rendering~')
    # best_eq = torch.tensor([1.4180,  0.2578, -0.9454, -9.4957], device="cuda")
    render_sets(lp.extract(args), -1, pp.extract(args), best_eq = best_eq)
    print("\nRendering complete.")

    # calc metrics
    print("\nStarting evaluation...")
    evaluate([str(args.model_path)])
    print("\nEvaluating complete.")
