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
import sys

import torch
from IPython.terminal.ipapp import flags
from PIL import Image
from typing import NamedTuple
import pyransac3d as pyrsc
from networkx.classes import neighbors
from numpy.distutils.system_info import xft_info
from pandas.core.config_init import pc_max_info_rows_doc

from tensorboard.summary.v1 import image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.general_utils import PILtoTorch, plot_plane_and_points, flip_image_horizontally
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    mask: np.array
    mask_pre: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        folders = image_path.split(os.sep)

        if 'real' in folders:
            # 真实数据集
            # mask_path = image_path.replace('images', 'masks')
            # test pre_mask
            mask_path2 = image_path.replace('images', 'pre_mask')
            mask_path = image_path.replace('images', 'masks')
        else:
            # 合成数据集
            mask_path = image_path.replace('images/Image', 'masks/MirrorMask')
            mask_path2 = image_path.replace('images/Image', 'pre_mask/MirrorMask')


        # mirror_mask = Image.open(mask_path).convert('L')
        # test pre_mask
        mirror_mask = Image.open(mask_path).convert('L')
        mirror_mask_pre = Image.open(mask_path2)


        # # 假设 mirror_mask 是一个 PIL 图像
        # mirror_mask_np = np.array(mirror_mask)  # 将 PIL 图像转换为 NumPy 数组
        #
        # # 确保数据在 0 到 1 范围内，如果需要的话
        # mirror_mask_np = mirror_mask_np / 255.0 if mirror_mask_np.max() > 1 else mirror_mask_np
        #
        # # 应用阈值操作
        # resized_mirror_mask = np.where(mirror_mask_np >= 0.5, 1.0, 0.0)
        #
        # level_mirror_resized_mirror_mask = flip_image_horizontally(mask_path)
        #
        # level_mirror_resized_mirror_mask_np = np.array(level_mirror_resized_mirror_mask)
        # level_mirror_resized_mirror_mask_np = level_mirror_resized_mirror_mask_np / 255.0 if level_mirror_resized_mirror_mask_np.max() > 1 else level_mirror_resized_mirror_mask_np
        # resized_level_mirror_resized_mirror_mask_np = np.where(level_mirror_resized_mirror_mask_np >= 0.5, 1.0, 0.0)


        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mirror_mask, mask_pre = mirror_mask_pre,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    # point3d_pos = pcd.points
    # print("begin precompute mirror position ~")
    # # read all mask
    # dic_mask = {}
    # cam_num = len(cam_infos)
    #
    # for i in range(cam_num):
    #     single_camera = cam_infos[i]
    #     image_path = os.path.join(path, reading_dir)
    #     image_name_get = single_camera.image_name
    #     # image_path = os.path.join(image_path, os.path.basename(image_name_get + ".png"))
    #     image_path = os.path.join(image_path, os.path.basename(image_name_get + ".jpg"))
    #
    #     # mask_path = image_path.replace('images/Image', 'masks/MirrorMask')
    #     mask_path = image_path.replace('images', 'masks')
    #     mirror_mask = Image.open(mask_path).convert('L')
    #     resized_mirror_mask = PILtoTorch(mirror_mask, mirror_mask.size)
    #     resized_mirror_mask[resized_mirror_mask >= 0.5] = 1.0
    #     resized_mirror_mask[resized_mirror_mask < 0.5] = 0.0
    #     dic_mask.update({image_name_get : resized_mirror_mask})
    #
    # mirror_edge_idx = set()
    # invalid_num = 0
    # for i in cam_extrinsics:
    #     cam_single = cam_extrinsics[i]
    #     feature_num = len(cam_single.point3D_ids)
    #     image_name_idx = cam_single.name.split(".")[0]
    #     image_mask = dic_mask[image_name_idx]
    #     for j in range(feature_num):
    #         point3d_idx = cam_single.point3D_ids[j]
    #         if point3d_idx != -1:
    #             point2d_pos = cam_single.xys[j]  #(x, y)
    #             point2d_pos = np.round(point2d_pos)
    #             x, y = point2d_pos[0], point2d_pos[1]
    #             x, y = int(x), int(y)
    #             # look up in mask if it is in edge then collect it
    #             if 0 < x < image_mask.shape[2]-1 and 0 < y < image_mask.shape[1]-1:
    #                 neighbours = image_mask[0][y-1:y+2, x-1:x+2]
    #                 is_edge = neighbours.max() != neighbours.min()
    #                 if is_edge:
    #                     if point3d_idx < len(point3d_pos):
    #                         if point3d_idx not in mirror_edge_idx:
    #                             mirror_edge_idx.add(point3d_idx)
    #                     else:
    #                         invalid_num = invalid_num + 1
    # print(f"load finished~ all invalid_num is {invalid_num}~ all mirror_edge_num is {len(mirror_edge_idx)}")

    # mirror_edge = []
    # for k in mirror_edge_idx:
    #     mirror_edge.append(point3d_pos[k])
    #
    # # 将 mirror_edge 转换为 NumPy 数组，方便操作
    # mirror_edge = np.array(mirror_edge)

    # 检查是否有数据
    # best_eq = None
    # if len(mirror_edge) == 0:
    #     print("No 3D points found for edges.")
    # else:
    #
    #     plane1 = pyrsc.Plane()
    #
    #     best_eq, best_inliers = plane1.fit(mirror_edge, 0.01)
    #     # 示例参数
    #     A, B, C, D = best_eq[0], best_eq[1], best_eq[2], best_eq[3]  # 平面参数 nx, ny, nz, b
    #     print("sfm  ", best_eq)
    #
    #     # 调用函数绘图
    #     plot_plane_and_points(A, B, C, D, mirror_edge)

    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    folders = path.split(os.sep)
    flags = False #   0 means synthetic   1 means real

    if "real" in folders:
        flags = True

    flags = False

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)


        if flags:
            # real 数据集
            focal = contents["fx"]
        else:
            # 合成数据集
            fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):


            if flags:
                # real 数据集
                cam_name = os.path.join(path, frame["file_path"])
            else:
                # 合成数据集
                cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            if flags:
                # real数据集
                mask_path = image_path.replace('images', 'masks')
                mask_path2 = image_path.replace('images', 'pre_mask')
            else:
                # 合成数据集
                mask_path = image_path.replace('images/Image', 'masks/MirrorMask')
                mask_path2 = image_path.replace('images/Image', 'pre_mask/MirrorMask')
                mask_path = image_path.replace('images', 'masks')
                mask_path2 = image_path.replace('images', 'pre_mask')
                mask_path2 = mask_path2.replace('png','jpg')

            mirror_mask = Image.open(mask_path).convert('L')
            mirror_mask_pre = Image.open(mask_path2)

            if flags:
                # real 数据集
                fovx = focal2fov(focal, image.size[0])
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mask=mirror_mask, mask_pre = mirror_mask_pre,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}