# import os
# from PIL import Image
# import numpy as np
# from skimage import filters, morphology
# import matplotlib.pyplot as plt  # 用于显示图像
#
# def process_mask(image_path, mask_path, output_folder):
#     # 读取原始图片和 mask
#     image_name = os.path.basename(image_path).split(".")[0]
#     mask = Image.open(mask_path).convert('L')
#
#     # 将 mask 图像进行二值化，设定阈值
#     threshold = 128  # 设定一个合适的阈值
#     binary_mask = np.array(mask) > threshold  # 二值化图像，mask区域为白色
#
#     # 使用 Sobel 算子进行边缘检测，获取mask的边缘
#     edge_mask = filters.sobel(binary_mask.astype(float))  # 使用 sobel 算子进行边缘检测
#
#     # 阈值化边缘图像，去除弱边缘
#     edge_threshold = np.percentile(edge_mask, 90)  # 设置90%的强度作为阈值
#     edge_mask = edge_mask > edge_threshold  # 获取强边缘
#
#     # 应用腐蚀操作去除噪点
#     edge_mask = morphology.erosion(edge_mask, morphology.square(3))
#
#     # 保留黑色背景
#     black_background = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
#
#     # 创建一个空白图像，用于存放最终图像
#     final_image = np.copy(black_background)
#
#     # 保留原始mask粗50个像素的白色边界，其余mask边界内部为红色
#     eroded_mask = morphology.erosion(binary_mask, morphology.square(50))  # 缩小mask边界
#     red_area = binary_mask & ~eroded_mask  # 提取内部红色区域
#     white_boundary = eroded_mask & ~binary_mask  # 提取白色边界
#
#     final_image[white_boundary] = [255, 255, 255]  # 白色边界
#     final_image[red_area] = [255, 255, 255]  # 红色区域
#
#     # 转换回 PIL 图像
#     final_image = Image.fromarray(final_image.astype(np.uint8))
#
#     # 确保输出目录存在
#     os.makedirs(output_folder, exist_ok=True)
#     output_path = os.path.join(output_folder, f"{image_name}.jpg")
#
#     # 保存最终处理后的 mask 为 jpg 格式
#     final_image.save(output_path, "JPEG")
#     print(f"Processed mask saved at: {output_path}")
#
# # 示例调用
# images_folder = "/home/dell/fgm/idea/mirror/gaussian-splatting/datasets_blender/real/indoor/images"  # 你的图片文件夹
# masks_folder = "/home/dell/fgm/idea/mirror/gaussian-splatting/datasets_blender/real/indoor/masks"  # 你的 mask 文件夹
# output_folder = "/home/dell/fgm/idea/mirror/gaussian-splatting/datasets_blender/real/indoor/pre_mask"  # 处理后 mask 存放文件夹
#
# for image_name in os.listdir(images_folder):
#     image_path = os.path.join(images_folder, image_name)
#     mask_path = os.path.join(masks_folder, image_name.replace("image", "masks"))
#     if os.path.exists(mask_path):
#         process_mask(image_path, mask_path, output_folder)

import os
from PIL import Image
import numpy as np
from skimage import filters, morphology

def process_mask(image_path, mask_path, output_folder):
    # 读取原始图片和 mask
    image_name = os.path.basename(image_path).split(".")[0]
    mask = Image.open(mask_path).convert('L')

    # 将 mask 图像进行二值化，设定阈值
    threshold = 128  # 设定一个合适的阈值
    binary_mask = np.array(mask) > threshold  # 二值化图像，mask区域为白色

    # 使用 Sobel 算子进行边缘检测，获取mask的边缘
    edge_mask = filters.sobel(binary_mask.astype(float))  # 使用 sobel 算子进行边缘检测

    # 阈值化边缘图像，去除弱边缘
    edge_threshold = np.percentile(edge_mask, 90)  # 设置90%的强度作为阈值
    edge_mask = edge_mask > edge_threshold  # 获取强边缘

    # 应用腐蚀操作去除噪点
    edge_mask = morphology.erosion(edge_mask, morphology.square(3))

    # 保留黑色背景
    black_background = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)

    # 创建一个空白图像，用于存放最终图像
    final_image = np.copy(black_background)

    # 保留原始mask的边界，其余mask边界内部为红色
    eroded_mask = morphology.erosion(binary_mask, morphology.square(50))  # 缩小mask边界
    white_boundary = binary_mask & ~eroded_mask  # 提取白色边界
    red_area = eroded_mask  # 提取内部红色区域

    final_image[white_boundary] = [255, 255, 255]  # 白色边界
    final_image[red_area] = [255, 0, 0]  # 红色区域

    # 转换回 PIL 图像
    final_image = Image.fromarray(final_image.astype(np.uint8))

    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{image_name}.jpg")

    # 保存最终处理后的 mask 为 jpg 格式
    final_image.save(output_path, "JPEG")
    print(f"Processed mask saved at: {output_path}")

# 示例调用
images_folder = "/home/dell/fgm/idea/mirror/gaussian-splatting/datasets_blender/real/indoor/images"  # 你的图片文件夹
masks_folder = "/home/dell/fgm/idea/mirror/gaussian-splatting/datasets_blender/real/indoor/masks"  # 你的 mask 文件夹
output_folder = "/home/dell/fgm/idea/mirror/gaussian-splatting/datasets_blender/real/indoor/pre_mask"  # 处理后 mask 存放文件夹

for image_name in os.listdir(images_folder):
    image_path = os.path.join(images_folder, image_name)
    mask_path = os.path.join(masks_folder, image_name.replace("image", "masks"))
    if os.path.exists(mask_path):
        process_mask(image_path, mask_path, output_folder)