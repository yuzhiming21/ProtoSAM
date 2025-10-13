from scipy import ndimage
import numpy as np
from medpy import metric
import logging
import os
import time
import torch
import matplotlib.pyplot as plt

def count_parameters(img_encoder):
    return sum(p.numel() for p in img_encoder.parameters() if p.requires_grad) / 1e6

def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    """set up logger"""
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, "{}_{}.log".format(logger_name, get_timestamp()))
        fh = logging.FileHandler(log_file, mode="w")
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def get_timestamp():
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%Y%m%d")
    return timestampDate + "-" + timestampTime


def adjust_window(image, window_width = 1400, window_level = 400):
    # 计算窗口的上下界
    window_min = window_level - window_width / 2
    window_max = window_level + window_width / 2
    
    # 将图像灰度值限制在窗口范围内
    adjusted_image = np.clip(image, window_min, window_max)
    
    # 将灰度值重新映射到0-1范围
    adjusted_image = (adjusted_image - window_min) / (window_max - window_min)
    
    return adjusted_image

def visualize_CT_mask(ct, mask, predict, folder_name, name):

    ct = adjust_window(ct)
    ct = ct * 255
    
    # 创建一个新的图形，并设置子图布局为 1 行 2 列
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # 可视化矩阵 A 在第一个子图中
    axs[0].imshow(ct, cmap='gray')

    # 叠加矩阵 B 在矩阵 A 上面，并设置透明度为 0.5
    axs[1].imshow(ct, cmap='gray')

    axs[1].imshow(mask, alpha=0.5)
    
    # 叠加矩阵 B 在矩阵 A 上面，并设置透明度为 0.5
    axs[2].imshow(ct, cmap='gray')
    axs[2].imshow(predict, alpha=0.5)
    
    # 设置子图标题
    axs[0].set_title(' ')
    axs[1].set_title(' ')
    plt.title(' ')
    
    # 调整子图间距
    plt.tight_layout()

    plt.savefig(f"{folder_name}/{name}.png", dpi=120)
    # 显示图形
    plt.clf()
    #plt.show()