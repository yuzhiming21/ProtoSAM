import pygame
from PIL import Image

import argparse
import numpy as np
import logging


import torch.nn.functional as F
import torch

import os

from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
)


class Button:
    def __init__(self, position, size, color=(255, 255, 255), text='', text_color=(0, 0, 0), font_size=20):
        self.position = position
        self.size = size
        self.color = color
        self.text = text
        self.text_color = text_color
        self.font_size = font_size
        self.font = pygame.font.Font(None, self.font_size)

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, pygame.Rect(self.position, self.size))
        text_surface = self.font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=(self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2))
        surface.blit(text_surface, text_rect)

    def is_clicked(self, mouse_pos):
        return pygame.Rect(self.position, self.size).collidepoint(mouse_pos)



class Argments():
    def __init__(self):
        # self.data = ['lung','lung2','Lung421']
        self.data = 'lung'
        self.snapshot_path = "exps/bibm_128_1click_256_debug/['lung','lung2','Lung421']"
        # self.data_prefix = [f"../datafile/{dataset_name}_crop" for dataset_name in self.data]
        self.data_prefix = f"../datafile/lung_crop"
        # self.data_prefix = f"../datafile/lung"
        print("Datasets", self.data_prefix)
        self.rand_crop_size = (128, 128, 128)
        self.device = "cuda:0"
        self.num_prompts = 10
        self.bs = 1
        self.num_classes = 2
        self.num_worker = 6
        self.checkpoint = 'best'
        self.tolerance = 5


def clip_image_by_gt(img, seg, patch_size=128):
    seg = seg.float()
    prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
    img = img
    l = len(torch.where(prompt == 1)[0])
    #np.random.seed(0)
    #sample = np.random.choice(np.arange(l), 100, replace=True)
    #sample = sample[:3]
    # x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
    # y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
    # z = torch.where(prompt == 1)[2][sample].unsqueeze(1)
    # TODO: 这里sample对无点击情况下精度影响不小，值得研究一下
    x = torch.where(prompt == 1)[1].unsqueeze(1)
    y = torch.where(prompt == 1)[3].unsqueeze(1)
    z = torch.where(prompt == 1)[2].unsqueeze(1)

    x_m = int(torch.div(torch.max(x) + torch.min(x), 2))
    y_m = int(torch.div(torch.max(y) + torch.min(y), 2))
    z_m = int(torch.div(torch.max(z) + torch.min(z), 2))

    d_min = x_m - patch_size//2
    d_max = x_m + patch_size//2
    h_min = z_m - patch_size//2
    h_max = z_m + patch_size//2
    w_min = y_m - patch_size//2
    w_max = y_m + patch_size//2
    d_l = max(0, -d_min)
    d_r = max(0, d_max - prompt.shape[1])
    h_l = max(0, -h_min)
    h_r = max(0, h_max - prompt.shape[2])
    w_l = max(0, -w_min)
    w_r = max(0, w_max - prompt.shape[3])

    d_min = max(0, d_min)
    h_min = max(0, h_min)
    w_min = max(0, w_min)
    
    img_patch = img[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
    img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
    seg = seg.unsqueeze(0)
    seg_patch = seg[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
    seg_patch = F.pad(seg_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
    
    print("auto clip position", d_min, z_m, y_m)
    return img_patch, seg_patch, [d_min, d_max, h_min, h_max, w_min, w_max]


def clip_image(img, seg, clip_position):
    d_min, d_max, h_min, h_max, w_min, w_max = clip_position['min_max']
    pad = clip_position['padding']
    
    img_patch = img[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
    img_patch = F.pad(img_patch, pad)
    seg = seg.unsqueeze(0)
    seg_patch = seg[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
    seg_patch = F.pad(seg_patch, pad)
    return img_patch, seg_patch



def data_init(img_path):

    ct_image = torch.load(img_path).to(torch.float32)#.numpy()
    print("读取CT形状:", ct_image.shape)

    ct_image = preprocess_image(ct_image)
    return save_ct_image(ct_image)

def preprocess_ct_image(ct_image, img_name):
    def normalize(image):
        
        print(image.min(), image.max())
        if "case" in img_name:
            image = image.float()
            image = image - image.mean() - 350
            pos = 50
            width = 400
        else:
            image = image.float()
            image = image - image.mean() - 600#700
            pos = -500 # 3D Slicer
            # width = 1400 # 3D Slicer
            width = 1500
        image = torch.clamp(image, min=pos-width/2, max=pos+width/2)
        _min = image.min()
        _max = image.max()
        print("min and max", _min, _max)
        image = (image - _min) / (_max - _min) * 255
        
        # setting 1
        # image = image + 150
        # image = 255 * torch.pow(image / 255, 4)
        # image = torch.clamp(image, 0, 255)
        # print(image.mean())
        # if "LUNG" in img_name:
        #     shift = 200
        # elif "case" in img_name:
        #     shift = 180
        # else:
        #     shift = 220
        # image = image - image.mean() + shift
        # image = 255 * torch.pow(image / 255, 5)
        # image = torch.clamp(image, 0, 255)
        return image.int()
    
    ct_image = normalize(ct_image)

    return ct_image


def preprocess_image(ct_image):
    def normalize(image):
        _min = image.min()
        _max = image.max()
        image = (image - _min) / (_max - _min) * 255
        return image
    
    ct_image = normalize(ct_image)

    return ct_image


def save_ct_image(ct_image, with_click=True):
    # 定义点击位置列表
    click_poses = []
    for i in range(ct_image.shape[0]):
        # 通过 PIL 创建图像对象
        
        image_data_uint8 = ct_image[i].cpu().numpy().astype(np.uint8)
        image = Image.fromarray(image_data_uint8)

        # 保存图像为 PNG 格式，文件名为 "image_{}.png"，其中 {} 会被循环的索引替换
        filename = "{:03d}.png".format(i+1)
        image.save(os.path.join("ui_files", filename))

        click_poses.append([])
    if with_click:
        return ct_image, click_poses
    else:
        return ct_image



def load_CT_image(folder = 'ui_files'):
    image_data = []
    image_paths = sorted(os.listdir(folder))
    for image_path in image_paths:
        image_data.append(pygame.image.load(os.path.join(folder, image_path)))
    return image_data


def model_predict(img, prompt, patch_size = 128):
    global img_encoder
    global prompt_encoder_list
    global mask_decoder    

    # TODO: 检查一下图片的大小，按理说这里应该是128， 需要根据prompt来裁剪
    # img = crop_by_click(img, prompt)
    assert img.shape[-1] == patch_size
    print("feed to network: img shape", img.shape)
    
    out = F.interpolate(img.float(), scale_factor=256 / patch_size, mode='trilinear')  # change to 256 input
    input_batch = out[0].transpose(0, 1)

    # TODO: 这一步可以提前做好，与点击无关
    batch_features, feature_list = img_encoder(input_batch)


    feature_list.append(batch_features)
    points_torch = prompt.transpose(0, 1)
    new_feature = []
    # 只有第四个stage会加上点击操作
    # TODO: 训练时加上不需要点击的情况；或者训练一个点击预测网络
    for i, (feature, feature_decoder) in enumerate(zip(feature_list, prompt_encoder_list)):
        if i == 3:
            new_feature.append(
                feature_decoder(feature.to(device), points_torch.clone(), [patch_size, patch_size, patch_size])
            )
        else:
            new_feature.append(feature.to(device))
    img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(device), scale_factor=64/patch_size,
                                mode="trilinear")
    new_feature.append(img_resize)
    masks = mask_decoder(new_feature, 4, patch_size//64)  # for 256
    masks = masks.permute(0, 1, 4, 2, 3)

    
    return masks


def predict_by_click(img, prompt):

    original_shape = img.shape
    print("input image shape is:", original_shape)

    global device

    intensity_range = (0, 1749)
    global_mean = 984.0138
    global_std = 222.9426
    spatial_index = [0, 1, 2]
    img_spacing = (1.245471, 0.828125, 0.828125)
    target_spacing = (1, 1, 1)
    patch_size = 128

    

    transforms = [
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=intensity_range[0],
            b_max=intensity_range[1],
            clip=True,
        ),
    ]
    transforms.extend(
        [
            NormalizeIntensityd(
                keys=["image"],
                subtrahend=global_mean,
                divisor=global_std,
            )
        ]
    )
    transforms = Compose(transforms)
    
    img[np.isnan(img).bool()] = 0

    img = (
        F.interpolate(
            input=img[None, None, :, :, :].clone(),
            scale_factor=tuple(
                [img_spacing[i] / target_spacing[i] for i in range(3)]
            ),
            mode="trilinear",
        )
        .squeeze(0)
        .numpy()
    )

    trans_dict = transforms({"image": img})
    img = trans_dict["image"]

    print("transformed image shape is:", img.shape)

    img = img.repeat(3, 1, 1, 1)

    img = img.to(device)

    # 这里resize是因为img进行了缩放，后面prompt在裁剪时需要同步，否则对不齐
    prompt = F.interpolate(prompt[None, :, :, :, :], img.shape[1:], mode="nearest")[0]

    # get click poses
    x = torch.where(prompt == 1)[1].unsqueeze(1)
    y = torch.where(prompt == 1)[3].unsqueeze(1)
    z = torch.where(prompt == 1)[2].unsqueeze(1)

    # x_m = (torch.max(x) + torch.min(x)) // 2
    # y_m = (torch.max(y) + torch.min(y)) // 2
    # z_m = (torch.max(z) + torch.min(z)) // 2
    x_m = torch.div(torch.max(x) + torch.min(x), 2, rounding_mode='trunc')
    y_m = torch.div(torch.max(y) + torch.min(y), 2, rounding_mode='trunc')
    z_m = torch.div(torch.max(z) + torch.min(z), 2, rounding_mode='trunc')

    d_min = x_m - patch_size//2
    d_max = x_m + patch_size//2
    h_min = z_m - patch_size//2
    h_max = z_m + patch_size//2
    w_min = y_m - patch_size//2
    w_max = y_m + patch_size//2
    d_l = max(0, -d_min)
    d_r = max(0, d_max - img.shape[1])  # TODO: note this change, origin: prompt.shape[1]
    h_l = max(0, -h_min)
    h_r = max(0, h_max - img.shape[2])  # TODO: note this change, origin: prompt.shape[1]
    w_l = max(0, -w_min)
    w_r = max(0, w_max - img.shape[3])  # TODO: note this change, origin: prompt.shape[1]

    points = torch.cat([x-d_min, y-w_min, z-h_min], dim=1).unsqueeze(1).float()
    points_torch = points.to(device)
    d_min = max(0, d_min)
    h_min = max(0, h_min)
    w_min = max(0, w_min)

    if img.shape[0] != 1:
        img = img.unsqueeze(0)

    print("before clip, image shape is:", img.shape)
    print("clip range:", d_min, d_max, h_min, h_max, w_min, w_max)
    img_patch = img[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
    print("after clip, image shape is:", img_patch.shape)
    img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
    print("after padding, image shape is:", img_patch.shape)
    pred = model_predict(img_patch, points_torch)


    pred = pred[:,:, d_l:patch_size-d_r, h_l:patch_size-h_r, w_l:patch_size-w_r]
    pred = F.softmax(pred, dim=1)[:,1]
    pred_mask_debug = pred > 0.5
    print("预测mask形状为", pred_mask_debug.shape, '目标体积为:', pred_mask_debug.sum())

    seg_pred = torch.zeros_like(prompt).to(device)
    seg_pred[:, d_min:d_max, h_min:h_max, w_min:w_max] += pred

    final_pred = F.interpolate(seg_pred.unsqueeze(1), size = original_shape,  mode="trilinear")
    masks = final_pred > 0.5
    print("修改形状后，mask形状为", masks.shape, '目标体积为:', masks.sum())

    return img_patch


# snapshot_path = "_bibm_lung2_128/lung2/best.pth.tar"
# rand_crop_size = (128, 128, 128)
# device = "cuda:0"

# img_encoder = ImageEncoderViT_3d(
#     depth=12,
#     embed_dim=768,
#     img_size=1024,
#     mlp_ratio=4,
#     norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#     num_heads=12,
#     patch_size=16,
#     qkv_bias=True,
#     use_rel_pos=True,
#     global_attn_indexes=[2, 5, 8, 11],
#     window_size=14,
#     cubic_window_size=8,
#     out_chans=256,
#     num_slice = 16)

# print("读取网络权重：", os.path.join(snapshot_path))
# networks_weight_dict = torch.load(snapshot_path, map_location='cpu')

# img_encoder.load_state_dict(networks_weight_dict["encoder_dict"], strict=True)
# img_encoder.to(device)

# prompt_encoder_list = []
# for i in range(4):
#     prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
#             embedding_dim=256,
#             mlp_dim=2048,
#             num_heads=8))
#     prompt_encoder.load_state_dict(
#         networks_weight_dict["feature_dict"][i], strict=True)
#     prompt_encoder.to(device)
#     prompt_encoder_list.append(prompt_encoder)

# mask_decoder = VIT_MLAHead(img_size = 96).to(device)
# mask_decoder.load_state_dict(networks_weight_dict["decoder_dict"],
#                         strict=True)
# mask_decoder.to(device)

# img_encoder.eval()
# for i in range(len(prompt_encoder_list)):
#     prompt_encoder_list[i].eval()
# mask_decoder.eval()




