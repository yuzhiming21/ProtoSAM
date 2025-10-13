import pickle
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    #AddChanneld,
    RandCropByPosNegLabeld,
    CropForegroundd,
    Resized,
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
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F

# 该类继承自 MapTransform，表示它可以对数据字典中的特定键应用转换操作。它的主要功能是将指定键对应的数据张量按给定的阈值进行二值化（大于阈值的变为1，其他变为0）。
class BinarizeLabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,       # 待进行二值化操作的键的集合
            threshold: float = 0.5,     # 二值化的阈值
            allow_missing_keys: bool = False,   #如果为 True，当输入数据字典中缺少某个键时，转换操作会忽略这个键。如果为 False，缺少键时会抛出异常。
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        # data: 这个参数是一个字典，包含了多个张量（例如，图像数据、标签数据等）。其中的每个键对应一个数据项，每个数据项都是一个 torch.Tensor 张量。
        # key_iterator(d): 该方法遍历了所有需要进行转换的键（由 keys 参数指定）。它是 MapTransform 类的一部分，通常用于遍历和操作数据字典中的指定键。
        # torch.as_tensor(d[key]): 这行代码确保如果某个键对应的值不是 torch.Tensor 类型，则将其转换为 Tensor。这样可以确保数据的一致性。
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)    #二值化操作
        return d


class BaseVolumeDataset(Dataset):
    def __init__(
            self,
            image_paths,            # 图像数据的路径列表
            label_meta,             # 标签数据的路径列表
            augmentation,           # 图像增强的操作，可能是旋转、翻转、缩放等
            split="train",          # 数据集划分
            rand_crop_spatial_size=(96, 96, 96),    # 随机裁剪的空间尺寸，通常用于数据增强，将体积图像裁剪为指定大小的区域
            #rand_crop_spatial_size=(128, 128, 128),
            convert_to_sam=True,    # 是否将图像和标签数据转换为特定格式
            do_test_crop=True,      # 在测试时是否对图像进行裁剪
            do_val_crop=True,       # 在验证时是否对图像进行裁剪
            do_nnunet_intensity_aug=True,           #是否进行 intensity augmentation（对图像强度进行增强）
            seed=2024               # 随机种子，用于控制数据增强的随机性，确保可重现性
    ):
        super().__init__()
        # 将.nii.gz 格式的图像路径替换为 .pt 格式的路径
        self.img_dict = [p.replace(".nii.gz", ".pt") for p in image_paths]
        self.label_dict = [p.replace(".nii.gz", ".pt") for p in label_meta]
        # 参数设置，初始化
        self.aug = augmentation
        self.split = split
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.convert_to_sam = convert_to_sam
        self.do_test_crop = do_test_crop
        self.do_nnunet_intensity_aug = do_nnunet_intensity_aug
        self.do_val_crop = do_val_crop
        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None
        # 数据集统计设置
        self._set_dataset_stat()
        # 数据集增强变换设置
        self.transforms = self.get_transforms()
        # 使用不同的随机种子，确保不同的拆分使用不同的随机数序列，从而避免数据泄漏和增强不一致性
        if self.split == 'val':
            self.transforms.set_random_state(seed=seed)
        elif self.split == 'test':
            self.transforms.set_random_state(seed=seed*2)
        # 多模态标记符号
        self.modal_idx = 0
        self.max_retry = 20  # 新增：最大重试次数，防止无限递归

    def _set_dataset_stat(self):
        pass

    def __len__(self):
        return len(self.img_dict)

    def __getitem__(self, idx):
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                # 获取图像和标签的路径
                img_path = self.img_dict[idx]
                label_path = self.label_dict[idx]
                # 加载图像和标签
                img = np.float32(torch.load(img_path, weights_only=False))
                seg = np.float32(torch.load(label_path, weights_only=False))

                if len(img.shape) != 3:
                    img = img.squeeze()  # 从 (1, H, W, D) -> (H, W, D)
                    #img = img[:, :, :, self.modal_idx]
                    seg = seg.squeeze()
                elif len(img.shape) == 3:
                    pass  # 已经是三维，无需处理
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
                # 空间变换
                # TODO: this could be useful
                #print("Image shape before transpose:", img.shape)
                #print("Current spatial_index:", self.spatial_index)
                img = img.transpose(self.spatial_index)
                # img.transpose(self.spatial_index) 将图像数据的维度按照 self.spatial_index 指定的顺序重新排列。
                # self.spatial_index 是一个表示空间维度顺序的索引列表，通常在医学图像处理中，图像的维度顺序（如 z, y, x）可能需要转换。
                seg = seg.transpose(self.spatial_index[:3])
                # 对于 seg（标签），我们只使用 self.spatial_index[:3]，因为标签通常只有 3 个空间维度（x, y, z），没有通道维度。
                # 图像间隔设置
                img_spacing = (1, 1, 1)
                # print("spacing target  img", self.target_spacing, img_spacing)
                # DHW shape
                # 处理NaN值，一般替换为0
                img[np.isnan(img)] = 0
                seg[np.isnan(seg)] = 0

                #scale_factor=(
                #    img_spacing[0]/self.target_spacing[0],
                #    img_spacing[1]/self.target_spacing[1],
                #    img_spacing[2]/self.target_spacing[2]
                #)

                # 标签转换为二值图像。所有等于 self.target_class 的像素值会被设置为 1，其他的设置为 0。这种操作常见于语义分割任务，用于提取目标类的区域。
                seg = (seg == self.target_class).astype(np.float32)
                # 判断图像和标签的空间间隔差异
                if (np.max(img_spacing) / np.min(img_spacing) > 8) or (
                    np.max(self.target_spacing / np.min(self.target_spacing) > 8)
                ):  
                    raise "check this"
                    # resize 2D
                    # 2D图像插值
                    img_tensor = F.interpolate(     # 使用 F.interpolate 进行插值操作
                        input=torch.tensor(img[:, None, :, :]),     # img[:, None, :, :] 将图像扩展到一个 4D 张量，其中 None 增加一个维度以适应插值操作。
                        scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),     # scale_factor 是根据图像空间分辨率和目标空间分辨率之间的比例进行计算
                        mode="bilinear",    # 双线性插值方法，适用于2D图像
                    )
                    # 2D标签插值
                    if self.split != "test":
                        seg_tensor = F.interpolate(
                            input=torch.tensor(seg[:, None, :, :]),
                            scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                            mode="bilinear",
                        )
                    # 3D图像插值
                    img = (
                        F.interpolate(
                            input=img_tensor.unsqueeze(0).permute(0, 1, 2, 3, 4).contiguous(),  # 对 img_tensor 增加了一个额外的维度（unsqueeze(0)），然后通过 permute(0, 2, 1, 3, 4) 调整维度顺序
                            scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),   # scale_factor 根据图像的第一个维度（通常是 z）和目标空间的分辨率进行计算
                            mode="trilinear",   # 三线性插值
                        )
                        .squeeze()
                        #.numpy()
                    )
                    # 3D标签插值
                    if self.split != "test":
                        seg = (
                            F.interpolate(
                                input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                                scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                                mode="trilinear",
                            )
                            .squeeze()
                            #.numpy()
                        )
                else:   # 处理图像和标签的空间重采样
                    img = (
                        F.interpolate(
                            input=torch.tensor(img[None, None, :, :, :]),
                            scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),# scale_factor 是在三个维度上进行计算的，保证图像和标签的空间分辨率都与目标分辨率一致
                            mode="trilinear",
                        )
                    .squeeze(0)
                    #.numpy()
                    )
                    #if self.split != "test":  # TODO: why?
                    seg = (
                        F.interpolate(
                            input=torch.tensor(seg[None, None, :, :, :]),
                            scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(3)]),
                            mode="trilinear",
                        )
                    .squeeze(0)
                    #.numpy()
                    )
        
                # 训练集是否进行数据增强；验证集和测试集是否进行裁剪
                if (self.aug and self.split == "train") or ((self.do_val_crop  and (self.split=='val' or self.split=='test'))):
                    trans_dict = self.transforms({"image": img, "label": seg})[0]
                    img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
                else:
                    trans_dict = self.transforms({"image": img, "label": seg})
                    img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
            
                # 检查变换后的图像尺寸是否有效
                if trans_dict["image"].shape[1] == 0 or trans_dict["image"].shape[2] == 0 or trans_dict["image"].shape[3] == 0:
                    raise ValueError(f"Invalid image size after transforms: {trans_dict['image'].shape}")

                #seg_aug = seg_aug.squeeze()             # 标签压缩（去掉维度为 1 的轴）
                #img_aug = img_aug.repeat(3, 1, 1, 1)    # 图像扩展。扩展为3通道
                seg_aug = trans_dict["label"].squeeze()
                img_aug = trans_dict["image"].repeat(3, 1, 1, 1)

                # 返回有效数据
                if self.split == "train":
                    return img_aug, seg_aug, np.array(img_spacing)
                else:
                    return img_aug, seg_aug, np.array(img_spacing), img_path
            except Exception as e:
                # 打印错误信息
                print(f"Error processing sample {idx} (retry {retry_count + 1}/{self.max_retry}): {str(e)}")
                # 选择下一个样本（循环索引）
                idx = (idx + 1) % len(self)
                retry_count += 1
        # 如果超过最大重试次数仍失败，抛出异常
        raise RuntimeError(f"Failed to load valid sample after {self.max_retry} retries.")
    
    # 数据变换操作
    def get_transforms(self):
        transforms = [
            ScaleIntensityRanged(   # 将图像的强度（像素值）进行重新缩放，以确保图像的强度值落在一个指定的范围内
                keys=["image"],                     # 指定应用此转换的字段
                a_min=self.intensity_range[0],      # 输入图像强度值的范围
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[0],      # 转换后的输出强度值范围
                b_max=self.intensity_range[1],
                clip=True,                          # 当 clip=True 时，如果图像中的强度值超出了指定的输出范围 [b_min, b_max]，则会将它们裁剪到该范围内
            ),
        ]

        if self.split == "train":
            transforms.extend(
                [
                    # 对图像的强度进行随机偏移。它会随机地将图像的像素值增加或减少一个偏移量，通常用于增强数据的多样性，模拟不同的拍摄条件、设备变化等。
                    RandShiftIntensityd(    
                        keys=["image"],
                        #offsets 偏移量是根据图像的强度范围 self.intensity_range 来计算的。0.025 是一个比例因子，意味着偏移量为强度范围的 2.5%
                        offsets=(self.intensity_range[1]-self.intensity_range[0])*0.025,  # TODO: check data aug
                        prob=0.5,   # 该变换的应用概率为 50%
                    ),
                    # 裁剪图像中包含前景的部分，去除背景。它根据一个强度阈值来选择前景区域，通常用于医学图像，确保只有包含感兴趣区域（例如肿瘤或病变）的部分被保留下来。
                    CropForegroundd(    
                        keys=["image", "label"],    # 同时应用于图像和标签，避免图像和标签不同步
                        source_key="image",     # 指定图像数据作为裁剪参考。裁剪的区域将基于图像的强度来选择
                        select_fn=lambda x: x > self.intensity_range[0],    # 选择强度大于 self.intensity_range[0] 的像素作为前景
                        allow_smaller=True,  # 禁止裁剪后尺寸过小
                    ),
                    Resized(  # 强制调整尺寸
                        keys=["image", "label"],
                        spatial_size=(64, 64, 64),
                        mode=("trilinear", "nearest"),
                        ),
                    # 对图像的强度进行标准化，使图像的像素值具有零均值和单位标准差。一种常见的图像预处理方法，有助于提高模型训练的效果和稳定性。
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,    # 将每个图像的像素值减去一个全局均值 self.global_mean，使得图像的强度分布以该均值为中心。
                        divisor=self.global_std,        # 将结果除以一个全局标准差 self.global_std，使得图像的强度分布的标准差为 1。
                    ),
                ]
            )

            if self.do_dummy_2D:
                transforms.extend(
                    [
                        # 一个随机旋转变换，用于对图像和标签进行旋转操作，常用于增强数据的多样性，尤其是当模型需要具有旋转不变性的能力时。
                        RandRotated(
                            keys=["image", "label"],
                            prob=0.3,                   # 旋转的概率为30%
                            range_x=30 / 180 * np.pi,   # 旋转的角度范围；此处的 30 / 180 * np.pi 表示一个角度范围从 -30° 到 30°，然后转换为弧度。
                            keep_size=False,            # 设置 False 意味着旋转后图像的大小可能会发生变化。如果设置为 True，则会保证旋转后的图像大小与原始图像一致。
                                ),
                        # 对图像和标签进行随机缩放
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=[1, 0.95, 0.95],     # 缩放比例范围；此处表示在第一个维度上不变，后两个维度进行最小0.9的缩放
                            max_zoom=[1, 1.1, 1.1],
                            mode=["trilinear", "trilinear"],    # 插值方法
                        ),
                    ]
                )
            else:
                transforms.extend(
                    [
                        # 3D图像随机旋转
                        # RandRotated(
                        #     keys=["image", "label"],
                        #     prob=0.3,
                        #     range_x=30 / 180 * np.pi,
                        #     range_y=30 / 180 * np.pi,
                        #     range_z=30 / 180 * np.pi,
                        #     keep_size=False,
                        # ),
                        # 缩放
                        RandZoomd(
                            keys=["image", "label"],
                            prob=0.8,
                            min_zoom=0.85,      # 缩放范围固定，不依赖于维度
                            max_zoom=1.25,
                            mode=["trilinear", "trilinear"],
                        ),
                    ]
                )

            transforms.extend(
                [
                    # 将标签二值化
                    BinarizeLabeld(keys=["label"]),
                    # 对图像和标签进行空间上的填充（padding）
                    SpatialPadd(
                        keys=["image", "label"],
                        # 填充的目标尺寸由 self.rand_crop_spatial_size 控制，并且每个维度会按 1.2 倍的比例进行调整（即增加 20% 的尺寸）
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),
                    # 执行基于标签的随机裁剪，目的是通过选择正样本和负样本的区域进行裁剪。这种方法通常用于处理类别不平衡的问题。
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,      # 表示裁剪时要保留至少有两个正样本（标签值为 2 的区域）
                        #neg=1,
                        #neg=2,     # NOTE: 防止假阳太多
                        neg=0,      # 表示不允许裁剪负样本（标签值为 0 的区域），以防止裁剪到无关区域
                        num_samples=1,  # 指定裁剪样本的数量为 1
                    ),
                    # 对图像和标签进行随机空间裁剪
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,   # 表示裁剪区域的尺寸
                        random_size=False,  # 表示裁剪的区域大小是固定的，不会随机变化
                    ),
                    # 对图像和标签进行随机翻转操作
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),   # 深度维度
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),   # 高度维度
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),   # 宽度维度
                    # 对图像和标签进行随机的 90 度旋转，max_k=3 表示最多可以旋转三次
                    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
                    # 随机调整图像的强度
                    # RandScaleIntensityd(keys="image", factors=0.1, prob=0.2),
                    # 随机在给定范围内平移图像的强度
                    # RandShiftIntensityd(
                    #     keys=["image"],
                    #     offsets=0.10,
                    #     prob=0.2,
                    # ),
                    # 向图像添加随机高斯噪声
                    # RandGaussianNoised(keys=["image"], prob=0.1),
                    # 向图像添加高斯平滑噪声
                    # RandGaussianSmoothd(
                    #     keys=["image"],
                    #     prob=0.2,
                    #     sigma_x=(0.5, 1),
                    #     sigma_y=(0.5, 1),
                    #     sigma_z=(0.5, 1),
                    # ),
                    # 向图像和标签添加额外的通道
                    # AddChanneld(keys=["image", "label"]),
                    # RandShiftIntensityd(keys=["image"], offsets=10),
                    # RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),]
                ]
            )
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    # 裁剪出图像和标签中前景区域
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        allow_smaller=True,
                    ),
                    # 标签（label）二值化
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.split == "val" or self.split == "test" ):
            transforms.extend(
                [
                    # CropForegroundd(
                    #     keys=["image", "label"],
                    #     source_key="image",
                    #     select_fn=lambda x: x > self.intensity_range[0],
                    # ),
                    # 填充
                    SpatialPadd(
                        keys=["image", "label"],
                        #spatial_size=[i for i in self.rand_crop_spatial_size],
                        spatial_size=self.rand_crop_spatial_size,
                    ),
                    # 裁剪
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    # 标准化
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            raise "s"
            transforms.extend(
                [
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms