import pickle
import os, sys
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from .base_dataset import BaseVolumeDataset
from .base_dataset_nii import BaseVolumeDataset as BaseVolumeDataset_nii    # type: ignore
from torch.utils.data import ConcatDataset

# use all pixel to calculate mean and std
dataset_stats = {'lung': {'mean': -322.5155297041428, 'std': 469.8601382457649, 'min': -1024.0, 'max': 19315.0}, 'lung2': {'mean': 674.6135723435852, 'std': 528.6858683312973, 'min': -10748.0, 'max': 20339.0}, 'Lung421': {'mean': 388.9518236165722, 'std': 465.7339401348388, 'min': -1024.0, 'max': 4095.0}, 'pancreas': {'mean': -125.650033914228, 'std': 359.4353566304122, 'min': -2048.0, 'max': 4009.0}, 'kits23': {'mean': -139.8815115572298, 'std': 362.21033950802564, 'min': -6986.0, 'max': 18326.0}, 'hepatic': {'mean': -97.51201450512347, 'std': 340.1144827002664, 'min': -1024.0, 'max': 3072.0}}

# use foreground pixel to calculate mean and std
dataset_stats = {'lung': {'mean': -158.40746487386428, 'std': 324.82639474513553, 'min': -1393.5, 'max': 3040.5}, 'lung2': {'mean': 843.4502890881769, 'std': 332.2021557825976, 'min': -369.5, 'max': 4064.5}, 'Lung421': {'mean': 681.1772523003802, 'std': 502.2254769460235, 'min': -1509.4, 'max': 4315.4}, 'pancreas': {'mean': 79.326213811815, 'std': 78.33553180106968, 'min': -1404.9, 'max': 3477.9}, 'kits23': {'mean': 114.43007901992068, 'std': 75.53780947379572, 'min': -1432.4, 'max': 3480.4}, 'hepatic': {'mean': 163.68194368793922, 'std': 39.023300794136574, 'min': -983.7, 'max': 3440.7}}

# change upper bound and lower bound
dataset_stats = {'lung': {'mean': -158.40746487386428, 'std': 324.82639474513553, 'min': -1024.0, 'max': 3040.5}, 'lung2': {'mean': 843.4502890881769, 'std': 332.2021557825976, 'min': -369.5, 'max': 4064.5}, 'Lung421': {'mean': 681.1772523003802, 'std': 502.2254769460235, 'min': -1024.0, 'max': 4095.0}, 'pancreas': {'mean': 79.326213811815, 'std': 78.33553180106968, 'min': -1404.9, 'max': 3477.9}, 'kits23': {'mean': 114.43007901992068, 'std': 75.53780947379572, 'min': -1432.4, 'max': 3480.4}, 'hepatic': {'mean': 163.68194368793922, 'std': 39.023300794136574, 'min': -983.7, 'max': 3072.0}}

# use foreground pixel to calculate min and max
# use min and max to clip image, then use all pixel to calculate mean and std
dataset_stats = {'lung': {'mean': -322.51959015195416, 'std': 469.82400813647934, 'min': -1024.0, 'max': 3040.5}, 'lung2': {'mean': 694.7543464205737, 'std': 480.00401269466624, 'min': -369.5, 'max': 4064.5}, 'Lung421': {'mean': 388.9518236165722, 'std': 465.7339401348388, 'min': -1024.0, 'max': 4095.0}, 'pancreas': {'mean': -125.64860468617131, 'std': 359.4285336838109, 'min': -1404.9, 'max': 3477.9}, 'kits23': {'mean': -136.80715503557894, 'std': 352.0108393545522, 'min': -1432.4, 'max': 3480.4}, 'hepatic': {'mean': -97.00934024739867, 'std': 338.96319518567566, 'min': -983.7, 'max': 3072.0}}

# change target class
dataset_stats = {'lung': {'mean': -158.40746487386428, 'std': 324.82639474513553, 'min': -1393.5, 'max': 3040.5}, 'lung2': {'mean': 843.4502890881769, 'std': 332.2021557825976, 'min': -369.5, 'max': 4064.5}, 'Lung421': {'mean': 681.1772523003802, 'std': 502.2254769460235, 'min': -1509.4, 'max': 4315.4}, 'pancreas': {'mean': 71.47238368378886, 'std': 57.31853454016189, 'min': -879.1, 'max': 3430.1}, 'kits23': {'mean': 60.97325306066615, 'std': 56.02929501689018, 'min': -1431.3, 'max': 3480.3}, 'hepatic': {'mean': 83.64586158520629, 'std': 40.57069472710642, 'min': -1105.7, 'max': 3450.7}}

# .nii格式数据
class LiTSVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)       # 图像数据的强度范围
        self.target_spacing = (1, 1, 1)         # 目标重采样间距（单位：毫米）
        self.global_mean = 60.057533            # 数据标准化
        self.global_std = 40.198017              
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW 图像空间轴的顺序，一般为深度，高度，宽度
        self.do_dummy_2D = False                # 是否进行2D虚拟处理
        self.target_class = 2                   # 目标类别数量
# 3D脑部图像数据
class BrainVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: w, h, d
        self.intensity_range = (-2252.0, 11260.0)
        self.target_spacing = (1, 1, 1)
        #self.intensity_range = (-1442.0, 7210.0)
        
        # all
        self.global_mean = 86.57460801234248
        self.global_std = 215.53386770762629
        # 0
        # self.global_mean = 73.70442524169478
        # self.global_std = 179.69913807142927
        self.spatial_index = [2, 1, 0, 3]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 1
# 海马区
# class HippoVolumeDataset(BaseVolumeDataset_nii):
#     def _set_dataset_stat(self):
#         # org load shape: d, h, w
#         self.intensity_range = (0, 486420.2188)
#         self.target_spacing = (0.2, 0.2, 0.2)
#         self.global_mean = 21508.0996
#         self.global_std = 60030.5781
#         self.spatial_index = [2, 1, 0]  # index used to convert to DHW
#         self.do_dummy_2D = False
#         self.target_class = 2

# 胰腺数据
class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1404.9, 3477.9)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -125.64860468617131
        self.global_std = 359.4285336838109
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2
# 肝脏数据
class LiverVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1298.5, 4606.5)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -231.4665
        self.global_std = 434.2647
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 结肠数据
class ColonVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1009.625, 606.625)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -170.58464
        self.global_std = 389.29116
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 海马体数据
class HippoVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-598.950, 5390.99)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 293.91948
        self.global_std = 128.70902
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 脾脏数据
class SpleenVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-659.1875, 1092.6875)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -226.8739
        self.global_std = 327.7521
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 结肠体积数据
# class ColonVolumeDataset(BaseVolumeDataset_nii):
#     def _set_dataset_stat(self):
#         # org load shape: d, h, w
#         self.intensity_range = (-57, 175)
#         self.target_spacing = (1, 1, 1)
#         self.global_mean = 65.175035
#         self.global_std = 32.651197
#         self.spatial_index = [2, 1, 0]  # index used to convert to DHW
#         self.do_dummy_2D = True
#         self.target_class = 1

# 肺部CT数据
# NOTE: 这里如果是128的话，需要是BaseVolumeDataset
class LungVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.0, 3040.5)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -322.51959015195416
        self.global_std = 469.82400813647934
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 肺部数据
class Lung2VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-369.5, 4064.5)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 694.7543464205737
        self.global_std = 480.00401269466624
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 肺部病患数据  
class LungPatVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-87.25, 1264.25)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 407.629
        self.global_std = 452.113
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 肺部病患数据 .nii
class LungTestVolumeDataset(BaseVolumeDataset_nii):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-939.75, 576.75)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -167.6621
        self.global_std = 367.4284
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 肺部病患数据 3D
class Lung3VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.0, 4095.0)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 388.9518236165722
        self.global_std = 465.7339401348388
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 肺部CT扫描数据 
class LungHosVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-27, 520)
        #self.intensity_range = (-527, 1020)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 216.8536
        self.global_std = 75.9415
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
# 胸部CT数据
class Kits23VolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1432.4, 3480.4)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -136.80715503557894
        self.global_std = 352.0108393545522
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2
# 肝脏CT扫描数据
class HepaticVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-983.7, 3072.0)
        self.target_spacing = (1, 1, 1)
        self.global_mean = -97.51201450512347 # -97.00934024739867
        self.global_std = 340.1144827002664 # 338.96319518567566
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        # self.target_class = 1
        self.target_class = 2

class TotalSegmentatorVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (0.0, 4095.0)
        self.target_spacing = (1.5, 1.5, 1.5)
        self.global_mean = 3301.7114
        self.global_std = 1591.4854
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class AbdomenC1K1Dataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.00, 2757.20)
        self.target_spacing = (0.74, 0.74, 2.5)
        self.global_mean = -552.60887
        self.global_std = 527.2337557823096
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1

class AbdomenC1K2Dataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.00, 2757.20)
        self.target_spacing = (0.74, 0.74, 2.5)
        self.global_mean = -552.60887
        self.global_std = 527.2337557823096
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2

class AbdomenC1K3Dataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.00, 2757.20)
        self.target_spacing = (0.74, 0.74, 2.5)
        self.global_mean = -552.60887
        self.global_std = 527.2337557823096
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 3

class AbdomenC1K4Dataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-1024.00, 2757.20)
        self.target_spacing = (0.74, 0.74, 2.5)
        self.global_mean = -552.60887
        self.global_std = 527.2337557823096
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 4

DATASET_DICT = {
    "lits": LiTSVolumeDataset,
    "pancreas": PancreasVolumeDataset,
    "colon": ColonVolumeDataset,
    "lung": LungVolumeDataset,
    "lung2": Lung2VolumeDataset,
    'lung3': Lung3VolumeDataset,
    'Lung421': Lung3VolumeDataset,
    "lung_center": LungVolumeDataset,
    "lung2_center": Lung2VolumeDataset,
    'Lung421_center': Lung3VolumeDataset,
    'brain_torch': BrainVolumeDataset,
    'hippo': HippoVolumeDataset,
    "kits23": Kits23VolumeDataset,
    'hepatic': HepaticVolumeDataset,
    'lung_test': LungTestVolumeDataset,
    'lung_hospital': LungHosVolumeDataset,
    'lung_hospital_multi': LungHosVolumeDataset,
    'lung_hospital_sac_fake': LungHosVolumeDataset,
    'spleen': SpleenVolumeDataset,
    'liver':LiverVolumeDataset,
    'lung_pat':LungPatVolumeDataset,
    'total_spleen':TotalSegmentatorVolumeDataset, 
    'total_pancreas':TotalSegmentatorVolumeDataset,
    'total_kidney_left':TotalSegmentatorVolumeDataset,
    'total_kidney_right':TotalSegmentatorVolumeDataset,
    'total_lung_lower_lobe_left':TotalSegmentatorVolumeDataset,
    'total_lung_lower_lobe_right':TotalSegmentatorVolumeDataset,
    'total_lung_middle_lobe_right':TotalSegmentatorVolumeDataset,
    'total_lung_upper_lobe_left':TotalSegmentatorVolumeDataset,
    'total_lung_upper_lobe_right':TotalSegmentatorVolumeDataset,
    'AbdomenCT_1K1':AbdomenC1K1Dataset,
    'AbdomenCT_1K2':AbdomenC1K2Dataset,
    'AbdomenCT_1K3':AbdomenC1K3Dataset,
    'AbdomenCT_1K4':AbdomenC1K4Dataset
}

# 加载图像数据函数
def load_data_volume(
    *,
    datas,                  # 数据集列表
    path_prefixes,          # 数据集对应的路径前缀列表
    batch_size,             # 每个批次加载的样本数量
    data_dir=None,          # 数据存放的目录路径
    split="train",          # 训练集划分
    deterministic=False,    # 是否要求数据加载是确定性的（即不打乱顺序，常用于验证或测试阶段，确保可复现性）
    augmentation=False,     # 是否在加载数据时应用数据增强
    fold=0,                 # 交叉验证的折数，通常用于训练时的验证集划分
    rand_crop_spatial_size=(96, 96, 96),    # 目标随机裁剪的空间大小
    convert_to_sam=False,   # 如果为True，则会将数据转换为特定格式
    do_test_crop=True,      # 是否对测试数据应用裁剪
    do_val_crop = True,     # 是否对验证数据应用裁剪
    do_nnunet_intensity_aug=False,          # 是否应用nnU-Net特有的强度增强
    #num_worker=4,           # 数据加载的CPU线程数
):
    datasets = []
    for data, path_prefix in zip(datas, path_prefixes):
        dataset = build_dataset(    # 负责初始化数据集，并根据传入的参数进行数据加载和预处理
            data=data,
            path_prefix=path_prefix,
            batch_size=batch_size,
            data_dir=data_dir,
            split=split,
            deterministic=deterministic,
            augmentation=augmentation,
            fold=fold,
            rand_crop_spatial_size=rand_crop_spatial_size,
            convert_to_sam=convert_to_sam,
            do_test_crop=do_test_crop,
            do_val_crop = do_val_crop,
            do_nnunet_intensity_aug=do_nnunet_intensity_aug,
            #num_worker=num_worker,
        )
        datasets.append(dataset)
    concat_dataset = ConcatDataset(datasets)    # 合并数据集
    # 创建dataloader
    if deterministic:
        loader = DataLoader(
            #concat_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
            concat_dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
    else:
        loader = DataLoader(
            #concat_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
            concat_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
    return loader

def load_data_volume_dataset(
    *,
    datas,
    path_prefixes,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    #num_worker=4,
):
    datasets = []
    for data, path_prefix in zip(datas, path_prefixes):
        dataset = build_dataset(
            data=data,
            path_prefix=path_prefix,
            batch_size=batch_size,
            data_dir=data_dir,
            split=split,
            deterministic=deterministic,
            augmentation=augmentation,
            fold=fold,
            rand_crop_spatial_size=rand_crop_spatial_size,
            convert_to_sam=convert_to_sam,
            do_test_crop=do_test_crop,
            do_val_crop = do_val_crop,
            do_nnunet_intensity_aug=do_nnunet_intensity_aug,
            #num_worker=num_worker,
        )
        datasets.append(dataset)
    concat_dataset = ConcatDataset(datasets)
    
    return concat_dataset
# 加载单一数据集
def load_data_volume_single(
    *,
    data,
    path_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    #num_worker=4,
):
    # 路径检验
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:
        data_dir = os.path.join(path_prefix, "split.pkl")
    # 加载数据集分割信息
    with open(data_dir, "rb") as f:
        d = pickle.load(f)[fold][split]
    # 图像和分割标签路径
    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]
    # 初始化数据集
    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )
    # 创建dataloader
    if deterministic:
        loader = DataLoader(
            #dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, drop_last=True
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )
    else:
        loader = DataLoader(
            #dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, drop_last=True
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
    return loader

def load_datasets(
    *,
    datas,
    path_prefixes,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(128, 128, 128),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    #num_worker=8,
):
    datasets = []
    for data, path_prefix in zip(datas, path_prefixes):
        dataset = build_dataset(
            data=data,
            path_prefix=path_prefix,
            batch_size=batch_size,
            data_dir=data_dir,
            split=split,
            deterministic=deterministic,
            augmentation=augmentation,
            fold=fold,
            rand_crop_spatial_size=rand_crop_spatial_size,
            convert_to_sam=convert_to_sam,
            do_test_crop=do_test_crop,
            do_val_crop = do_val_crop,
            do_nnunet_intensity_aug=do_nnunet_intensity_aug,
            #num_worker=num_worker,
        )
        datasets.append(dataset)
    concat_dataset = ConcatDataset(datasets)

    return concat_dataset

def build_dataset(
    *,
    data,
    path_prefix,
    batch_size,
    data_dir=None,
    split="train",
    deterministic=False,
    augmentation=False,
    fold=0,
    rand_crop_spatial_size=(96, 96, 96),
    convert_to_sam=False,
    do_test_crop=True,
    do_val_crop = True,
    do_nnunet_intensity_aug=False,
    #num_worker=4,
):
    # 数据目录验证
    if not path_prefix:
        raise ValueError("unspecified data directory")
    if data_dir is None:    # 设置数据路径
        data_dir = os.path.join(path_prefix, "split.pkl")
    # 加载数据集划分信息

    with open(data_dir, "rb") as f:
        d = pickle.load(f)[fold][split]
        #d = pickle.load(f)[split][fold]

    # 获取图像和标签文件路径
    img_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())]
    seg_files = [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())]
    # 构建数据集对象
    dataset = DATASET_DICT[data](
        img_files,
        seg_files,
        split=split,
        augmentation=augmentation,
        rand_crop_spatial_size=rand_crop_spatial_size,
        convert_to_sam=convert_to_sam,
        do_test_crop=do_test_crop,
        do_val_crop=do_val_crop,
        do_nnunet_intensity_aug=do_nnunet_intensity_aug,
    )
    
    return dataset