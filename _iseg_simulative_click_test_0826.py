from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
import sys
from monai.losses import DiceCELoss, DiceLoss
import ast
import torch.nn.functional as F
#from modeling.Med_SAM.mask_decoder_mamba import MaskDecoder
from modeling.Med_SAM.mask_decoder_mamba_kmeans_0826 import MaskDecoder
import torch
from modeling.Med_SAM.prompt_encoder_simple import PromptEncoderS
#from modeling.Med_SAM.prompt_encoder_sam import TwoWayTransformer
from utils.click_encoding import DistMaps
import torch.nn as nn
from functools import partial
import os
from utils.util import setup_logger, count_parameters
import time
import random
from utils.online_utils import *
from utils.click_utils import get_click_batch
from monai.metrics import HausdorffDistanceMetric
import matplotlib.pyplot as plt
from pathlib import Path

from modeling.Med_SAM.mask_decoder_mamba_kmeans_0826 import UncertaintySampler


# 随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#置信区间
def calculate_confidence_interval(data, confidence_level=0.95):
    mean = np.mean(data)
    std_dev = np.std(data)
    n = len(data)
    if confidence_level == 0.95:
        z_value = 1.96
    elif confidence_level == 0.99:
        z_value = 2.576
    elif confidence_level == 0.90:
        z_value = 1.645
    else:
        raise ValueError("Unsupported confidence level")
    margin_of_error = z_value * (std_dev / np.sqrt(n))
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound

def sample_points_based_on_strategy(strategy, model, img, seg, initial_masks, num_points, device):
    """根据策略采样点"""
    if strategy == "random":
        # 随机采样
        return sample_random_points(seg, num_points)
    elif strategy == "uncertainty":
        # 不确定性采样
        uncertainty_sampler = UncertaintySampler(num_points=num_points, strategy='entropy')
        points, _ = uncertainty_sampler.sample_points(initial_masks, seg.unsqueeze(1))
        return points
    elif strategy == "boundary":
        # 边界采样
        pred_mask = (F.softmax(initial_masks, dim=1)[:, 1] > 0.5).float()
        boundary_sampler = UncertaintySampler(num_points=num_points, strategy='confidence')
        points = boundary_sampler.get_boundary_points(pred_mask.unsqueeze(1), num_points)
        return points
    elif strategy == "mixed":
        # 混合策略：一半随机，一半基于不确定性
        random_points = sample_random_points(seg, num_points // 2)
        uncertainty_sampler = UncertaintySampler(num_points=num_points - num_points // 2, strategy='entropy')
        uncertain_points, _ = uncertainty_sampler.sample_points(initial_masks, seg.unsqueeze(1))
        return torch.cat([random_points, uncertain_points], dim=1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def sample_random_points(seg, num_points):
    """随机采样点"""
    batch_size = seg.shape[0]
    sampled_points = []
    
    for b in range(batch_size):
        # 从前景和背景中分别采样
        foreground_coords = torch.where(seg[b] == 1)
        background_coords = torch.where(seg[b] == 0)
        
        points = []
        if len(foreground_coords[0]) > 0:
            # 采样前景点
            num_foreground = min(num_points // 2, len(foreground_coords[0]))
            indices = np.random.choice(len(foreground_coords[0]), num_foreground, replace=False)
            for idx in indices:
                points.append([foreground_coords[0][idx], foreground_coords[1][idx], foreground_coords[2][idx]])
        
        if len(background_coords[0]) > 0:
            # 采样背景点
            num_background = num_points - len(points)
            indices = np.random.choice(len(background_coords[0]), num_background, replace=False)
            for idx in indices:
                points.append([background_coords[0][idx], background_coords[1][idx], background_coords[2][idx]])
        
        sampled_points.append(torch.tensor(points, device=seg.device))
    
    return torch.stack(sampled_points)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default=None, type=str, choices=["sam", "baidu", "tri_attn_loraAdapter_pEncodeS_miniDe"]
    )
    parser.add_argument(
        "--pretrained", action="store_true"
    )
    parser.add_argument(
        "--data", default=None, type=str
    )
    parser.add_argument(
        "--use_ft_weight", default='no', type=str
    )
    parser.add_argument(
        "--save_result", default='no', type=str
    )
    parser.add_argument(
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(
        "--load_weight",
        default="",
        type=str,
    )
    parser.add_argument(
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(
        "--input_image_size",
        default=256,
        type=int,
    )
    parser.add_argument(
        "--num_click",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--device",
        default="cuda:2",
        type=str,
    )
    parser.add_argument("-bs", "--batch_size", default=1, type=int)
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_epoch", default=500, type=int)
    parser.add_argument("--eval_interval", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    #parser.add_argument("--num_worker", default=12, type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--click_strategy", default="uncertainty", choices=["random", "uncertainty", "boundary", "mixed"], help="点击策略选择")
    parser.add_argument("--max_interactions", default=10, type=int, help="最大交互次数")

    args = parser.parse_args()

    if args.method == "sam":
        from modeling.Med_SAM.image_encoder_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "baidu":
        from modeling.Med_SAM.image_encoder_baidu_simple import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        #from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter_kmeans_0826 import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora
    else:
        raise "unknown method"
    input_image_size = args.input_image_size
    device = args.device
    if args.rand_crop_size == 0:
        if args.data in ["kits", "colon"]:
            args.rand_crop_size = (256, 256, 256)
        if args.data in ["pancreas", "lits", "brain", "hepatic", "kits23"]:
            args.rand_crop_size = (128, 128, 128)
    else:
        if len(args.rand_crop_size) == 1:
            args.rand_crop_size = tuple(args.rand_crop_size * 3)
        else:
            args.rand_crop_size = tuple(args.rand_crop_size)
    if args.use_ft_weight == 'no':
        #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic']")
        #args.weight_path = os.path.join(args.snapshot_path, "['total_spleen','total_pancreas','total_lung_upper_lobe_right','total_kidney_right']")
        #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic','total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']")
        #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic','liver','spleen','colon']")
        #args.weight_path = os.path.join(args.snapshot_path, "['total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right','lung_hospital']")
        args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic','lung_hospital']")
    elif args.use_ft_weight == 'yes':
        args.weight_path = os.path.join(args.snapshot_path, args.data)
    else:
        raise "Error"
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)

    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))

    args.data = ast.literal_eval(args.data)
    args.data_prefix = []
    for dataset_name in args.data:
        args.data_prefix.append(f"datafile/{dataset_name}")
        #args.data_prefix.append(f"datafile/{dataset_name}_crop")
    #args.data_prefix = [f"../datafile/{dataset_name}_resize_simMask" for dataset_name in args.data]
    #args.data_prefix = [f"../datafile/{dataset_name}_resize_userMask" for dataset_name in args.data]
    print(args.data_prefix)

    val_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=1,
        augmentation=True, # train情况下需要是True不然会报错
        split="test",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        #num_worker = args.num_worker
    )
    
    if args.load_weight=="original":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    elif args.load_weight=="medsam":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/medsam_vit_b.pth")
    else:
        raise "Unknown pretrain weight."
    logger.info(f'Using pretrained weight: {args.load_weight}')

    mask_generator = SamAutomaticMaskGenerator(sam)
    img_encoder = ImageEncoderViT_3d(
        depth=12,
        embed_dim=768,
        img_size=1024,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[2, 5, 8, 11],
        window_size=14,
        #cubic_window_size=8,
        out_chans=256,
        num_slice = 16,
        cluster_layers=(4, 8),  # 指定在第4层和第8层使用聚类注意力
        num_clusters=64  # 设置聚类中心数量
    )

    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)
    # Choose training weights
    if args.method == "baidu":
        for p in img_encoder.parameters():
            p.requires_grad = False
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.MLP_Adapter.parameters():
                p.requires_grad = True
            for p in i.Space_Adapter.parameters():
                p.requires_grad = True
            for p in i.Depth_Adapter.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for p in img_encoder.neck.parameters():
            p.requires_grad = True
    elif args.method == "sam":
        for p in img_encoder.parameters():
            p.requires_grad = False
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            # for p in i.adapter.parameters():
            #     p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for p in img_encoder.neck_3d.parameters():
            p.requires_grad = True
        
    elif args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        
        lora.mark_only_lora_as_trainable(img_encoder)
        logger.info(f'LORA The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
        
        img_encoder.depth_embed.requires_grad = True
        for p in img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in img_encoder.blocks:
            #for p in i.TailAdapter.parameters():
            #    p.requires_grad = True
            #i.gamma.requires_grad = True
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            #i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for p in img_encoder.neck.parameters():
            p.requires_grad = True
    else:
        raise "wtf network are you used?"

    load_pretrained = args.pretrained
    file = "best_debug.pth.tar"
    #file = "last_debug.pth.tar"
    
    print("load_pretrained", load_pretrained)
    if not load_pretrained:
        #img_encoder.load_state_dict(mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
        pretrained_dict = mask_generator.predictor.model.image_encoder.state_dict()
        model_dict = img_encoder.state_dict()
        # 过滤掉不兼容的位置编码参数
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                 if not k.endswith(('rel_pos_h', 'rel_pos_w'))}
        # 部分加载权重
        model_dict.update(filtered_dict)
        img_encoder.load_state_dict(model_dict, strict=False)
        print("Loaded pretrained weights (excluding position encodings)")
    else:
        #img_encoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["encoder_dict"], strict=True)
        pretrained_dict = torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["encoder_dict"]
        model_dict = img_encoder.state_dict()
        # 过滤掉不兼容的位置编码参数
        filtered_dict = {k: v for k, v in pretrained_dict.items() 
                 if not k.endswith(('rel_pos_h', 'rel_pos_w'))}
        # 部分加载权重
        model_dict.update(filtered_dict)
        img_encoder.load_state_dict(model_dict, strict=False)
        print("Loaded pretrained weights (excluding position encodings)")
    del sam
    img_encoder.to(device)
    
    prompt_encoder = PromptEncoderS(32)
    if load_pretrained:
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["feature_dict"], strict=True)
    prompt_encoder.to(device)
    
    mask_decoder = MaskDecoder()
    
    if load_pretrained:
        mask_decoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["decoder_dict"],
                            strict=False)
    mask_decoder.to(device)

    logger.info(f'The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
    logger.info(f'The mask_decoder model has {count_parameters(mask_decoder):,}M trainable parameters.')

    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    patch_size = args.rand_crop_size[0]
    debug_time = True

    loss_summary = []
    hd95_overview = []
    loss_overview = []
    test_loss_history = []
    
    dis_map = DistMaps(2, use_disks=True)
    
    # 初始化不确定性采样器
    uncertainty_sampler = UncertaintySampler(num_points=5, strategy='entropy')

    img_encoder.eval()
    prompt_encoder.eval()
    mask_decoder.eval()
    
    if debug_time:
        batch_end = time.perf_counter()
        
    os.makedirs(f"Online_TrainPredict_0707/3/{args.data}", exist_ok=True)
    #os.makedirs(f"Online_TrainPredict/{args.data}", exist_ok=True)

    for idx, (img, seg, spacing, *_rest) in enumerate(val_data):
        
        if debug_time:
            batch_start = time.perf_counter()
            print("data loading spend time", batch_start - batch_end)
            logger.info(f"data loading {idx} time: {batch_end - batch_start:.4f} sec")
        
        out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: 256 input
        input_batch = out.to(device)
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
        
        seg = seg.to(device)

        # 获取初始预测
        with torch.no_grad():
            initial_features = img_encoder(input_batch, batchsize, None)
            initial_masks, initial_confidence = mask_decoder(initial_features)
            initial_masks = initial_masks.permute(0, 1, 4, 2, 3)
        
        # 根据策略采样初始点
        initial_points = sample_points_based_on_strategy(
            args.click_strategy, img_encoder, img, seg, initial_masks, 
            args.num_click, device
        )
        
        # 分离正负样本点
        positive_points = []
        negative_points = []
        
        for point in initial_points[0]:  # batch_size=1
            d, h, w = point.int().tolist()
            pred_label = torch.argmax(initial_masks[0, :, d, h, w])
            
            if pred_label == 1:  # 预测为正样本
                positive_points.append(point.unsqueeze(0))
            else:  # 预测为负样本
                negative_points.append(point.unsqueeze(0))
        
        # 确保有正负样本
        if len(positive_points) == 0:
            # 从真实标注中采样正样本
            pos_coords = torch.where(seg[0] == 1)
            if len(pos_coords[0]) > 0:
                idx_p = np.random.choice(len(pos_coords[0]), min(3, len(pos_coords[0])), replace=False)
                for i_p in idx_p:
                    positive_points.append(torch.tensor([
                        pos_coords[0][i_p], pos_coords[1][i_p], pos_coords[2][i_p]
                    ], device=device).unsqueeze(0))
        
        if len(negative_points) == 0:
            # 从真实标注中采样负样本
            neg_coords = torch.where(seg[0] == 0)
            if len(neg_coords[0]) > 0:
                idx_n = np.random.choice(len(neg_coords[0]), min(3, len(neg_coords[0])), replace=False)
                for i_n in idx_n:
                    negative_points.append(torch.tensor([
                        neg_coords[0][i_n], neg_coords[1][i_n], neg_coords[2][i_n]
                    ], device=device).unsqueeze(0))
        
        # 生成点击特征
        if positive_points:
            points_pos = torch.cat(positive_points, dim=0)
            positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
        else:
            positive_feat = torch.zeros(1, 1, 128, 128, 128).to(device)
        
        if negative_points:
            points_neg = torch.cat(negative_points, dim=0)
            negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
        else:
            negative_feat = torch.zeros(1, 1, 128, 128, 128).to(device)
        
        prompt_input = torch.cat([positive_feat, negative_feat], dim=1)
       
        with torch.no_grad():
            point_feature = prompt_encoder(prompt_input)
            batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
            masks, confidence_map = mask_decoder(batch_features)
        masks = masks.to(device)
        masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W   [2, 2, 128, 128, 128]
        
        seg = seg.unsqueeze(1)
        # channel是3和2的时候计算不一样
        loss = dice_loss(masks, seg)
        print("original loss", round(float(loss), 5))

        for _sim_idx in range(40):
            masks_for_click = F.softmax(masks.clone().detach().cpu(), dim=1)[:,1] > 0.5
            seg_for_click = seg.clone().squeeze(1).detach().cpu()
            click_pos, is_positive = get_click_batch(masks_for_click, seg_for_click)
            #print(click_pos, is_positive)  # D, H, W
            click_pos = click_pos.to(device)
            
            # NOTE: 只能bs=1
            if is_positive[0]:
                points_pos = torch.cat([points_pos, click_pos], dim = 0)
            else:
                points_neg = torch.cat([points_neg, click_pos], dim = 0)
            
            #print("dismap input points_pos", points_pos.shape)
            new_positive_feat = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
            new_negative_feat = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
            
            prompt_input = torch.cat([new_positive_feat, new_negative_feat], dim=1)
            
            with torch.no_grad():
                point_feature = prompt_encoder(prompt_input)
                batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
                masks, confidence_map = mask_decoder(batch_features)
            masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W   [2, 2, 128, 128, 128]
            loss = dice_loss(masks, seg)
            print(_sim_idx, "loss", round(float(loss), 5))
        
        pos_disk = dis_map.get_coord_features(points_pos, 1, 128, 128, 128)
        neg_disk = dis_map.get_coord_features(points_neg, 1, 128, 128, 128)
        
        if debug_time:
            batch_end = time.perf_counter()  # 高精度计时
            print("batch spend time", batch_end - batch_start)
            logger.info(f"Batch {idx} time: {batch_end - batch_start:.4f} sec")

        if args.save_result == 'yes':
            torch.save(
                {
                    "img": img.detach().cpu(),
                    "predict": masks.detach().cpu(), # bs, C, D, H, W
                    "GT": seg.detach().cpu(), # bs, 1, D, H, W
                    "pos_clicks": points_pos.detach().cpu(),
                    "neg_clicks": points_neg.detach().cpu(),
                    "pos_disk": pos_disk.detach().cpu(),
                    "neg_disk": neg_disk.detach().cpu(),
                    "dice_score": round(float(loss), 5),
                    #"decoder_feats": decoder_feats.detach().cpu(),
                }
                , f"Online_TrainPredict_0714/samed/{args.data}/data_{idx}.pt")
        
    
        loss_summary.append(loss.detach().cpu().numpy())
        logger.info(
            'iter: {}/{}'.format(idx, len(val_data)) + ": loss:" + str(
                loss_summary[-1].flatten()[0]))
        
        masks = F.softmax(masks, dim=1)#[:, 1]
        masks = masks > 0.5
        hd95 = hd95_metric(masks, seg)

        loss_overview.append(loss.mean().detach().cpu().numpy())
        if not torch.isnan(hd95):
            hd95_overview.append(hd95.mean().detach().cpu().numpy())
   
    logger.info("- Val metrics: " + str(np.mean(loss_summary)))

    print("dice score")
    dice_score = 1 - sum(loss_overview) / len(loss_overview)
    print(dice_score)
    logger.info(f"- Dice score: {dice_score:.4f}")
    dice_lower, dice_upper = calculate_confidence_interval(loss_overview)
    print(f"loss_summary 置信区间: ({round(dice_lower, 4)}, {round(dice_upper, 4)})")
    logger.info(f"- Dice CI: ({round(1 - dice_upper, 4)}, {round(1 - dice_lower, 4)})")

    print("hd95")
    hd95_score = sum(hd95_overview) / len(hd95_overview)
    print(hd95_score)
    logger.info(f"- HD95: {hd95_score:.4f}")
    hd95_lower, hd95_upper = calculate_confidence_interval(hd95_overview)
    print(f"hd95_overview 置信区间: ({round(hd95_lower, 4)}, {round(hd95_upper, 4)})")
    logger.info(f"- hd95_score CI: ({round(hd95_lower, 4)}, {round(hd95_upper, 4)})")

if __name__ == "__main__":
    set_seed(42)
    main()