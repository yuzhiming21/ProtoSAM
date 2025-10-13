from dataset.datasets import load_data_volume
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry # type: ignore
import argparse
from torch.optim import AdamW
import numpy as np
import logging
from utils.script_util import save_checkpoint
from monai.losses import DiceCELoss, DiceLoss
import ast
import torch.nn.functional as F
#from modeling.Med_SAM.mask_decoder_mamba import MaskDecoder
from modeling.Med_SAM.mask_decoder_mamba_kmeans_0826 import MaskDecoder
#from modeling.Med_SAM.mask_decoder_simple import MaskDecoder
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
from sklearn.cluster import KMeans # type: ignore

from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter_kmeans_0826 import DynamicClusterBlock
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
def calculate_confidence_interval(data, confidence_level):
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

#def plot_metrics(train_losses, val_losses, save_path):
#    plt.figure(figsize=(10, 6))
#    plt.plot(train_losses, label="Train Loss")
#    plt.plot(val_losses, label="Val Loss")
#    plt.title("Training and Validation Loss Curve")
#    plt.xlabel("Epoch")
#    plt.ylabel("Loss")
#    plt.legend()
#    plt.grid(True)
#    plt.savefig(Path(save_path) / "loss_curve.png")
#    plt.close()

def init_cluster_centers(model, data_loader, device):
    """使用第一个batch的数据初始化聚类中心"""
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        img, seg, _ = batch[:3]
        img = batch[0].to(device)

        out = F.interpolate(img.float(), scale_factor=256/128, mode='trilinear')  # 根据实际输入尺寸调整
        input_batch = out.to(device)
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
        
        # 获取特征
        model = model.to(device)
        features = model.patch_embed(input_batch)
        flat_features = features.reshape(-1, features.size(-1)).cpu().numpy()

        cluster_block = next(b for b in model.blocks if isinstance(b, DynamicClusterBlock))
        prototypes = cluster_block.attn.prototypes

        # 使用K-means初始化原型
        kmeans = KMeans(n_clusters=model.num_clusters)
        kmeans.fit(flat_features)
        prototypes.prototypes = torch.from_numpy(kmeans.cluster_centers_).float().to(device)

        # 更新模型中的聚类中心
        for module in model.modules():
            if hasattr(module, 'centers'):
                module.centers.data = torch.from_numpy(kmeans.cluster_centers_).float().to(device)


def generate_pseudo_labels(model, unlabeled_imgs, seg, confidence_thresh=0.3):
    """
    生成高置信度伪标签
    参数:
        model: 当前模型
        unlabeled_imgs: 未标注图像 (B, C, D, H, W)
        confidence_thresh: 置信度阈值
    返回:
        pseudo_labels: 伪标签 (B, 1, D, H, W)
        confidence_mask: 高置信度掩码 (B, 1, D, H, W)
    """
    model.eval()
    with torch.no_grad():
        input_batch = F.interpolate(unlabeled_imgs.float(), scale_factor=2, mode='trilinear')
        batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
        input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)

        features = model(input_batch, batchsize=batchsize, points_feat=None, labels=seg, pseudo_labels=None)  # 假设返回元组的第一个元素是logits
        probs = torch.softmax(features, dim=1)
        max_probs, pseudo_labels = torch.max(probs, dim=1, keepdim=True)
        confidence_mask = (max_probs > confidence_thresh).float()
        pseudo_labels = pseudo_labels * confidence_mask
        
        # 添加边缘一致性约束
        edge_mask = F.avg_pool3d(max_probs, kernel_size=3, stride=1, padding=1) - \
                   F.avg_pool3d(max_probs, kernel_size=3, stride=1, padding=1)
        edge_mask = (edge_mask.abs() > 0.1).float()
        confidence_mask = confidence_mask * (1 - edge_mask)  # 降低边缘区域置信度

    return pseudo_labels, confidence_mask

def main():
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument(    # 选择训练模型
        "--method", default=None, type=str, choices=["sam", "sam3d", "baidu", "tri_attn_loraAdapter_pEncodeS_miniDe"]
    )
    parser.add_argument(    # 是否加载预训练权重
        "--pretrained", action="store_true"
    )
    parser.add_argument(    # 数据路径
        "--data", default=None, type=str
    )
    parser.add_argument(    # 存储模型检查点路径
        "--snapshot_path",
        default="",
        type=str,
    )
    parser.add_argument(    # 指定加载模型权重路径
        "--load_weight",
        default="",
        type=str,
    )
    parser.add_argument(    # 数据集路径前缀
        "--data_prefix",
        default="",
        type=str,
    )
    parser.add_argument(    # 数据增强时的随机裁剪大小
        "--rand_crop_size",
        default=0,
        nargs='+', type=int,
    )
    parser.add_argument(    # 输入图像大小
        "--input_image_size",
        default=256,
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
    parser.add_argument("--eval_interval", default=4, type=int) # 评估间隔
    parser.add_argument("--resume", action="store_true")
    #parser.add_argument("--num_worker", default=1, type=int)
    parser.add_argument("-tolerance", default=5, type=int)
    parser.add_argument("--num_clusters", default=32, type=int, help="Number of clusters")

    parser.add_argument("--active_learning", action="store_true", help="启用主动学习训练模式")
    parser.add_argument("--uncertainty_threshold", default=0.3, type=float, help="不确定性阈值")

    args = parser.parse_args()
    # 配置不同模型架构
    if args.method == "sam":
        from modeling.Med_SAM.image_encoder_sam import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "sam3d":
        from modeling.Med_SAM.image_encoder_sam3d import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "baidu":
        from modeling.Med_SAM.image_encoder_baidu_simple import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
    elif args.method == "tri_attn_loraAdapter_pEncodeS_miniDe":
        from modeling.Med_SAM.image_encoder_tri_attn_Lora_Adapter_kmeans_0826 import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
        import loralib as lora  # type: ignore
    else:
        raise "unknown method"
    input_image_size = args.input_image_size
    device = args.device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置随机裁剪尺寸
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
            
    #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic']")
    #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','pancreas','kits23','hepatic','liver','spleen','colon']")
    #args.weight_path = os.path.join(args.snapshot_path,"['total_lung_lower_lobe_left','total_lung_lower_lobe_right','total_lung_middle_lobe_right','total_lung_upper_lobe_left','total_lung_upper_lobe_right']")
    #args.weight_path = os.path.join(args.snapshot_path, "['total_spleen','total_pancreas','total_kidney_left','total_kidney_right','total_lung_lower_lobe_left','total_lung_lower_lobe_right','total_lung_middle_lobe_right','total_lung_upper_lobe_left','total_lung_upper_lobe_right']")
    args.weight_path = os.path.join(args.snapshot_path, "['lung_hospital','total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']")
    #args.weight_path = os.path.join(args.snapshot_path, "['lung','lung2','Lung421','lung_hospital','pancreas','kits23','hepatic','total_spleen','total_pancreas','total_kidney_right','total_lung_upper_lobe_right']")
    args.snapshot_path = os.path.join(args.snapshot_path, args.data)
    if not os.path.exists(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    # 设置日志记录
    setup_logger(logger_name="train", root=args.snapshot_path, screen=True, tofile=True)
    logger = logging.getLogger(f"train")
    logger.info(str(args))
    # 解析args数据并构建数据文件路径
    args.data = ast.literal_eval(args.data)
    args.data_prefix = [f"datafile/{dataset_name}" for dataset_name in args.data]
    #args.data_prefix = [f"datafile/{dataset_name}_crop" for dataset_name in args.data]
    #args.data_prefix = [f"datafile/{dataset_name}_crop_pruning" for dataset_name in args.data]
    #args.data_prefix = [f"../datafile/{dataset_name}_resize_simMask" for dataset_name in args.data]
    #args.data_prefix = [f"../datafile/{dataset_name}_resize_userMask" for dataset_name in args.data]
    print(args.data_prefix)
    # 加载训练验证数据
    train_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=args.batch_size,
        augmentation=True,
        split="train",
        rand_crop_spatial_size=args.rand_crop_size,
        #num_worker = args.num_worker
    )
    val_data = load_data_volume(
        datas=args.data,
        path_prefixes=args.data_prefix,
        batch_size=args.batch_size,
        augmentation=False,
        split="val",
        deterministic=True,
        rand_crop_spatial_size=args.rand_crop_size,
        #num_worker = args.num_worker
    )

    # 配置预训练权重
    if args.load_weight=="original":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/sam_vit_b_01ec64.pth")
    elif args.load_weight=="medsam":
        sam = sam_model_registry["vit_b"](checkpoint="ckpt/medsam_vit_b.pth")
    else:
        raise "Unknown pretrain weight."
    logger.info(f'Using pretrained weight: {args.load_weight}')
    # 加载自动掩码解码器，图像编码器
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
        out_chans=256,
        num_slice = 16,
        cluster_layers=(4, 8),  # 指定在第4层和第8层使用聚类注意力
        num_clusters=64,
        #num_clusters=args.num_clusters,
    )

    #hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=True)
    # 配置图像编码器参数
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
    # 加载预训练权重
    load_pretrained = args.pretrained
    file = "best_debug.pth.tar"     # 预训练权重
    #file = "last_debug.pth.tar"     # 预训练权重
    print("load_pretrained", load_pretrained)
    # 加载预训练权重或使用 mask_generator 中的编码器
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
        init_cluster_centers(img_encoder, train_data, device)
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
        init_cluster_centers(img_encoder, train_data, device)
        print("Loaded pretrained weights (excluding position encodings)")
    del sam     # 删除sam对象，通常用于释放不再使用的内存或避免不必要的变量占用内存空间
    img_encoder.to(device)
    # 初始化提示编码器并加载预训练权重
    prompt_encoder = PromptEncoderS(32)
    if load_pretrained:     # 
        prompt_encoder.load_state_dict(
            torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["feature_dict"], strict=True)
    prompt_encoder.to(device)

    # 加载掩码解码器并加载预训练权重
    mask_decoder = MaskDecoder()
    if load_pretrained:
        mask_decoder.load_state_dict(torch.load(os.path.join(args.weight_path, file), map_location='cpu', weights_only=False)["decoder_dict"],
                            strict=True)
    mask_decoder.to(device)
    # 计算可训练参数
    logger.info(f'The img_encoder model has {count_parameters(img_encoder):,}M trainable parameters.')
    logger.info(f'The mask_decoder model has {count_parameters(mask_decoder):,}M trainable parameters.')
    # 加载优化器和学习率调度器
    encoder_opt = AdamW(img_encoder.parameters(), lr=args.lr, weight_decay=0)
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=args.max_epoch)
    feature_opt = AdamW(prompt_encoder.parameters(), lr=args.lr, weight_decay=0)
    feature_scheduler = torch.optim.lr_scheduler.LinearLR(feature_opt, start_factor=1.0, end_factor=0.01,
                                                          total_iters=args.max_epoch)
    decoder_opt = AdamW([i for i in mask_decoder.parameters() if i.requires_grad == True], lr=args.lr, weight_decay=0)
    decoder_scheduler = torch.optim.lr_scheduler.LinearLR(decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=args.max_epoch)
    dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
    loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    best_loss = np.inf
    patch_size = args.rand_crop_size[0]
    debug_time = True
    debug = False
    # 初始化距离映射
    dis_map = DistMaps(2, use_disks=True)

    # 初始化不确定性采样器
    uncertainty_sampler = UncertaintySampler(num_points=5, strategy='entropy')
    boundary_sampler = UncertaintySampler(num_points=3, strategy='confidence')
    
    train_loss_history = []
    val_loss_history = []

    for epoch_num in range(args.max_epoch):
        loss_summary = []
        hd95_overview = []
        
        # 训练
        img_encoder.train()
        prompt_encoder.train()
        mask_decoder.train()

        if debug_time:
            batch_end = time.perf_counter()  # 高精度计时

        #use_click = 0
        for idx, (img, seg, spacing) in enumerate(train_data):
            if debug_time:
                batch_start = time.perf_counter()  # 高精度计时
                print("data loading spend time", batch_start - batch_end)
                logger.info(f"data loading {idx} time: {batch_end - batch_start:.4f} sec")
            
            organ_types = torch.zeros(img.size(0), dtype=torch.long, device=device)  # 生成默认器官类型

            #if (idx < 100) or (idx <= 100 and idx % 10 == 0):
            #if epoch_num < 5 or idx % 10 == 0:
            if idx <= 400:
                pseudo = False
            else:
                pseudo = True

            img = img.to(device)
            seg = seg.to(device) if seg is not None else None

            out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')
            input_batch = out.to(device)
            batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
            input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)

            # 是否使用点击特征
            use_click = 1
            if use_click:
                with torch.no_grad():
                    # 获取初始预测
                    initial_features = img_encoder(input_batch, batchsize, None, labels=seg, pseudo_labels=None)
                    initial_masks, initial_confidence = mask_decoder(initial_features)
                    initial_masks = initial_masks.permute(0, 1, 4, 2, 3)
                    
                    # 基于不确定性采样点
                    uncertain_points, uncertainties = uncertainty_sampler.sample_points(
                        initial_masks, seg.unsqueeze(1) if seg is not None else None
                    )
                
                # 生成点击特征
                this_batch_points_feature = []
                for b in range(batchsize):
                    points_batch = uncertain_points[b]
                    positive_points = []
                    negative_points = []
                    
                    for point in points_batch:
                        d, h, w = point.int().tolist()
                        pred_label = torch.argmax(initial_masks[b, :, d, h, w])
                        if pred_label == 1:
                            positive_points.append(point.unsqueeze(0))
                        else:
                            negative_points.append(point.unsqueeze(0))
                    
                    # 确保正负样本存在
                    if len(positive_points) == 0 and seg is not None:
                        pos_coords = torch.where(seg[b] == 1)
                        if len(pos_coords[0]) > 0:
                            idx_p = np.random.choice(len(pos_coords[0]))
                            positive_points.append(torch.tensor([
                                pos_coords[0][idx_p], pos_coords[1][idx_p], pos_coords[2][idx_p]
                            ], device=device).unsqueeze(0))
                    
                    if len(negative_points) == 0 and seg is not None:
                        neg_coords = torch.where(seg[b] == 0)
                        if len(neg_coords[0]) > 0:
                            idx_n = np.random.choice(len(neg_coords[0]))
                            negative_points.append(torch.tensor([
                                neg_coords[0][idx_n], neg_coords[1][idx_n], neg_coords[2][idx_n]
                            ], device=device).unsqueeze(0))
                    
                    # 生成特征
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
                
                    this_batch_points_feature.append(
                        torch.cat([positive_feat, negative_feat], dim=1)
                    )
                    
                prompt_input = torch.cat(this_batch_points_feature, 0).float()
            else:
                prompt_input = torch.zeros(batchsize, 2, 128, 128, 128).to(device)

            point_feature = prompt_encoder(prompt_input)

            # 区分有监督与无监督
            if pseudo:
                # 无监督分支：生成伪标签
                pseudo_labels, confidence_mask = generate_pseudo_labels(
                    img_encoder, 
                    img,
                    seg,
                    confidence_thresh=0.5
                )
                batch_features = img_encoder(input_batch, batchsize, point_feature, labels=None, pseudo_labels=pseudo_labels)
            else:
                # 有监督分支
                batch_features = img_encoder(input_batch, batchsize, point_feature, labels=seg, pseudo_labels=None)
            masks, confidence_map = mask_decoder(batch_features)
            masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W
            # 损失计算
            seg = seg.unsqueeze(1)
            # channel是3和2的时候计算不一样
            loss = loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            # 梯度清零
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            feature_opt.zero_grad()
            loss.backward()     # 反向传播
            # 梯度裁剪，防止梯度爆炸
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(train_data)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(prompt_encoder.parameters(), 1.0)
            # 优化器更新
            encoder_opt.step()
            feature_opt.step()
            decoder_opt.step()
            
            if debug_time:
                batch_end = time.perf_counter()  # 高精度计时
                print("batch spend time", batch_end - batch_start)
                logger.info(f"Batch {idx} time: {batch_end - batch_start:.4f} sec")

            #logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # 学习率更新
        encoder_scheduler.step()
        feature_scheduler.step()
        decoder_scheduler.step()
        logger.info("- Train metrics: " + str(np.mean(loss_summary)))

        train_loss_history.append(np.mean(loss_summary))

        # 验证
        img_encoder.eval()
        prompt_encoder.eval()
        mask_decoder.eval()
        with torch.no_grad():   # 关闭梯度计算
            
            for idx, (img, seg, spacing, path) in enumerate(val_data):
                loss_summary = []   # 损失记录表
                # print('seg: ', seg.sum())
                out = F.interpolate(img.float(), scale_factor=input_image_size / patch_size, mode='trilinear')  # TODO: for 256
                input_batch = out.to(device)
                batchsize, d_channel, d_slice, d_h, d_w = input_batch.shape
                input_batch = input_batch.permute(0, 2, 1, 3, 4).reshape(-1, d_channel, d_h, d_w)
                seg = seg.to(device)
                # 使用与训练阶段一致的预测引导点击策略
                this_batch_points_feature = []
                for b in range(batchsize):
                    _seg = seg[b].unsqueeze(0)
            
                    # 获取初始预测来引导点击
                    initial_features = img_encoder(input_batch, batchsize, None)
                    initial_masks, initial_confidence = mask_decoder(initial_features)
                    initial_masks = initial_masks.permute(0, 1, 4, 2, 3)
            
                    # 基于不确定性采样点（与训练一致）
                    uncertain_points, _ = uncertainty_sampler.sample_points(
                        initial_masks, _seg.unsqueeze(1) if _seg is not None else None
                    )
            
                    points_batch = uncertain_points[b]  # [num_points, 3]
            
                    # 为正负样本分配标签（基于初始预测）
                    positive_points = []
                    negative_points = []
            
                    for point in points_batch:
                        d, h, w = point.int().tolist()
                        pred_label = torch.argmax(initial_masks[b, :, d, h, w])
                
                        if pred_label == 1:  # 预测为正样本
                            positive_points.append(point.unsqueeze(0))
                        else:  # 预测为负样本
                            negative_points.append(point.unsqueeze(0))
            
                    # 确保有正负样本（使用真实标注作为后备）
                    if len(positive_points) == 0:
                        pos_coords = torch.where(_seg[0] == 1)
                        if len(pos_coords[0]) > 0:
                            idx_p = np.random.choice(len(pos_coords[0]), min(3, len(pos_coords[0])), replace=False)
                            for i_p in idx_p:
                                positive_points.append(torch.tensor([
                                    pos_coords[0][i_p], pos_coords[1][i_p], pos_coords[2][i_p]
                                ], device=device).unsqueeze(0))
            
                    if len(negative_points) == 0:
                        neg_coords = torch.where(_seg[0] == 0)
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
            
                    this_batch_points_feature.append(
                        torch.cat([positive_feat, negative_feat], dim=1)
                    )
                # 通过模型进行特征提取和掩码生成
                prompt_input = torch.cat(this_batch_points_feature, 0).float()
                point_feature = prompt_encoder(prompt_input)
                batch_features = img_encoder(input_batch, batchsize, point_feature)  # bs, C, H, W, D
                masks, confidence_map = mask_decoder(batch_features)
                masks = masks.permute(0, 1, 4, 2, 3)  # bs, C, D, H, W
                # 计算损失
                seg = seg.unsqueeze(1)
                # channel是3和2的时候计算不一样
                loss = dice_loss(masks, seg)
                loss_summary.append(loss.detach().cpu().numpy())
                logger.info(
                    'epoch: {}/{}, iter: {}/{}'.format(epoch_num, args.max_epoch, idx, len(val_data)) + ": loss:" + str(
                        loss_summary[-1].flatten()[0]))
                val_loss_history.append(np.mean(loss_summary))
        logger.info("- Val metrics: " + str(np.mean(loss_summary)))
        # 判断是否为最佳模型
        is_best = False
        if np.mean(loss_summary) < best_loss:
            best_loss = np.mean(loss_summary)
            is_best = True
            
        img_encoder.train()  # very tricky
        # NOTE: The devil lies in the detail:
        # Calling model.eval() will trigger the merging of LoRA parameters with the corresponding pretrained ones, which eliminates additional latency for subsequent forward passes. Calling model.train() again will undo the merge. This can be disabled by passing merge_weights=False to LoRA layers.
        save_checkpoint({"epoch": epoch_num + 1,
                        "best_val_loss": best_loss,
                         "encoder_dict": img_encoder.state_dict(),
                         "decoder_dict": mask_decoder.state_dict(),
                         "feature_dict": prompt_encoder.state_dict(),
                         "encoder_opt": encoder_opt.state_dict(),
                         "feature_opt": feature_opt.state_dict(),
                         "decoder_opt": decoder_opt.state_dict()
                         },
                        is_best=is_best,
                        checkpoint=args.snapshot_path)
        logger.info("- Val metrics best: " + str(best_loss))

        #plot_metrics(train_loss_history, val_loss_history, args.snapshot_path)

if __name__ == "__main__":
    set_seed(42)
    main()


