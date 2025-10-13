import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import loralib as lora
from typing import Optional, Tuple, Type
import numpy as np

class NonparametricPrototypes(nn.Module):
    """非参数化原型聚类层"""
    def __init__(self, num_clusters, feature_dim, alpha=0.1, momentum=0.999):
        super().__init__()
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.alpha = alpha
        # 非学习原型（通过数据驱动更新）
        self.register_buffer("prototypes", torch.zeros(num_clusters, feature_dim))
        
    def forward(self, x):
        """
        输入: 
            x - (B, N, C) 特征
            labels - (B, N) 像素类别标签（训练时提供）
        输出: 
            soft_assign - (B, N, K) 软分配概率
            hard_assign - (B, N) 硬分配
        """
        B, N, C = x.shape
        x_flat = x.view(-1, C)
        
        # 计算特征与原型的距离（余弦相似度）
        x_norm = F.normalize(x_flat, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        distances = -torch.mm(x_norm, p_norm.t())  # (B*N, K)
        distances = distances.view(B, N, self.num_clusters)
        
        # 软分配（可微分）
        soft_assign = F.softmax(-self.alpha * distances, dim=-1)
        # 硬分配（非可微分）
        with torch.no_grad():
            hard_assign = torch.argmin(distances, dim=-1)
            self.update_prototypes(x_flat.detach(), hard_assign.view(-1))
        
        return soft_assign, hard_assign
    
    @torch.no_grad()
    def update_prototypes(self, x, labels):
        """动量更新原型"""
        x_norm = F.normalize(x, p=2, dim=-1)
        for k in range(self.num_clusters):
            mask = (labels == k)
            if mask.sum() > 0:
                cluster_features = x_norm[mask]
                new_proto = cluster_features.mean(dim=0)
                self.prototypes[k] = self.momentum * self.prototypes[k] + (1 - self.momentum) * new_proto

class DynamicClusterAttention(nn.Module):
    """完全动态的聚类增强注意力机制"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_clusters: int = 64,
        use_lora: bool = True,
        alpha: float = 0.1,
        momentum: float = 0.999,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_clusters = num_clusters
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 可微分聚类层
        self.prototypes = NonparametricPrototypes(num_clusters, dim, momentum)
        
        # 注意力投影层（LoRA可选）
        if use_lora:
            self.qkv = lora.MergedLinear(dim, 3*dim, r=96, enable_lora=[True, False, True])
            self.proj = lora.Linear(dim, dim, r=96)
        else:
            self.qkv = nn.Linear(dim, 3 * dim)
            self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None , pseudo_labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        B, H, W, C = x.shape
        N = H * W
        x_flat = x.reshape(B, N, C)
        
        # 动态聚类分配（完全可微分）
        soft_assign, _ = self.prototypes(x_flat)
        
        # 训练时更新原型
        if labels is not None and self.training:
            self.prototypes.update_prototypes(x_flat, labels.flatten())
        if pseudo_labels is not None and self.training:
            self.prototypes.update_prototypes(x_flat, pseudo_labels.flatten())

        # 计算聚类特征（加权平均）
        cluster_features = torch.matmul(soft_assign.transpose(1, 2), x_flat)  # (B, K, C)
        cluster_counts = soft_assign.sum(dim=1, keepdim=True).transpose(1, 2)  # (B, K, 1)
        cluster_features = cluster_features / (cluster_counts + 1e-6)

        # 聚类中心注意力
        qkv_cluster = self.qkv(cluster_features)
        qkv_cluster = qkv_cluster.reshape(B, self.num_clusters, 3, self.num_heads, self.head_dim)
        q_cluster, k_cluster, v_cluster = qkv_cluster.permute(2, 0, 3, 1, 4).unbind(0)

        # 跨聚类注意力
        attn_cluster = (q_cluster @ k_cluster.transpose(-2, -1)) * self.scale
        attn_cluster = attn_cluster.softmax(dim=-1)
        cluster_output = (attn_cluster @ v_cluster).transpose(1, 2).reshape(B, self.num_clusters, -1)
        
        output = torch.matmul(soft_assign, cluster_output)
        output = self.proj(output)
        
        return output.reshape(B, H, W, -1)

class DynamicClusterBlock(nn.Module):
    """动态聚类Transformer块"""
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_clusters: int = 64,
        mlp_ratio: float = 4.0,
        use_lora: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DynamicClusterAttention(
            dim,
            num_heads=num_heads,
            num_clusters=num_clusters,
            use_lora=use_lora,
        )
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        
    def forward(self, x , labels=None, pseudo_labels=None):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, labels, pseudo_labels)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class ImageEncoderViT_3d_v2(nn.Module):
    """原型损失计算"""
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        patch_depth: int=32,
        num_slice=1,
        cluster_layers: Tuple[int, ...] = (),
        num_clusters: int = 64,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_slice = num_slice
        self.num_clusters = num_clusters

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            use_cluster_attn = i in cluster_layers
            block = DynamicClusterBlock(
                dim=embed_dim,
                num_heads=num_heads,
                num_clusters=num_clusters,
                mlp_ratio=mlp_ratio,
                use_lora=True,
                norm_layer=norm_layer,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

        self.slice_embed = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim,
                                         kernel_size=(1,1,self.num_slice), stride=(1,1,self.num_slice),
                                         groups=embed_dim)
        
        self.depth_embed = nn.Parameter(
                torch.ones(1, patch_depth, embed_dim)
            )

    def forward(self, x: torch.Tensor, batchsize = 1, points_feat = None, labels = None, pseudo_labels = None) -> torch.Tensor:
        # x: 256, 3, 256, 256
        # x: bs x 256, 3, 256, 256
        #print("input", x.shape)
        with torch.no_grad():
            x = self.patch_embed(x)  # x: D, H, W, C

        bsxD, H, W, C = x.shape
        if self.num_slice > 1:
            x = self.slice_embed(x.reshape(batchsize, -1, H, W, C).permute(0, 4, 2, 3, 1))  # bs, C, H, W, D做3D Conv
            x = x.permute(0, 2, 3, 4, 1)  # bs, H, W, D, C
        else:
            x = x.permute(1, 2, 0, 3).unsqueeze(0)
        
        if self.pos_embed is not None:
            #test
            #pos_embed = self.simplified_pos_embed
            #train
            pos_embed = F.avg_pool2d(self.pos_embed.permute(0,3,1,2), kernel_size=2).permute(0,2,3,1).unsqueeze(3)
            pos_embed = pos_embed + (self.depth_embed.unsqueeze(1).unsqueeze(1))
            pos_embed = F.interpolate(pos_embed.float().permute(0, 4, 1, 2, 3), scale_factor=0.5, mode='trilinear').permute(0, 2, 3, 4, 1)  # TODO: this change for 256 input
            x = x + pos_embed

        # print("x in block", x.shape)  # [1, 16, 16, 16, 768]
        bs, h, w, d, c = x.shape
        # bs, c, d, h, w -> bs, h, w, d, c
        if not points_feat is None:
            points_feat = points_feat.permute(0, 3, 4, 2, 1)
            x = x + points_feat
        # bs, d, h, w, c
        x = x.permute(0, 3, 1, 2, 4).reshape(bs*d, h, w, c)
        
        for blk in self.blocks:
            x = blk(x)
            
        #x = self.neck(x.permute(0, 3, 1, 2)).reshape(bs, d, 256, h, w).permute(0, 2, 3, 4, 1)
        # 修改neck部分输出，确保维度正确
        x = self.neck(x.permute(0, 3, 1, 2))  # [bs*d, C, H, W]
        x = x.reshape(bs, d, 256, h, w)  # [bs, d, C, h, w]
        x = x.permute(0, 2, 3, 4, 1)  # [bs, C, h, w, d]

        return x

class PatchEmbed(nn.Module):
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.permute(0, 2, 3, 1)

class Adapter(nn.Module):
    def __init__(self, D_features, hidden_feature=32, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = hidden_feature
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
