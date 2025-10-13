import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusterAwareUpsampling(nn.Module):
    """聚类感知的上采样模块"""
    def __init__(self, in_channels, out_channels, scale_factor, num_clusters):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + num_clusters, out_channels, 3, padding=1),  # 直接拼接聚类特征
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x, cluster_assign):
        # 上采样聚类特征并拼接
        cluster_feat = F.interpolate(
            cluster_assign, 
            scale_factor=self.scale_factor, 
            mode='trilinear'
        )
        x = F.interpolate(
            x, 
            size=cluster_feat.shape[2:],  # 使用cluster_feat的空间尺寸
            mode='trilinear'
        )
        x = torch.cat([x, cluster_feat], dim=1)  # 通道维度拼接
        x = self.conv(x)
        return x

class MaskDecoder(nn.Module):
    """优化后的Mask Decoder，适配动态聚类编码器"""
    def __init__(self, mlahead_channels=256, num_classes=2, num_clusters=64, output_size=128):
        super().__init__()
        self.num_clusters = num_clusters
        self.mlahead_channels = mlahead_channels
        self.output_size = output_size

        # 聚类特征提取器（与编码器一致）
        self.cluster_projection = nn.Sequential(
            nn.Conv3d(mlahead_channels, num_clusters, 1),
            nn.Softmax(dim=1))
        
        # 多尺度特征提取分支（加入聚类感知）
        # 传入num_clusters参数确保通道匹配
        self.branch0 = ClusterAwareUpsampling(mlahead_channels, mlahead_channels//2, 
                                           scale_factor=1, num_clusters=num_clusters)
        self.branch1 = ClusterAwareUpsampling(mlahead_channels, mlahead_channels//2, 
                                           scale_factor=2, num_clusters=num_clusters)
        self.branch2 = ClusterAwareUpsampling(mlahead_channels, mlahead_channels//2, 
                                           scale_factor=4, num_clusters=num_clusters)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv3d(mlahead_channels*3//2, mlahead_channels, 3, padding=1),
            nn.InstanceNorm3d(mlahead_channels),
            nn.ReLU(inplace=True))
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv3d(mlahead_channels, mlahead_channels//2, 1),
            nn.InstanceNorm3d(mlahead_channels//2),
            nn.ReLU(inplace=True),
            nn.Upsample(size=output_size, mode='trilinear', align_corners=True),
            nn.Conv3d(mlahead_channels//2, num_classes, 1))

        self.morph_kernel = torch.ones(3, 3, 3)  # 3D形态学核

    def forward(self, input):
        # input: [B, C, H, W, D] 来自编码器的特征
        
        # 提取聚类分配概率
        cluster_assign = self.cluster_projection(input)  # [B, K, H, W, D]
        
        # 多尺度特征提取
        x0 = self.branch0(input, cluster_assign)
        x1 = self.branch1(input, cluster_assign)
        x2 = self.branch2(input, cluster_assign)
        
        # 调整到相同尺寸
        target_size = (input.size(2)*4, input.size(3)*4, input.size(4)*4)
        x0 = F.interpolate(x0, size=target_size, mode='trilinear', align_corners=True)
        x1 = F.interpolate(x1, size=target_size, mode='trilinear', align_corners=True)
        x2 = F.interpolate(x2, size=target_size, mode='trilinear', align_corners=True)
        
        # 特征融合
        x = torch.cat([x0, x1, x2], dim=1)
        x = self.fusion(x)
        
        # 最终输出
        x = self.final(x)

        #新增置信度
        confidence_map = torch.sigmoid(x).max(dim=1, keepdim=True)[0]

        # 添加形态学后处理
        if self.training:
            return x, confidence_map  # 训练时不启用
        else:
            # 二值化并应用闭运算
            binary_mask = (torch.sigmoid(x) > 0.5)
            binary_mask = binary_mask.float()
            closed_mask = self._morphological_close(binary_mask)
            x = closed_mask * x  # 保留原始概率值
            return x, confidence_map
            
    def _morphological_close(self, x):
        """3D闭运算：先膨胀后腐蚀（支持多通道输入）"""
        # 确保kernel形状正确 [out_channels, in_channels, ...]
        kernel = self.morph_kernel.to(x.device)[None,None].repeat(x.shape[1], 1, 1, 1, 1)  # [2,1,3,3,3]
    
        # 膨胀
        padded = F.pad(x, (1,1,1,1,1,1), mode='constant', value=0)
        dilated = F.conv3d(padded, kernel, groups=x.shape[1], padding=0) > 0
    
        # 腐蚀
        padded = F.pad(dilated.float(), (1,1,1,1,1,1), mode='constant', value=1)
        eroded = F.conv3d(padded, kernel, groups=x.shape[1], padding=0) == 27
    
        return eroded.float()
    


import numpy as np

# 不确定性采样器
class UncertaintySampler:
    """不确定性采样器，用于选择需要交互的点"""
    
    def __init__(self, num_points=10, strategy='entropy'):
        self.num_points = num_points
        self.strategy = strategy
    
    def sample_points(self, logits, seg_mask=None):
        """
        根据不确定性采样点
        Args:
            logits: 模型预测的logits [B, C, D, H, W]
            seg_mask: 可选的分割掩码，用于排除背景区域
        Returns:
            points: 采样的点坐标 [B, num_points, 3]
            uncertainties: 不确定性值
        """
        # 检查输入有效性
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            print("Warning: Invalid logits detected, returning empty points")
            return torch.empty(0, device=logits.device), torch.empty(0, device=logits.device)
        
        probs = torch.softmax(logits, dim=1)
        
        # 检查概率有效性
        if torch.any(torch.isnan(probs)) or torch.any(probs < 0) or torch.sum(probs) <= 0:
            # 返回空点或默认点
            return torch.empty(0), torch.empty(0)
        
        batch_size = probs.shape[0]
        
        if self.strategy == 'entropy':
            uncertainties = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # 熵不确定性
        elif self.strategy == 'margin':
            probs_sorted, _ = torch.sort(probs, dim=1, descending=True)
            uncertainties = 1 - (probs_sorted[:, 0] - probs_sorted[:, 1])  # 边界不确定性
        elif self.strategy == 'confidence':
            uncertainties = 1 - probs.max(dim=1)[0]  # 置信度不确定性
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # 如果有分割掩码，只在目标区域采样
        if seg_mask is not None:
            uncertainties = uncertainties * seg_mask.float()
        
        sampled_points = []
        for b in range(batch_size):
            # 将不确定性展平并采样
            uncert_flat = uncertainties[b].view(-1)
            # 确保不确定性值为正
            uncert_flat = torch.clamp(uncert_flat, min=1e-10)
            indices = torch.multinomial(uncert_flat, self.num_points, replacement=False)
            
            # 将索引转换为3D坐标
            coords = []
            for idx in indices:
                d = idx // (uncertainties.shape[2] * uncertainties.shape[3])
                h = (idx % (uncertainties.shape[2] * uncertainties.shape[3])) // uncertainties.shape[3]
                w = (idx % (uncertainties.shape[2] * uncertainties.shape[3])) % uncertainties.shape[3]
                coords.append([d, h, w])
            
            # 在采样点之前添加概率有效性检查
            if torch.any(probs <= 0) or torch.sum(probs) <= 0:
                # 如果概率无效，跳过这个样本或使用默认采样
                print(f"Warning: Invalid probability distribution in sample {b}, sum: {torch.sum(probs)}")
                continue  # 或者使用其他采样策略
            
            sampled_points.append(torch.tensor(coords, device=logits.device))
        
        return torch.stack(sampled_points), uncertainties
    
    def get_boundary_points(self, pred_mask, num_points=10):
        """
        从预测掩码的边界区域采样点
        """
        batch_size = pred_mask.shape[0]
        
        sampled_points = []
        for b in range(batch_size):
            mask = pred_mask[b, 0].cpu().numpy()  # [D, H, W]
            
            # 使用形态学梯度检测边界
            from scipy import ndimage
            structure = ndimage.generate_binary_structure(3, 2)
            eroded = ndimage.binary_erosion(mask, structure)
            dilated = ndimage.binary_dilation(mask, structure)
            boundaries = dilated ^ eroded
            
            # 获取边界点坐标
            boundary_coords = np.argwhere(boundaries)
            
            if len(boundary_coords) > 0:
                # 随机采样边界点
                if len(boundary_coords) > num_points:
                    indices = np.random.choice(len(boundary_coords), num_points, replace=False)
                    points = boundary_coords[indices]
                else:
                    points = boundary_coords
                
                sampled_points.append(torch.tensor(points, device=pred_mask.device))
            else:
                # 如果没有边界点，使用随机采样
                coords = []
                for _ in range(num_points):
                    d = np.random.randint(0, pred_mask.shape[2])
                    h = np.random.randint(0, pred_mask.shape[3])
                    w = np.random.randint(0, pred_mask.shape[4])
                    coords.append([d, h, w])
                sampled_points.append(torch.tensor(coords, device=pred_mask.device))
        
        return torch.stack(sampled_points)