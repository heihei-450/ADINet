import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSourceQualityEstimator(nn.Module):
    """多源深度质量评估器"""
    def __init__(self, in_channels=256, reduction=16):
        super().__init__()
        # 低层级线索 (结构一致性)
        self.low_level = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 中层级线索 (全局重要性)
        self.mid_level = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        # 高层级线索 (跨模态余弦相似度)
        self.cosine = nn.CosineSimilarity(dim=1)
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb_feat, depth_feat):
        # 低层级质量分数
        low_score = self.low_level(depth_feat)
        # 中层级质量分数
        mid_score = self.mid_level(rgb_feat)
        # 高层级质量分数
        b, c, h, w = rgb_feat.shape
        rgb_pool = F.adaptive_avg_pool2d(rgb_feat, (1, 1)).view(b, -1)
        depth_pool = F.adaptive_avg_pool2d(depth_feat, (1, 1)).view(b, -1)
        high_score = self.cosine(rgb_pool, depth_pool).abs()
        # 融合质量图
        quality_map = self.fusion(low_score)
        return quality_map, mid_score, high_score


class HybridGateModulation(nn.Module):
    """混合门控调制模块"""
    def __init__(self, in_channels=256):
        super().__init__()
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, depth_feat, quality_map, high_score, rgb_feat):
        # 全局门控
        global_gate_val = self.global_gate(depth_feat)
        global_gate_val = global_gate_val * high_score.view(-1, 1)
        # 空间门控 + 质量图调制
        spatial_gate_val = self.spatial_gate(depth_feat)
        spatial_gate_val = spatial_gate_val * torch.sigmoid(quality_map)
        # 跨模态注意力引导
        query = self.query_conv(rgb_feat)
        key = depth_feat
        attention = torch.sigmoid(torch.sum(query * key, dim=1, keepdim=True))
        # 综合调制
        calibrated_depth = depth_feat * global_gate_val.unsqueeze(-1).unsqueeze(-1)
        calibrated_depth = calibrated_depth * spatial_gate_val
        calibrated_depth = calibrated_depth + depth_feat * attention
        return calibrated_depth


class DepthQualityAdaptiveCalibration(nn.Module):
    """
    深度质量自适应校准 (DQAC) 模块
    用于替换原始 CalibNet 中的 DSA 模块
    """
    def __init__(self, in_channels=256, reduction=16):
        super().__init__()
        self.quality_estimator = MultiSourceQualityEstimator(in_channels, reduction)
        self.gate_modulation = HybridGateModulation(in_channels)
        # 边缘增强分支
        self.edge_enhance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        # 最终细化卷积
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1)
        )

    def forward(self, rgb_feat, depth_feat):
        # 1. 多源质量评估
        quality_map, mid_score, high_score = self.quality_estimator(rgb_feat, depth_feat)
        # 2. 混合门控调制
        calibrated_depth = self.gate_modulation(depth_feat, quality_map, high_score, rgb_feat)
        # 3. 边缘信息引导
        edge_weight = self.edge_enhance(calibrated_depth)
        calibrated_depth = calibrated_depth + calibrated_depth * edge_weight
        # 4. 最终细化
        calibrated_depth = self.refine(calibrated_depth)
        return calibrated_depth