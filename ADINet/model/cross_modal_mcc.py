import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalMCC(nn.Module):
    """多模态互补模块 (MCC)，用于替换CalibNet中的WSF。"""

    def __init__(self, in_channels, reduction=16):
        super(CrossModalMCC, self).__init__()
        # 1x1 卷积用于通道对齐和降维
        self.conv_rgb = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.conv_depth = nn.Conv2d(in_channels, in_channels // reduction, 1)

        # 用于生成注意力权重的小型卷积网络
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True)
        )

        # 计算互注意力时的激活函数
        self.sigmoid = nn.Sigmoid()

        # 恢复通道数
        self.conv_fuse = nn.Conv2d(in_channels // reduction, in_channels, 1)

        # 最终特征细化
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 1)
        )

    def forward(self, rgb, depth):
        # 1. 降维并生成基础特征
        rgb_f = self.conv_rgb(rgb)
        depth_f = self.conv_depth(depth)

        # 2. 生成各自的注意力权重 (信息提炼)
        attn_rgb = self.sigmoid(self.attn_conv(rgb_f))
        attn_depth = self.sigmoid(self.attn_conv(depth_f))

        # 3. 交换注意力权重，实现信息互补 (注意力交换)
        refined_rgb = rgb_f * attn_depth  # RGB特征接收Depth的关注点
        refined_depth = depth_f * attn_rgb  # Depth特征接收RGB的关注点

        # 4. 融合互补后的特征
        fused = refined_rgb + refined_depth
        fused = self.conv_fuse(fused)

        # 5. 最终特征细化 (轻量级残差)
        out = self.final_conv(fused) + rgb

        return out