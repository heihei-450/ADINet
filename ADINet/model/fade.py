import torch
import torch.nn as nn

class FrequencyAdaptiveDepthEnhancement(nn.Module):
    """
    频域自适应深度增强模块 (FADE)
    替代原始 DSA，利用 RGB 特征引导深度特征在频域进行自适应滤波
    """
    def __init__(self, channels=256, reduction=16, sigma=0.3):
        super().__init__()
        self.sigma = sigma  # 高斯低通滤波器的标准差
        # 生成通道级频域偏置（每个通道学习一个标量，控制低通/高通倾向）
        self.freq_bias = nn.Parameter(torch.zeros(channels))
        # 利用 RGB 特征生成实例自适应的全局掩码
        self.mask_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mask_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, rgb_feat, depth_feat):
        """
        Args:
            rgb_feat: (B, C, H, W)  RGB特征（引导信息）
            depth_feat: (B, C, H, W) 深度特征（待增强）
        Returns:
            enhanced_depth: (B, C, H, W)
        """
        B, C, H, W = depth_feat.shape
        # 1. 利用 RGB 特征生成全局掩码（控制滤波强度）
        mask = self.mask_net(rgb_feat)  # (B, C, 1, 1)

        # 2. 转换到频域（使用 rfft2，只保存一半频率，节省计算）
        depth_fft = torch.fft.rfft2(depth_feat, norm='ortho')  # (B, C, H, W//2+1)

        # 3. 构造频域滤波器（基于归一化频率半径的高斯低通）
        freq_y = torch.fft.fftfreq(H, device=depth_feat.device).abs()  # (H,)
        freq_x = torch.fft.rfftfreq(W, device=depth_feat.device).abs() # (W//2+1,)
        radius = torch.sqrt(freq_y[:, None]**2 + freq_x[None, :]**2)   # (H, W')
        lowpass = torch.exp(-radius**2 / (2 * self.sigma**2))          # (H, W')

        # 通道自适应偏置：每个通道学习一个偏移量，使滤波器可以从低通向高通偏移
        channel_weight = torch.sigmoid(self.freq_bias)  # (C,)
        filter = lowpass[None, None, :, :] + channel_weight.view(1, C, 1, 1) * (1 - lowpass[None, None, :, :])
        filter = filter * mask.view(B, C, 1, 1)  # 实例自适应调制

        # 4. 频域滤波
        depth_fft = depth_fft * filter

        # 5. 逆变换回空间域
        enhanced_depth = torch.fft.irfft2(depth_fft, s=(H, W), norm='ortho')
        return enhanced_depth