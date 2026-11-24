import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill
from calibnet.model.fade import FrequencyAdaptiveDepthEnhancement

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

CALIBNET_DECODER_REGISTRY = Registry("CalibNet Decoder")
CALIBNET_DECODER_REGISTRY.__doc__ = "registry for CalibNet decoder"

class CondConv2d(nn.Module):
    """轻量条件卷积（CondConv简化版）：为每个样本动态融合多个静态卷积核"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 groups=1, num_experts=4, reduction=8, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_experts = num_experts
        self.use_bias = bias

        # 多个静态卷积核（参数可学习）
        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        # 路由网络：为每个样本生成融合权重
        self.route = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, num_experts),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for w in self.weight:
            nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        for m in self.route.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        route_weight = self.route(x)  # [B, num_experts]
        # 对 batch 维平均，得到全局权重 [num_experts]
        global_weight = route_weight.mean(dim=0)
        # 混合专家核
        weight = (global_weight[:, None, None, None, None] * self.weight).sum(dim=0)
        bias = (global_weight[:, None] * self.bias).sum(dim=0) if self.bias is not None else None
        out = F.conv2d(x, weight, bias, self.stride, self.padding, groups=self.groups)
        return out
def _make_stack_3x3_convs(num_convs, in_channels, out_channels):
    convs = []
    for _ in range(num_convs):
        convs.append(
            Conv2d(in_channels, out_channels, 3, padding=1))
        convs.append(nn.ReLU(True))
        in_channels = out_channels
    return nn.Sequential(*convs)
@CALIBNET_DECODER_REGISTRY.register()
class CalibNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.MODEL.CALIBNET.DECODER.INPUT_CHANNELS
        in_channels = input_dim + 2  # +2 for coordinates
        self.scale_factor = cfg.MODEL.CALIBNET.DECODER.SCALE_FACTOR

        # 使用 FADE 替代 DSA
        self.fade = FrequencyAdaptiveDepthEnhancement(channels=input_dim)

        self.inst_branch = DynamicInteractiveKernel(cfg, in_channels)

        # 删除原有的 self.cosine_similarity, self.rgb_input, self.depth_input

        # 空权重初始化（各模块自己初始化）
        self._init_weights()

    def _init_weights(self):
        # 所有子模块已自行初始化，此处留空
        pass

    @torch.no_grad()
    def compute_coordinates(self, x):
        h, w = x.size(2), x.size(3)
        y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
        x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
        y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
        y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
        x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
        locations = torch.cat([x_loc, y_loc], 1)
        return locations.to(x)

    def forward(self, features, features_depth, mask_features):
        coord_features = self.compute_coordinates(features)

        # 使用 FADE 校准深度特征（替代 DSA）
        features_depth = self.fade(features, features_depth)

        features = torch.cat([features, coord_features], dim=1)
        features_depth = torch.cat([features_depth, coord_features], dim=1)
        pred_scores, pred_kernel, pred_objectness = self.inst_branch(features, features_depth)

        N = pred_kernel.shape[1]
        B, C, H, W = mask_features.shape
        pred_masks = torch.bmm(pred_kernel, mask_features.view(B, C, H * W)).view(B, N, H, W)

        if self.scale_factor != 1:
            pred_masks = F.interpolate(pred_masks, scale_factor=self.scale_factor,
                                       mode='bilinear', align_corners=False)

        output = {
            "pred_logits": pred_scores,
            "pred_masks": pred_masks,
            "pred_scores": pred_objectness,
        }
        return output
# @CALIBNET_DECODER_REGISTRY.register()
# class CalibNetDecoder(nn.Module):
#
#     def __init__(self, cfg):
#         super().__init__()
#         # add 2 for coordinates
#         input_dim = cfg.MODEL.CALIBNET.DECODER.INPUT_CHANNELS
#         in_channels = cfg.MODEL.CALIBNET.DECODER.INPUT_CHANNELS + 2
#
#         self.scale_factor = cfg.MODEL.CALIBNET.DECODER.SCALE_FACTOR
#
#         # self.inst_branch = DynamicInteractiveKernel(cfg, in_channels)
#         # 使用 FADE 替换 DSA
#         self.fade = FrequencyAdaptiveDepthEnhancement(channels=input_dim)
#
#         self.inst_branch = DynamicInteractiveKernel(cfg, in_channels)
#         self._init_weights()
#         # self.cosine_similarity = nn.CosineSimilarity(dim=1)
#         # self.rgb_input = nn.Sequential(nn.Conv2d(input_dim, 1, 1))
#         # self.depth_input = nn.Sequential(nn.Conv2d(input_dim, 1, 1))
#         # self._init_weights()
#
#     def _init_weights(self):
#         # for modules in [self.rgb_input, self.depth_input]:
#         #     for l in modules.modules():
#         #         if isinstance(l, nn.Conv2d):
#         #             nn.init.kaiming_normal_(l.weight)
#         #             if l.bias is not None:
#         #                 nn.init.constant_(l.bias, 0)
#         pass
#
#     @torch.no_grad()
#     def compute_coordinates(self, x):
#         h, w = x.size(2), x.size(3)
#         y_loc = -1.0 + 2.0 * torch.arange(h, device=x.device) / (h - 1)
#         x_loc = -1.0 + 2.0 * torch.arange(w, device=x.device) / (w - 1)
#         y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
#         y_loc = y_loc.expand([x.shape[0], 1, -1, -1])
#         x_loc = x_loc.expand([x.shape[0], 1, -1, -1])
#         locations = torch.cat([x_loc, y_loc], 1)
#         return locations.to(x)
#
#     def compute_depth_similarity_assessment(self, rgb_feat, depth_feat):
#         bs,c,h,w = rgb_feat.shape
#         rgb_feat_cosine = self.rgb_input(rgb_feat).view(bs,-1)
#         depth_feat_consine = self.depth_input(depth_feat).view(bs,-1)
#         cosine_sim = self.cosine_similarity(rgb_feat_cosine, depth_feat_consine)
#         cosine_sim = (1-cosine_sim) / 2.0
#         cosine_sim = 1-cosine_sim
#         return cosine_sim
#
#     def forward(self, features, features_depth, mask_features):
#         coord_features = self.compute_coordinates(features)
#
#         # 使用 FADE 校准深度特征
#         features_depth = self.fade(features, features_depth)
#
#         features = torch.cat([features, coord_features], dim=1)
#         features_depth = torch.cat([features_depth, coord_features], dim=1)
#         pred_scores, pred_kernel, pred_objectness = self.inst_branch(features, features_depth)
#         # compute cosine simiarity
#         cosine_sim = self.compute_depth_similarity_assessment(features, features_depth)
#         bs = features.shape[0]
#         features_depth = cosine_sim.view(bs,1,1,1) * features_depth
#
#         features = torch.cat([features, coord_features], dim=1)
#         features_depth = torch.cat([features_depth, coord_features], dim=1)
#         pred_scores, pred_kernel, pred_objectness = self.inst_branch(features, features_depth)
#
#         N = pred_kernel.shape[1]
#         # mask_features: BxCxHxW
#         B, C, H, W = mask_features.shape
#         pred_masks = torch.bmm(pred_kernel, mask_features.view(
#             B, C, H * W)).view(B, N, H, W)
#
#         if self.scale_factor != 1:
#             pred_masks = F.interpolate(
#                 pred_masks, scale_factor=self.scale_factor,
#                 mode='bilinear', align_corners=False)
#
#         output = {
#             "pred_logits": pred_scores,
#             "pred_masks": pred_masks,
#             "pred_scores": pred_objectness,
#         }
#
#         return output
class DynamicInteractiveKernel(nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        dim = cfg.MODEL.CALIBNET.DECODER.CONV_DIM
        num_kernels = cfg.MODEL.CALIBNET.DECODER.NUM_KERNELS
        kernel_dim = cfg.MODEL.CALIBNET.FUSION_DECODER.MASK_DIM
        self.num_groups = 4
        self.num_classes = cfg.MODEL.CALIBNET.NUM_CLASSES

        # 使用共享动态卷积核（全局权重共享，但样本间不同）
        self.inst_conv = CondConv2d(in_channels, dim, 3, padding=1, num_experts=4, reduction=8)
        self.inst_conv_depth = CondConv2d(in_channels, dim, 3, padding=1, num_experts=4, reduction=8)

        expand_dim = dim * self.num_groups
        self.kernel_conv = CondConv2d(dim, num_kernels * self.num_groups, 3, padding=1,
                                       groups=self.num_groups, num_experts=4, reduction=8)
        self.kernel_conv_depth = CondConv2d(dim, num_kernels * self.num_groups, 3, padding=1,
                                             groups=self.num_groups, num_experts=4, reduction=8)

        self.fc = nn.Linear(expand_dim, expand_dim)
        self.squeeze = nn.Linear(2*num_kernels, num_kernels)
        self.score_linear = nn.Linear(expand_dim, self.num_classes)
        self.mask_linear = nn.Linear(expand_dim, kernel_dim)
        self.objectness = nn.Linear(expand_dim, 1)

        self.prior_prob = 0.01
        self._init_weights()

    def _init_weights(self):
        c2_xavier_fill(self.fc)
        c2_xavier_fill(self.squeeze)
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for module in [self.score_linear]:
            init.constant_(module.bias, bias_value)
        init.normal_(self.score_linear.weight, std=0.01)
        init.normal_(self.mask_linear.weight, std=0.01)
        init.constant_(self.mask_linear.bias, 0.0)

    def forward(self, features_rgb, features_depth):
        # 动态卷积提取特征
        f_rgb = self.inst_conv(features_rgb)          # [B, dim, H, W]
        f_depth = self.inst_conv_depth(features_depth)

        # 跨模态交互（互补门控）
        gate = torch.sigmoid(f_rgb + f_depth)
        f_rgb_mixed = f_rgb * gate
        f_depth_mixed = f_depth * (1 - gate)

        # 生成实例感知注意力图
        kernel_feat = self.kernel_conv(f_rgb_mixed).sigmoid()
        kernel_feat_depth = self.kernel_conv_depth(f_depth_mixed).sigmoid()

        B, N = kernel_feat.shape[:2]   # N = num_kernels * groups
        C = f_rgb.size(1)              # 通道数 dim

        # 空间归一化
        kernel_feat = kernel_feat.view(B, N, -1)
        normalizer = kernel_feat.sum(-1).clamp(min=1e-6)
        kernel_feat = kernel_feat / normalizer[:, :, None]

        kernel_feat_depth = kernel_feat_depth.view(B, N, -1)
        normalizer = kernel_feat_depth.sum(-1).clamp(min=1e-6)
        kernel_feat_depth = kernel_feat_depth / normalizer[:, :, None]

        kernel_feat = torch.cat([kernel_feat, kernel_feat_depth], dim=1)

        # 实例嵌入
        inst_features = torch.bmm(kernel_feat, f_rgb.view(B, C, -1).permute(0, 2, 1))
        inst_features = inst_features.reshape(
            B, self.num_groups, 2 * N // self.num_groups, -1).transpose(1, 2).reshape(B, 2 * N // self.num_groups, -1)

        inst_features = self.squeeze(inst_features.transpose(1,2))
        inst_features = F.relu(inst_features).transpose(1,2)
        inst_features = F.relu_(self.fc(inst_features))

        pred_scores = self.score_linear(inst_features)
        pred_kernel = self.mask_linear(inst_features)
        pred_objectness = self.objectness(inst_features)
        return pred_scores, pred_kernel, pred_objectness
# class DynamicInteractiveKernel(nn.Module):
#
#     def __init__(self, cfg, in_channels):
#         super().__init__()
#         dim = cfg.MODEL.CALIBNET.DECODER.CONV_DIM
#         num_convs = 1
#         num_kernels = cfg.MODEL.CALIBNET.DECODER.NUM_KERNELS
#         kernel_dim = cfg.MODEL.CALIBNET.FUSION_DECODER.MASK_DIM
#         self.num_groups = 4
#         self.num_classes = cfg.MODEL.CALIBNET.NUM_CLASSES
#
#         self.inst_convs = _make_stack_3x3_convs(num_convs, in_channels, dim)
#         self.inst_convs_depth = _make_stack_3x3_convs(num_convs, in_channels, dim)
#
#         expand_dim = dim * self.num_groups
#         self.kernel_conv = nn.Conv2d(
#             dim, num_kernels * self.num_groups, 3, padding=1, groups=self.num_groups)
#         self.kernel_conv_depth = nn.Conv2d(
#             dim, num_kernels * self.num_groups, 3, padding=1, groups=self.num_groups)
#         # outputs
#         self.fc = nn.Linear(expand_dim, expand_dim)
#         self.squeeze = nn.Linear(2*num_kernels, num_kernels)
#         self.score_linear = nn.Linear(expand_dim, self.num_classes)
#         self.mask_linear = nn.Linear(expand_dim, kernel_dim)
#         self.objectness = nn.Linear(expand_dim, 1)
#
#         self.prior_prob = 0.01
#
#         self._init_weights()
#
#     def _init_weights(self):
#         for m in self.inst_convs.modules():
#             if isinstance(m, nn.Conv2d):
#                 c2_msra_fill(m)
#         for m in self.inst_convs_depth.modules():
#             if isinstance(m, nn.Conv2d):
#                 c2_msra_fill(m)
#         bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
#         for module in [self.kernel_conv, self.kernel_conv_depth, self.score_linear]:
#             init.constant_(module.bias, bias_value)
#         init.normal_(self.kernel_conv.weight, std=0.01)
#         init.normal_(self.kernel_conv_depth.weight, std=0.01)
#         init.normal_(self.score_linear.weight, std=0.01)
#
#         init.normal_(self.mask_linear.weight, std=0.01)
#         init.constant_(self.mask_linear.bias, 0.0)
#         c2_xavier_fill(self.fc)
#         c2_xavier_fill(self.squeeze)
#
#     def forward(self, features_rgb, features_depth):
#         features_rgb = self.inst_convs(features_rgb)
#         features_depth = self.inst_convs_depth(features_depth)
#         kernel_feat = self.kernel_conv(features_rgb).sigmoid()
#         kernel_feat_depth = self.kernel_conv_depth(features_depth).sigmoid()
#
#         B, N = kernel_feat.shape[:2]
#         C = features_rgb.size(1)
#         kernel_feat = kernel_feat.view(B, N, -1)
#         normalizer = kernel_feat.sum(-1).clamp(min=1e-6)
#         kernel_feat = kernel_feat / normalizer[:, :, None]
#         kernel_feat_depth = kernel_feat_depth.view(B, N, -1)
#         normalizer = kernel_feat_depth.sum(-1).clamp(min=1e-6)
#         kernel_feat_depth = kernel_feat_depth / normalizer[:, :, None]
#
#         kernel_feat = torch.cat([kernel_feat, kernel_feat_depth], dim=1)
#
#         inst_features = torch.bmm(
#             kernel_feat, features_rgb.view(B, C, -1).permute(0, 2, 1))
#
#         inst_features = inst_features.reshape(
#             B, self.num_groups, 2 * N // self.num_groups, -1).transpose(1, 2).reshape(B, 2 * N // self.num_groups, -1)
#
#         inst_features = self.squeeze(inst_features.transpose(1,2))
#         inst_features = F.relu(inst_features).transpose(1,2)
#         inst_features = F.relu_(self.fc(inst_features))
#
#         pred_scores = self.score_linear(inst_features)
#         pred_kernel = self.mask_linear(inst_features)
#         pred_objectness = self.objectness(inst_features)
#         return pred_scores, pred_kernel, pred_objectness

def build_kernel_branch(cfg):
    name = cfg.MODEL.CALIBNET.DECODER.NAME
    return CALIBNET_DECODER_REGISTRY.get(name)(cfg)