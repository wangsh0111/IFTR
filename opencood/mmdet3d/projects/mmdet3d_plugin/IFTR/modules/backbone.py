import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from efficientnet_pytorch import EfficientNet
from mmcv.runner.base_module import BaseModule
from mmdet.models.builder import BACKBONES
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor,
                              mode='bilinear', align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(  # 两个3x3卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)                                            # 对x1进行上采样
        x1 = torch.cat([x2, x1], dim=1)                             # 将x1和x2 concat 在一起
        return self.conv(x1)


@BACKBONES.register_module()
class Efficientnet(nn.Module):
    def __init__(self, out_c, downsample=8, scale="b0", frozen_block=23, frozen_neck=True):
        super(Efficientnet, self).__init__()

        num_block = {
            "b0": 16, "b1": 23, "b2": 23, "b3": 26,
            "b4": 32, "b5": 39, "b6": 45, "b7": 55
        }

        inter_outc = {
            "b0": [16, 24, 40, 112, 320],
            "b1": [16, 24, 40, 112, 320],
            "b2": [16, 24, 48, 120, 352],
            "b3": [24, 32, 48, 136, 384],
            "b4": [24, 32, 56, 160, 448],
            "b5": [24, 40, 64, 176, 512],
            "b6": [32, 40, 72, 200, 576],
            "b7": [32, 48, 80, 224, 640]
        }

        self.C = out_c
        self.downsample = downsample

        self.trunk = EfficientNet.from_pretrained(f"efficientnet-{scale}")  # 加载预训练 EfficientNet
        # self.trunk = EfficientNet.from_name(f"efficientnet-{scale}")  # 加载 EfficientNet 网络结构, 无预训练参数

        self.inter_outc = inter_outc[scale]
        self.up1 = Up(self.inter_outc[-1] + self.inter_outc[-2], 512)

        if downsample == 8:
            self.up2 = Up(512 + self.inter_outc[-3], 512)

        self.image_head = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

        if frozen_block >= 0:
            for name, param in self.trunk._swish.named_parameters():
                param.requires_grad = False
            for name, param in self.trunk._bn0.named_parameters():
                param.requires_grad = False
            frozen_block -= 1

        for block in self.trunk._blocks:
            if frozen_block < 0:
                break
            for name, param in block.named_parameters():
                param.requires_grad = False
            frozen_block -= 1

    def get_eff_features(self, x):
        endpoints = dict()
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'],
                     endpoints['reduction_4'])  # 先对endpoints[4]进行上采样，然后将 endpoints[5]和endpoints[4] concat 在一起
        if self.downsample == 8:
            x = self.up2(x, endpoints['reduction_3'])

        return x

    def forward(self, x):
        x_img = x[:, :3:, :, :]
        features = self.get_eff_features(x_img)
        x_img = self.image_head(features)

        return x_img


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss

