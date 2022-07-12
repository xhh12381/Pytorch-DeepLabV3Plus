import torch
from torch import nn
from torch.nn import functional as F

from nets.mobilenetv2 import MobileNetV2


class AtrousConvMobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8):
        super().__init__()
        from functools import partial

        model = MobileNetV2(num_classes=1000)
        # print(*[(name, param.shape) for name, param in model.named_parameters()])
        model.load_state_dict(
            torch.load('/kaggle/working/Pytorch-DeepLabV3Plus/model_data/mobilenet_v2-b0353104.pth'),
            strict=False)

        self.features = model.features[:-1]
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                # apply()遍历 model 中所有的模块，并对每个模块应用 partial 函数
                self.features.apply(
                    # partial()用来扩展函数
                    partial(func=self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(func=self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        # 使用膨胀（空洞）卷积在保证感受野大小不变的情况下而不对特征图进行下采样
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)  # 膨胀系数
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)  # x 只会前向传播到第 4 层
        x = self.features[4:](low_level_features)
        return low_level_features, x


# -----------------------------------------#
#   ASPP 特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
# -----------------------------------------#
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))]
        # 2~4
        atrous_rates = (6, 12, 18)
        for rate in atrous_rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # 5
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        size = x.shape[-2:]
        _res = []
        for conv in self.convs[:-1]:
            _res.append(conv(x))

        x = self.convs[-1](x)
        x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        _res.append(x)

        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes, downsample_factor=16):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = AtrousConvMobileNetV2(downsample_factor=downsample_factor)
        in_channels = 320
        low_level_channels = 24

        self.aspp = ASPP(in_channels=in_channels, out_channels=256)

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.cls_conv = nn.Conv2d(256, self.num_classes, 1)

    def forward(self, x):
        x_size = x.shape[-2:]

        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.low_level_conv(low_level_features)

        low_level_features_size = low_level_features.shape[-2:]
        x = F.interpolate(x, size=low_level_features_size, mode="bilinear", align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=x_size, mode="bilinear", align_corners=True)

        return x
