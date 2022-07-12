import torch
from torch import nn
from torch.nn import functional as F

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ration):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(inp * expand_ration))
        self.use_res_connect = stride == 1 and inp == oup

        layers: list[nn.Module] = []

        if expand_ration != 1:
            # 升维
            layers.append(nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)))
        layers.extend([
            # depthwise conv
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ),
            # pointwise conv
            nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, stride=1, bias=False),
                nn.BatchNorm2d(oup)
            )
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        input_channel = 32
        last_channel = 1280

        # building first layer
        features: list[nn.Module] = [nn.Sequential(
            nn.Conv2d(3, input_channel, 3, stride=2, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        block = InvertedResidual

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, c, stride, expand_ration=t))
                input_channel = c

        # building last several layers
        features.append(nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(last_channel),
            nn.ReLU6(inplace=True)
        ))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = F.adaptive_avg_pool2d(input=x, output_size=(1, 1))  # 自适应平均池化
        x = torch.flatten(x, start_dim=1)  # 展平
        x = self.classifier(x)
        return x
