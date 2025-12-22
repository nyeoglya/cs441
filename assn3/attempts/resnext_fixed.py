import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Stochastic Depth per-sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SEResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        cardinality=32,
        base_width=64,
        reduction=16,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        D = int(planes * (base_width / 64.0))
        C = cardinality
        width = D * C

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=C, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.se = SEModule(planes * self.expansion, reduction)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.drop_prob = float(drop_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out = drop_path(out, self.drop_prob, self.training)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class SEResNeXt50(nn.Module):
    """
    SE-ResNeXt (from scratch) for 64x64 classification.

    Notes:
    - No maxpool, conv1 stride=1 (better for small images).
    - layer2 stride=1 (preserve spatial detail); layer3/layer4 downsample.
    - Stochastic Depth (drop_path_rate) improves generalization without adding params.
    """
    BLOCK = SEResNeXtBottleneck
    LAYERS = [3, 4, 6, 3]  # deeper than the previous [2,3,5,2] but narrower settings are used in main.py.

    def __init__(self, num_classes=15, cardinality=32, base_width=12, drop_path_rate: float = 0.0):
        super().__init__()
        self.inplanes = 64
        self.cardinality = cardinality
        self.base_width = base_width

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        total_blocks = sum(self.LAYERS)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        self._dp_idx = 0
        self._dp_rates = dp_rates

        self.layer1 = self._make_layer(self.BLOCK, 64,  self.LAYERS[0], stride=1)
        self.layer2 = self._make_layer(self.BLOCK, 128, self.LAYERS[1], stride=1)
        self.layer3 = self._make_layer(self.BLOCK, 256, self.LAYERS[2], stride=2)
        self.layer4 = self._make_layer(self.BLOCK, 512, self.LAYERS[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(512 * self.BLOCK.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # zero-initialize the last BN in each residual branch (stabilizes training)
        for m in self.modules():
            if isinstance(m, SEResNeXtBottleneck):
                nn.init.constant_(m.bn3.weight, 0.0)

    def _next_drop_prob(self) -> float:
        if self._dp_idx >= len(self._dp_rates):
            return 0.0
        p = self._dp_rates[self._dp_idx]
        self._dp_idx += 1
        return float(p)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                cardinality=self.cardinality,
                base_width=self.base_width,
                drop_prob=self._next_drop_prob(),
            )
        ]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    downsample=None,
                    cardinality=self.cardinality,
                    base_width=self.base_width,
                    drop_prob=self._next_drop_prob(),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
