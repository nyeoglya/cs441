# densenet_se.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation for channel attention.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        assert reduction >= 1
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc2(self.relu(self.fc1(y)))
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y


class _DenseLayer(nn.Module):
    """
    DenseNet layer: BN-ReLU-1x1Conv -> BN-ReLU-3x3Conv -> (optional SE) -> (optional Dropout),
    then concat with input.
    """
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()
        inter_features = bn_size * growth_rate

        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, inter_features, kernel_size=1, stride=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_features)
        self.conv2 = nn.Conv2d(inter_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.se = SEModule(growth_rate, reduction=se_reduction) if use_se else nn.Identity()
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out = self.se(out)

        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()
        layers = []
        features = num_input_features
        for _ in range(num_layers):
            layers.append(
                _DenseLayer(
                    num_input_features=features,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    use_se=use_se,
                    se_reduction=se_reduction,
                )
            )
            features += growth_rate

        self.block = nn.Sequential(*layers)
        self.num_output_features = features

    def forward(self, x):
        return self.block(x)


class _Transition(nn.Module):
    """
    Transition: BN-ReLU-1x1Conv -> 2x2 AvgPool (stride 2)
    """
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x), inplace=True))
        x = self.pool(x)
        return x


class DenseNet121SE(nn.Module):
    """
    DenseNet-121 + optional SE in each DenseLayer's growth output.

    Defaults are tuned for small images (e.g., 64x64):
      - CIFAR-style 3x3 stem (stride=1) instead of ImageNet 7x7/stride=2
      - block_config = (6, 12, 24, 16)

    Constraints-friendly:
      - Parameter-efficient (DenseNet121 ~ 8M) + SE overhead is small.
    """
    def __init__(
        self,
        num_classes: int = 15,
        growth_rate: int = 32,
        block_config=(6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        compression: float = 0.5,
        classifier_dropout: float = 0.2,
        use_se: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()
        assert 0.0 < compression <= 1.0

        # Small-image stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
        )

        # Dense blocks + transitions
        features = num_init_features
        blocks = []
        for i, num_layers in enumerate(block_config):
            db = _DenseBlock(
                num_layers=num_layers,
                num_input_features=features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                use_se=use_se,
                se_reduction=se_reduction,
            )
            blocks.append(db)
            features = db.num_output_features

            if i != len(block_config) - 1:
                out_features = int(features * compression)
                blocks.append(_Transition(num_input_features=features, num_output_features=out_features))
                features = out_features

        self.features = nn.Sequential(*blocks)

        # Final BN + pooling + classifier
        self.final_bn = nn.BatchNorm2d(features)
        self.classifier_dropout = nn.Dropout(p=float(classifier_dropout)) if classifier_dropout > 0 else nn.Identity()
        self.fc = nn.Linear(features, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = F.relu(self.final_bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier_dropout(x)
        x = self.fc(x)
        return x
