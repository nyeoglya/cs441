# allconv.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class AllCNN_C(nn.Module):
    """
    All-CNN-C style network (Striving for Simplicity: The All Convolutional Net, 2015).

    CIFAR Model C 핵심 아이디어:
      - maxpool 제거, stride=2 conv로 downsample
      - FC 대신 1x1 conv로 class logits 만들고 global average

    논문 Table 1의 Model C 구성을 따라가되,
    입력이 64x64여도 작동하도록 global average는 adaptive로 처리.

    Recommended defaults for your setting:
      - num_classes=15
      - dropout=0.5
      - use_bn=True (논문 원형은 BN이 필수는 아니지만, 15-epoch 제한에서는 BN이 수렴/안정에 도움)
    """
    def __init__(self, num_classes=15, dropout=0.5, use_bn=True, width1=96, width2=192):
        super().__init__()
        self.use_bn = use_bn
        self.dropout_p = float(dropout)

        # Block 1 (논문 Model C 시작부: 3x3 conv 96, 3x3 conv 96, 3x3 conv 96)
        self.b1_1 = ConvBNReLU(3,      width1, k=3, stride=1, padding=1, use_bn=use_bn)
        self.b1_2 = ConvBNReLU(width1, width1, k=3, stride=1, padding=1, use_bn=use_bn)
        # downsample by stride=2 conv (pooling 대체) :contentReference[oaicite:1]{index=1}
        self.b1_3 = ConvBNReLU(width1, width1, k=3, stride=2, padding=1, use_bn=use_bn)

        # Block 2 (3x3 conv 192 x3, 마지막은 stride=2로 downsample)
        self.b2_1 = ConvBNReLU(width1, width2, k=3, stride=1, padding=1, use_bn=use_bn)
        self.b2_2 = ConvBNReLU(width2, width2, k=3, stride=1, padding=1, use_bn=use_bn)
        self.b2_3 = ConvBNReLU(width2, width2, k=3, stride=2, padding=1, use_bn=use_bn)

        # Block 3 (3x3 conv 192, 1x1 conv 192, 1x1 conv num_classes)
        self.b3_1 = ConvBNReLU(width2, width2, k=3, stride=1, padding=1, use_bn=use_bn)
        self.b3_2 = ConvBNReLU(width2, width2, k=1, stride=1, padding=0, use_bn=use_bn)

        # 마지막 classifier는 BN/ReLU 없이 logits만 (논문은 1x1 conv로 class map 만든 뒤 global average) :contentReference[oaicite:2]{index=2}
        self.classifier = nn.Conv2d(width2, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Block 1
        x = self.b1_1(x)
        x = self.b1_2(x)
        x = self.b1_3(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Block 2
        x = self.b2_1(x)
        x = self.b2_2(x)
        x = self.b2_3(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Block 3
        x = self.b3_1(x)
        x = self.b3_2(x)

        # Class map -> global average -> logits
        x = self.classifier(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
