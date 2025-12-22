import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Block
    채널 간의 상호 의존성을 모델링하여 중요한 피처를 강조합니다.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNeXtSEBasicBlock(nn.Module):
    """
    Grouped Convolution + SE Block이 적용된 BasicBlock
    ResNet-18/34 계열의 깊이에 적합합니다.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, reduction=16):
        super(ResNeXtSEBasicBlock, self).__init__()
        
        # ResNeXt의 width 계산 로직 (BasicBlock에서는 planes를 그대로 사용하는 경우가 많으나, 확장성을 위해 남겨둠)
        # width = int(planes * (base_width / 64.)) * groups 
        # 여기서는 ResNet-18 구조를 따르므로 planes를 그대로 사용하되 groups 옵션을 활성화합니다.
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # SE Block 추가
        self.se = SELayer(planes, reduction)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # SE 적용 (Residual 연결 전)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNeXtSE18_Small(nn.Module):
    def __init__(self, num_classes=100, groups=32):
        super(ResNeXtSE18_Small, self).__init__()
        
        self.inplanes = 64
        self.groups = groups
        
        # [수정됨] 64x64 입력을 위해 7x7/stride2 대신 3x3/stride1 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # [수정됨] MaxPool 제거 (정보 손실 방지)
        
        # Layers
        self.layer1 = self._make_layer(ResNeXtSEBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNeXtSEBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNeXtSEBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNeXtSEBasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNeXtSEBasicBlock.expansion, num_classes)

        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem (수정됨)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # MaxPool 없음

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
