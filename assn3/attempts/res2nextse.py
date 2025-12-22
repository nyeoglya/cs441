import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    # SE Block (이전 코드와 동일)
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


class Res2NeXtSEBasicBlock(nn.Module):
    """
    Res2Net + ResNeXt + SE Block이 통합된 BasicBlock
    Res2Net의 Scale Dimension (scales)을 4로 설정합니다.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=4, scale=4, reduction=16):
        super(Res2NeXtSEBasicBlock, self).__init__()
        
        # ResNet BasicBlock의 첫 번째 3x3 Conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.scales = scale  # Res2Net의 스케일 차원 (채널 분할 개수)
        assert planes % self.scales == 0, "planes must be divisible by scale"
        self.planes_per_scale = planes // self.scales # 각 스케일 그룹의 채널 수

        # 그룹 수가 1보다 커야 ResNeXt 효과가 발생 (3x3 Convs에 Grouped Conv 적용)
        self.groups = groups 

        # Res2Net의 3x3 Conv들 (Grouped Conv로 대체)
        # s-1 개의 3x3 Conv가 필요합니다.
        self.convs = nn.ModuleList()
        for i in range(self.scales - 1):
            self.convs.append(
                nn.Conv2d(self.planes_per_scale, self.planes_per_scale, kernel_size=3, stride=1, 
                          padding=1, groups=self.groups, bias=False)
            )
            # ResNeXt와 Res2Net의 조합을 위해, inplanes와 outplanes는 planes_per_scale이지만,
            # groups를 적용하여 Grouped Conv가 되도록 합니다.
            # BaseWidth (planes_per_scale)가 groups로 나누어떨어지지 않을 수 있어 ResNeXt의 원래 width 계산은 생략합니다.
            
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.planes_per_scale) for _ in range(self.scales - 1)])

        # ResNet BasicBlock의 세 번째 (Bottleneck이 아닌 BasicBlock에서는 두 번째) 1x1 Conv 역할
        # Res2Net의 융합된 출력을 최종적으로 처리합니다.
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        
        # SE Block 추가
        self.se = SELayer(planes, reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # --- Res2Net Core ---
        # 1. 채널 분할 (Split)
        spx = torch.split(out, self.planes_per_scale, dim=1) # (B, C, H, W) -> (B, C/s, H, W) * s
        
        sp = []
        sp.append(spx[0]) # x1은 그대로 통과
        
        for i in range(self.scales - 1):
            # 2. 계층적 연결 (Hierarchical Connection)
            if i == 0:
                # x2 = conv(x2)
                sp_i = spx[i+1] # x2
            else:
                # xi = conv(xi + y_i-1)
                # 이전 스케일의 출력(y_i-1)과 현재 스케일의 입력(x_i)을 더함
                sp_i = spx[i+1] + sp[i] 
            
            sp_i = self.convs[i](sp_i)
            sp_i = self.bns[i](sp_i)
            sp_i = self.relu(sp_i)
            sp.append(sp_i)

        # 3. 피처 융합 (Concatenation)
        out = torch.cat(sp, 1)

        # 4. 최종 3x3 Conv (BasicBlock의 마지막 Conv 역할)
        out = self.conv3(out)
        out = self.bn3(out)

        # --- SE Block ---
        out = self.se(out)

        # 5. 잔차 연결 (Residual Connection)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Res2NeXtSE18_Small(nn.Module):
    def __init__(self, num_classes=10, groups=4, scale=4):
        super(Res2NeXtSE18_Small, self).__init__()
        
        self.inplanes = 64
        self.groups = groups
        self.scale = scale
        
        # 64x64 입력용 Stem (CIFAR 스타일)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool 없음

        # Layers (각 2개의 블록 = 18 레이어)
        self.layer1 = self._make_layer(Res2NeXtSEBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(Res2NeXtSEBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(Res2NeXtSEBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(Res2NeXtSEBasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Res2NeXtSEBasicBlock.expansion, num_classes)

        # 가중치 초기화는 생략합니다.

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # groups와 scale 파라미터를 block 생성 시 전달
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups, scale=self.scale))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, scale=self.scale))

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
        x = self.fc(x)

        return x
