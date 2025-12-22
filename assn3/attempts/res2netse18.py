import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """Squeeze-and-Excitation Block"""
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

class Res2NetSEBasicBlock(nn.Module):
    """
    Res2Net + SE Block이 통합된 BasicBlock
    - ResNeXt의 Grouped Conv 대신 일반 Conv 사용 (groups=1)
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4, reduction=16):
        super(Res2NetSEBasicBlock, self).__init__()
        
        # ResNet BasicBlock의 첫 번째 3x3 Conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.scales = scale  # Res2Net의 스케일 차원 (채널 분할 개수)
        assert planes % self.scales == 0, "planes must be divisible by scale"
        self.planes_per_scale = planes // self.scales # 각 스케일 그룹의 채널 수

        # Res2Net의 3x3 Conv들 (일반 Conv)
        # s-1 개의 3x3 Conv가 필요하며, 입력/출력 채널은 planes_per_scale입니다.
        self.convs = nn.ModuleList()
        for i in range(self.scales - 1):
            self.convs.append(
                nn.Conv2d(self.planes_per_scale, self.planes_per_scale, kernel_size=3, stride=1, 
                          padding=1, groups=1, bias=False) # groups=1로 설정
            )
            
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.planes_per_scale) for _ in range(self.scales - 1)])

        # 최종 3x3 Conv (BasicBlock의 마지막 Conv 역할)
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
                sp_i = spx[i+1] # x2
            else:
                sp_i = spx[i+1] + sp[i] # xi = xi + y_i-1
            
            sp_i = self.convs[i](sp_i)
            sp_i = self.bns[i](sp_i)
            sp_i = self.relu(sp_i)
            sp.append(sp_i)

        # 3. 피처 융합 (Concatenation)
        out = torch.cat(sp, 1)

        # 4. 최종 3x3 Conv
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

class Res2NetSE18_Small(nn.Module):
    def __init__(self, num_classes=10, scale=4):
        super(Res2NetSE18_Small, self).__init__()
        
        self.inplanes = 64
        self.scale = scale
        
        # 64x64 입력용 Stem (CIFAR 스타일)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Layers (각 2개의 블록 = 18 레이어)
        self.layer1 = self._make_layer(Res2NetSEBasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(Res2NetSEBasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(Res2NetSEBasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(Res2NetSEBasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Res2NetSEBasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, scale=self.scale))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, scale=self.scale))

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
