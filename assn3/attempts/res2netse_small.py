import torch
import torch.nn as nn
import torch.nn.functional as F # F.relu6 대신 F 사용, math는 필요 없음

# 1. SE Layer (공통)
class SELayer(nn.Module):
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

# 2. Res2NetSE Basic Block (논리 통합)
class Res2NetSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, scale=4, reduction=16):
        super(Res2NetSEBasicBlock, self).__init__()
        
        self.scale = scale
        assert planes % self.scale == 0, "planes must be divisible by scale"
        self.width = planes // self.scale
        self.stride = stride
        
        # 1. Bottleneck/Initial 1x1 Conv (ResNet BasicBlock의 첫 번째 Conv 역할)
        # Res2Net의 논리를 위해 1x1 Conv 대신 3x3 Conv를 그대로 사용하되, 
        # 뒤이은 계층적 구조를 위해 Stride=1로 고정
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Res2Net Core (계층적 3x3 Convs)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.scale - 1):
            self.convs.append(
                nn.Conv2d(self.width, self.width, kernel_size=3, stride=1, padding=1, bias=False)
            )
            self.bns.append(nn.BatchNorm2d(self.width))
        
        # 3. 최종 Conv (여기서 Stride 적용)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        
        # 4. SE Block
        self.se = SELayer(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 1. Initial Conv (BasicBlock의 첫 Conv 역할)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # --- Res2Net Core ---
        # 1. 채널 분할
        spx = torch.split(out, self.width, 1)
        
        sp = []
        sp.append(spx[0]) # x1은 그대로 통과 (y1 = x1)
            
        # 2. 계층적 연산 수행
        for i in range(self.scale - 1):
            # 현재 입력 x_i+1 (i+2번째 그룹)
            current_input = spx[i+1] 
            
            if i > 0:
                # 이전 단계의 출력 (sp[i])과 현재 입력 (current_input)을 더함
                current_input = current_input + sp[i] 

            # Conv -> BN -> ReLU
            sp_i = self.convs[i](current_input) 
            sp_i = self.bns[i](sp_i)
            sp_i = self.relu(sp_i)
            sp.append(sp_i)
            
        # 3. 피처 융합
        out = torch.cat(sp, 1)
        
        # 4. 최종 Conv (Stride 적용)
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 5. SE Block
        out = self.se(out)

        # 6. 잔차 연결
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 3. General Res2NetSE Class (이 부분은 이전 코드와 동일하며 안정적입니다.)
class Res2NetSE_General(nn.Module):
    # ... (이전 코드와 동일)
    def __init__(self, layers, num_classes=10, scale=4):
        super(Res2NetSE_General, self).__init__()
        
        self.inplanes = 64
        self.scale = scale
        
        # Stem (64x64 Input Optimized: 3x3, s1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Layers
        self.layer1 = self._make_layer(Res2NetSEBasicBlock, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(Res2NetSEBasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Res2NetSEBasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(Res2NetSEBasicBlock, 512, layers[3], stride=2)

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

# 4. 모델 생성 함수들
def res2netse11(num_classes=15):
    """Layers: [1, 1, 1, 1] -> Depth approx 11"""
    return Res2NetSE_General([1, 1, 1, 1], num_classes=num_classes)

def res2netse13(num_classes=15):
    """Layers: [1, 1, 2, 1] -> Depth approx 13"""
    return Res2NetSE_General([1, 1, 2, 1], num_classes=num_classes)

def res2netse15(num_classes=15):
    """Layers: [2, 1, 2, 1] -> Depth approx 15"""
    return Res2NetSE_General([2, 1, 2, 1], num_classes=num_classes)

def res2netse17(num_classes=15):
    """Layers: [2, 2, 2, 1] -> Depth approx 17"""
    return Res2NetSE_General([2, 2, 2, 1], num_classes=num_classes)
