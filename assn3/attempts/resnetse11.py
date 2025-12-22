import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Squeeze-and-Excitation Block
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

# 2. ResNetSE Basic Block (groups=1)
class ResNetSEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(ResNetSEBasicBlock, self).__init__()
        # 3x3 Conv 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=1, bias=False) 
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        # 3x3 Conv 2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # SE Block 추가
        self.se = SELayer(planes, reduction)
        
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 3. ResNetSE11 메인 네트워크
class ResNetSE11_Small(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetSE11_Small, self).__init__()
        
        self.inplanes = 64
        
        # 64x64 입력용 Stem (3x3/s1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # MaxPool 없음

        # Layers: 각 Stage에 1개의 BasicBlock만 사용 (총 4개 블록, 8개 Conv Layer)
        # Stem Conv 1개 + 8개 Conv Layer = 총 9개 Conv Layer (약 11-Layer)
        self.layer1 = self._make_layer(ResNetSEBasicBlock, 64, blocks=1, stride=1)
        self.layer2 = self._make_layer(ResNetSEBasicBlock, 128, blocks=1, stride=2)
        self.layer3 = self._make_layer(ResNetSEBasicBlock, 256, blocks=1, stride=2)
        self.layer4 = self._make_layer(ResNetSEBasicBlock, 512, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNetSEBasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

# --- 테스트 실행 ---
if __name__ == "__main__":
    model = ResNetSE11_Small(num_classes=10) 
    
    # 64x64 입력 테스트
    dummy_input = torch.randn(2, 3, 64, 64) 
    output = model(dummy_input)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: ResNetSE11 (Approx. 5.8 Million Parameters)")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of Parameters: {num_params / 1e6:.2f} Million")