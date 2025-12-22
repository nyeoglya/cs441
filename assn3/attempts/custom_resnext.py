import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------
# Mish 클래스 삭제 (ReLU로 대체)
# ----------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        
        # Keras 코드의 로직: filters // ratio
        squeeze_channels = max(1, in_channels // ratio)
        
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.squeeze(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out

class SEMXResNeXtBlock(nn.Module):
    """Residual Module with Grouped Convs, Pre-activation, and SE"""
    def __init__(self, in_channels, out_channels, stride, groups, width_per_group, base_width):
        super(SEMXResNeXtBlock, self).__init__()
        
        # width 계산
        width_ratio = out_channels / float(base_width)
        inner_width = int(width_ratio * width_per_group) * groups

        self.bn1 = nn.BatchNorm2d(in_channels, eps=2e-5)
        self.act1 = nn.ReLU(inplace=True) # <--- ReLU로 변경
        self.conv1 = nn.Conv2d(in_channels, inner_width, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inner_width, eps=2e-5)
        self.act2 = nn.ReLU(inplace=True) # <--- ReLU로 변경
        # padding=1은 kernel_size=3일 때 'same' 패딩 효과
        self.conv2 = nn.Conv2d(inner_width, inner_width, kernel_size=3, stride=stride, 
                               groups=groups, padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(inner_width, eps=2e-5)
        self.act3 = nn.ReLU(inplace=True) # <--- ReLU로 변경
        self.conv3 = nn.Conv2d(inner_width, out_channels, kernel_size=1, bias=False)

        self.se = SEBlock(out_channels)

        # Shortcut connection logic
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            layers = []
            if stride != 1:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=stride, padding=0)) 
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            self.shortcut = nn.Sequential(*layers)
            self.has_projection = True
        else:
            self.has_projection = False

    def forward(self, x):
        # Pre-activation: BN -> Act -> Conv
        out = self.bn1(x)
        out = self.act1(out)
        
        # Projection이 필요한 경우, 활성화된 'out'을 shortcut 입력으로 사용
        shortcut_input = out if self.has_projection else x
        
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.act2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act3(out)
        out = self.conv3(out)

        out = self.se(out)

        # Shortcut 더하기
        res = self.shortcut(shortcut_input)
        
        return out + res

class SEMXResNeXt(nn.Module):
    # height, width 인자는 사용되지 않으므로 제거
    def __init__(self, depth, classes, stages, filters, groups, width_per_group):
        super(SEMXResNeXt, self).__init__()
        
        self.base_width = filters[1]
        input_channels = depth
        
        # Stem
        self.stem = nn.Sequential()
        self.stem.add_module("conv", nn.Conv2d(input_channels, filters[0], 3, stride=1, padding=1, bias=False))
        
        current_channels = filters[0]
        
        # Stages
        self.stages = nn.ModuleList()
        for i, stage_len in enumerate(stages):
            stage_out_channels = filters[i + 1]
            stride = 1 if i == 0 else 2
            
            # First block in stage
            self.stages.append(
                SEMXResNeXtBlock(current_channels, stage_out_channels, stride, 
                                 groups, width_per_group, self.base_width)
            )
            current_channels = stage_out_channels
            
            # Remaining blocks
            for _ in range(stage_len - 1):
                self.stages.append(
                    SEMXResNeXtBlock(current_channels, stage_out_channels, 1, 
                                     groups, width_per_group, self.base_width)
                )

        # Final Layers
        self.final_bn = nn.BatchNorm2d(current_channels, eps=2e-5)
        self.final_act = nn.ReLU(inplace=True) # <--- ReLU로 변경
        
        # Concatenated Pooling (Avg + Max) -> 2 * Channels
        self.classifier = nn.Linear(current_channels * 2, classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # ReLU에 맞춰 Kaiming 초기화 수행 (기존과 동일)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # ReLU에 맞춰 Kaiming 초기화 수행
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for block in self.stages:
            x = block(x)
        x = self.final_bn(x)
        x = self.final_act(x)
        
        avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        x = self.classifier(x)
        return x

def SEMXResNeXtFac(depth=3, classes=15):
    return SEMXResNeXt(depth, classes,
                       stages=[3, 3, 6, 3],
                       filters=[64, 64, 128, 256, 512],
                       groups=32,
                       width_per_group=8)
