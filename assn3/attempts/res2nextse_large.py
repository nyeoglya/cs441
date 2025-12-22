import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. Core Modules (Fixed & Robust) - (수정 없음)
# -----------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
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

class Res2NeXtConv(nn.Module):
    def __init__(self, in_planes, out_planes, stride, scale, groups):
        super(Res2NeXtConv, self).__init__()
        self.scale = scale
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        
        self.process_x1 = (in_planes != out_planes)
        self.in_width = in_planes // scale
        self.out_width = out_planes // scale
        self.relu = nn.ReLU(inplace=True)
        
        num_convs = scale if self.process_x1 else scale - 1
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for _ in range(num_convs):
            self.convs.append(
                nn.Conv2d(self.in_width, self.out_width, kernel_size=3, stride=stride, 
                          padding=1, groups=groups, bias=False)
            )
            self.bns.append(nn.BatchNorm2d(self.out_width))

    def forward(self, x):
        xs = torch.split(x, self.in_width, 1)
        ys = []
        
        if self.process_x1:
            out = self.convs[0](xs[0])
            out = self.bns[0](out)
            out = self.relu(out)
            ys.append(out)
        else:
            if self.stride > 1:
                ys.append(F.avg_pool2d(xs[0], kernel_size=3, stride=self.stride, padding=1))
            else:
                ys.append(xs[0])

        start_idx = 1
        for i in range(start_idx, self.scale):
            x_in = xs[i]
            conv_idx = i if self.process_x1 else i - 1
            
            if self.stride == 1 and not self.process_x1:
                x_in = x_in + ys[-1]
            
            out = self.convs[conv_idx](x_in)
            out = self.bns[conv_idx](out)
            out = self.relu(out)
            ys.append(out)

        out = torch.cat(ys, 1)
        return out

# -----------------------------------------------------------------------------
# 2. Block Definitions - (수정 없음)
# -----------------------------------------------------------------------------
class Res2NeXtSEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 groups=1, base_width=64, scale=4, reduction=16):
        super(Res2NeXtSEBasicBlock, self).__init__()
        self.conv1 = Res2NeXtConv(inplanes, planes, stride, scale=scale, groups=groups)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, 
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Res2NeXtSEBottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, 
                 groups=1, base_width=64, scale=4, reduction=16):
        super(Res2NeXtSEBottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = Res2NeXtConv(width, width, stride, scale=scale, groups=groups)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# -----------------------------------------------------------------------------
# 3. Main Network Factory - [여기만 수정됨]
# -----------------------------------------------------------------------------
class Res2NeXtSE(nn.Module):
    def __init__(self, block, layers, num_classes=15, groups=1, scale=4, width_per_group=64):
        super(Res2NeXtSE, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.scale = scale
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Weight Initialization [수정됨]
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # bias가 None이 아닐 때만 0으로 초기화
                if m.bias is not None:
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
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            groups=self.groups, base_width=self.base_width, scale=self.scale))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, 
                                groups=self.groups, base_width=self.base_width, scale=self.scale))
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

# -----------------------------------------------------------------------------
# 4. Factory Functions - (수정 없음)
# -----------------------------------------------------------------------------
def res2nextse18(num_classes=15, **kwargs):
    return Res2NeXtSE(Res2NeXtSEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def res2nextse34(num_classes=15, **kwargs):
    return Res2NeXtSE(Res2NeXtSEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def res2nextse50(num_classes=15, **kwargs):
    return Res2NeXtSE(Res2NeXtSEBottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def res2nextse101(num_classes=15, **kwargs):
    return Res2NeXtSE(Res2NeXtSEBottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)

def res2nextse152(num_classes=15, **kwargs):
    return Res2NeXtSE(Res2NeXtSEBottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)