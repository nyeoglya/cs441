import numpy as np
import torch

import torch.nn.functional as F
import torch.nn as nn

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        # 람다(lambda)를 Beta 분포에서 샘플링 (일반적으로 alpha=1.0)
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    # 이미지 텐서 혼합
    mixed_x = lam * x + (1 - lam) * x[index, :]
    
    # 레이블 혼합 (y_a = 원래 레이블, y_b = 섞인 레이블)
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    '''Mixup을 위한 손실 함수 계산'''
    # 두 레이블에 대한 손실을 람다 비율로 가중 평균
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda (area ratio)'''
    if alpha > 0:
        # 람다(lambda)를 Beta 분포에서 샘플링
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1
        
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    
    # 람다를 조정 (면적 비율)
    lam = 1 - lam # 잘린 영역의 비율 (1-lam)을 혼합 비율로 사용
    
    # Bounding Box (잘릴 영역) 계산
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    # 이미지 텐서 혼합 (패치 복사 및 붙여넣기)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # 레이블 혼합 (면적 비율로 람다 재계산)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform 분포에서 바운딩 박스 중심 위치 선택
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# --- 2. Model Components (Optimized for 15 Epochs) ---

# [LayerNorm, DropPath, Block 코드는 사용자님의 기존 코드와 동일하되, DropPath만 약간 수정]
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape, )
        else:
            self.normalized_shape = normalized_shape
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True) if dim > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None: x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

# --- Optimized ConvNeXt (Nano Version) ---
class ConvNeXtNano(nn.Module):
    def __init__(self, in_chans=3, num_classes=15, 
                 depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], # 파라미터 대폭 축소
                 drop_path_rate=0.05): # 규제 대폭 완화
        super().__init__()
        self.downsample_layers = nn.ModuleList() 
        # Stem: 64x64 이미지 손실 최소화 (stride=1 유지)
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList() 
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]
        
        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(x.mean([-2, -1], keepdim=True)) # Global Average Pooling
        x = x.flatten(1)
        x = self.head(x)
        return x

# --- Optimized ResNeXt (Nano Version) ---
# 간단하고 가벼운 ResNeXt 블록 직접 구현
class ResNeXtBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, cardinality=4, base_width=4):
        super(ResNeXtBlock, self).__init__()
        width = int(planes * (base_width / 64.0)) * cardinality
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNeXtNano(nn.Module):
    def __init__(self, num_classes=15, cardinality=4, base_width=8): # 파라미터 대폭 축소
        super(ResNeXtNano, self).__init__()
        self.in_planes = 32 # 시작 채널 줄임
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Layers: [2, 2, 2, 2]로 블록 수 줄임 (Standard는 [3, 4, 6, 3])
        self.layer1 = self._make_layer(32, 2, stride=1, cardinality=cardinality, base_width=base_width)
        self.layer2 = self._make_layer(64, 2, stride=2, cardinality=cardinality, base_width=base_width)
        self.layer3 = self._make_layer(128, 2, stride=2, cardinality=cardinality, base_width=base_width)
        self.layer4 = self._make_layer(256, 2, stride=2, cardinality=cardinality, base_width=base_width)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * 4, num_classes)

    def _make_layer(self, planes, num_blocks, stride, cardinality, base_width):
        layers = []
        layers.append(ResNeXtBlock(self.in_planes, planes, stride, cardinality, base_width))
        self.in_planes = planes * 4
        for _ in range(num_blocks - 1):
            layers.append(ResNeXtBlock(self.in_planes, planes, 1, cardinality, base_width))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out
