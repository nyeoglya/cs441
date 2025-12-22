import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, r=4):
        super().__init__()
        squeeze_channels = in_channels // r
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, 1),
            nn.SiLU(),
            nn.Conv2d(squeeze_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=1, stride=1, drop_connect_rate=0.0):
        super().__init__()
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        # Expansion + Convolution (3x3)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
        else:
            # expand_ratio가 1이면 바로 3x3 Conv
            layers.append(nn.Conv2d(in_channels, hidden_dim, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())

        # Pointwise Conv (Project) -> Linear
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                out = self._drop_connect(out)
            out = x + out
        return out

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob + torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1, drop_connect_rate=0.0):
        super().__init__()
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        # 1. Expansion (1x1)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())

        # 2. Depthwise Conv (3x3)
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU())

        # 3. Squeeze and Excitation
        layers.append(SEBlock(hidden_dim, r=4))

        # 4. Pointwise Conv (1x1) - Project
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                out = self._drop_connect(out)
            out = x + out
        return out

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob + torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor

class EfficientNetV2_Tiny_64x64(nn.Module):
    def __init__(self, num_classes=15, use_tta=True):
        super().__init__()
        self.use_tta = use_tta
        
        # 64x64 이미지에 맞춘 간소화된 구조
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )

        # Body: [Operator, In, Out, Expand, Stride, Layers]
        self.blocks = nn.ModuleList()
        config = [
            # Stage 1: FusedMBConv (64x64 유지)
            [FusedMBConv, 24, 24, 1, 1, 2],
            # Stage 2: FusedMBConv (64x64 -> 32x32)
            [FusedMBConv, 24, 48, 4, 2, 3],
            # Stage 3: MBConv + SE (32x32 -> 16x16)
            [MBConv, 48, 64, 4, 2, 3],
            # Stage 4: MBConv + SE (16x16 -> 8x8)
            [MBConv, 64, 128, 4, 2, 4],
            # Stage 5: MBConv + SE (8x8 -> 8x8)
            [MBConv, 128, 160, 6, 1, 2],
        ]

        drop_connect_rate = 0.2
        total_layers = sum(c[5] for c in config)
        layer_idx = 0

        for block_type, cin, cout, expand, stride, num_layers in config:
            for i in range(num_layers):
                # 첫 레이어만 stride 적용, 나머지는 1
                s = stride if i == 0 else 1
                inp = cin if i == 0 else cout
                
                # Drop connect rate을 층 깊이에 따라 점진적으로 증가
                dc_rate = drop_connect_rate * layer_idx / total_layers
                
                self.blocks.append(block_type(inp, cout, expand, s, dc_rate))
                layer_idx += 1

        # Head
        self.head = nn.Sequential(
            nn.Conv2d(160, 1024, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

        # Weight Initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def forward(self, x):
        if self.training or not self.use_tta:
            return self.forward_features(x)
        else:
            logit_orig = self.forward_features(x)
            x_flip = torch.flip(x, dims=[3])
            logit_flip = self.forward_features(x_flip)
            
            return (logit_orig + logit_flip) / 2.0

class EfficientNetV2_Large_64x64(nn.Module):
    def __init__(self, num_classes=15, use_tta=True):
        super().__init__()
        self.use_tta = use_tta
        
        # Stem 채널 증가: 24 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # Body Config: [Operator, In, Out, Expand, Stride, Layers]
        # 파라미터를 대폭 늘리기 위해 Width(Out)와 Depth(Layers)를 키웠습니다.
        self.blocks = nn.ModuleList()
        config = [
            # Stage 1: FusedMBConv (64x64) - 저수준 특징 강화
            # 채널: 32->32, 레이어: 2->4
            [FusedMBConv, 32, 32, 1, 1, 4],
            
            # Stage 2: FusedMBConv (64x64 -> 32x32)
            # 채널: 32->64, 레이어: 3->6, 확장비: 4
            [FusedMBConv, 32, 64, 4, 2, 6],
            
            # Stage 3: MBConv + SE (32x32 -> 16x16)
            # 채널: 64->128, 레이어: 3->6
            [MBConv, 64, 128, 4, 2, 6],
            
            # Stage 4: MBConv + SE (16x16 -> 8x8) - 핵심 특징 추출 구간
            # 채널: 128->256, 레이어: 4->10 (가장 깊게 설정)
            [MBConv, 128, 256, 4, 2, 10],
            
            # Stage 5: MBConv + SE (8x8 -> 8x8) - 고차원 특징 정리
            # 채널: 256->384, 레이어: 2->6, 확장비: 6
            [MBConv, 256, 384, 6, 1, 6],
        ]

        # Drop Connect Rate도 모델이 커진 만큼 약간 상향 (Overfitting 방지)
        drop_connect_rate = 0.3 
        total_layers = sum(c[5] for c in config)
        layer_idx = 0

        for block_type, cin, cout, expand, stride, num_layers in config:
            for i in range(num_layers):
                # 첫 레이어만 stride 적용, 나머지는 1
                s = stride if i == 0 else 1
                inp = cin if i == 0 else cout
                
                # Drop connect rate을 층 깊이에 따라 점진적으로 증가
                dc_rate = drop_connect_rate * layer_idx / total_layers
                
                self.blocks.append(block_type(inp, cout, expand, s, dc_rate))
                layer_idx += 1

        # Head 확장: 384 -> 1792 (EfficientNet 표준)
        self.head = nn.Sequential(
            nn.Conv2d(384, 1792, 1, bias=False),
            nn.BatchNorm2d(1792),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3), # Dropout도 약간 강화
            nn.Linear(1792, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def forward(self, x):
        if self.training or not self.use_tta:
            return self.forward_features(x)
        else:
            logit_orig = self.forward_features(x)
            x_flip = torch.flip(x, dims=[3])
            logit_flip = self.forward_features(x_flip)
            return (logit_orig + logit_flip) / 2.0

class SharedHierarchicalClassifier(nn.Module):
    def __init__(self, use_tta=True):
        super().__init__()
        self.use_tta = use_tta
        
        # 1. Shared Backbone (EfficientNetV2-Tiny Config)
        # Stem: 3 -> 24
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )

        # Body Config: [Operator, In, Out, Expand, Stride, Layers]
        self.blocks = nn.ModuleList()
        config = [
            # Stage 1
            [FusedMBConv, 24, 24, 1, 1, 2],
            # Stage 2
            [FusedMBConv, 24, 48, 4, 2, 3],
            # Stage 3
            [MBConv, 48, 64, 4, 2, 3],
            # Stage 4
            [MBConv, 64, 128, 4, 2, 4],
            # Stage 5 (Last Block Output: 160)
            [MBConv, 128, 160, 6, 1, 2], 
        ]

        drop_connect_rate = 0.2
        total_layers = sum(c[5] for c in config)
        layer_idx = 0
        
        for block_type, cin, cout, expand, stride, num_layers in config:
            for i in range(num_layers):
                s = stride if i == 0 else 1
                inp = cin if i == 0 else cout
                dc_rate = drop_connect_rate * layer_idx / total_layers
                self.blocks.append(block_type(inp, cout, expand, s, dc_rate))
                layer_idx += 1
                
        # 2. Multi-Head Definition (수정된 부분)
        # Tiny 모델의 마지막 출력 채널은 160입니다. (Large는 384였음)
        self.last_channels = 160  
        
        # Head의 크기도 Tiny에 맞춰 줄입니다. (Large는 1792였음)
        self.final_features = 1024 
        
        # Head 공통 부분 (Conv 1x1 -> Pooling)
        self.head_conv = nn.Sequential(
            nn.Conv2d(self.last_channels, self.final_features, 1, bias=False),
            nn.BatchNorm2d(self.final_features),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Head 1: 1차 분류기 (14개 클래스: 10/13 통합됨)
        self.head_multi = nn.Sequential(
            nn.Dropout(0.2), # Tiny 모델이므로 Dropout을 0.3 -> 0.2로 약간 낮춤 (선택사항)
            nn.Linear(self.final_features, 14) 
        )
        
        # Head 2: 2차 분류기 (2개 클래스: 10 vs 13)
        self.head_binary = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.final_features, 2)
        )
        
        self._init_weights()
        
        # 상수 정의
        self.unified_class_idx = 13
        self.orig_10 = 10
        self.orig_13 = 13

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """백본을 통과하여 특징 벡터 추출 (공유됨)"""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_conv(x)
        return x

    def forward(self, x):
        """
        [추론 모드] 
        1차 분류 후, '파편류(13)'인 경우 2차 분류기 결과로 상세 구분하여
        최종 15개 클래스(0~14) 예측값 반환
        """
        if self.use_tta and not self.training:
            # TTA: 원본 + 좌우반전 평균
            feat_orig = self.forward_features(x)
            feat_flip = self.forward_features(torch.flip(x, dims=[3]))
            features = (feat_orig + feat_flip) / 2.0
        else:
            features = self.forward_features(x)
            
        # 1. 1차 분류 (14개)
        out_multi = self.head_multi(features)
        pred_multi = out_multi.argmax(dim=1)
        
        # 최종 예측 배열 (1차 예측 복사)
        final_preds = pred_multi.clone()
        
        # 2. 통합 클래스(13번)로 예측된 샘플 찾기
        indices = (pred_multi == self.unified_class_idx).nonzero(as_tuple=True)[0]
        
        if indices.numel() > 0:
            # 해당 샘플만 2차 헤드 통과 (연산 효율성)
            feat_refine = features[indices]
            out_binary = self.head_binary(feat_refine)
            pred_binary = out_binary.argmax(dim=1) # 0(Class 10) or 1(Class 13)
            
            # 매핑: 0 -> 10, 1 -> 13
            final_preds[indices] = torch.where(
                pred_binary == 1,
                torch.tensor(self.orig_13, device=x.device),
                torch.tensor(self.orig_10, device=x.device)
            )
            
        return final_preds
