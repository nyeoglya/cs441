import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. H-Swish 활성화 함수 (사용자 정의)
class HSwish(nn.Module):
    """
    Hard Swish: ReLU6와 Sigmoid를 사용해 근사한 활성화 함수. 
    계산 비용이 Swish보다 낮고 하드웨어 친화적입니다.
    """
    def __init__(self):
        super(HSwish, self).__init__()
        
    def forward(self, x):
        # x * ReLU6(x + 3) / 6
        out = x * F.relu6(x + 3., inplace=True) / 6.
        return out

# 2. Squeeze-and-Excitation Block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4): 
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            # [수정됨] nn.HSwish() -> HSwish() 
            # nn 모듈이 아니라 위에서 정의한 클래스를 직접 사용해야 합니다.
            HSwish() 
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * self.sigmoid(y)

# 3. MobileNetV3의 핵심: bneck (Inverse Residual with SE)
class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup

        # 활성화 함수 설정
        if use_hs:
            # [수정됨] 여기는 잘 작성되어 있었습니다. (HSwish 클래스 사용)
            self.activation = HSwish()
        else:
            self.activation = nn.ReLU(inplace=True)

        # 1. 확장 (Expansion) - 1x1 Conv
        if inp != hidden_dim:
            self.conv_pw = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self.activation
            )
        else:
            self.conv_pw = nn.Identity()

        # 2. 깊이별 합성곱 (Depthwise Conv) - kxk Conv
        self.conv_dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                      (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self.activation,
        )

        # SE 모듈 추가
        self.se = SELayer(hidden_dim) if use_se else nn.Identity()

        # 3. 투영 (Projection) - 1x1 Conv
        self.conv_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        identity = x
        
        out = self.conv_pw(x)
        out = self.conv_dw(out)
        out = self.se(out)
        out = self.conv_proj(out)

        if self.use_res_connect:
            return out + identity
        else:
            return out

# 4. MobileNetV3-Small 메인 네트워크
class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV3_Small, self).__init__()

        # 1. Stem (Initial Conv)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish() # 사용자 정의 클래스 사용
        )
        
        # 2. Bneck Sequence
        self.bnecks = nn.Sequential(
            InvertedResidual(16, 16, 16, 3, 1, True, False),
            InvertedResidual(16, 72, 24, 3, 2, False, False),
            InvertedResidual(24, 88, 24, 3, 1, False, False),
            InvertedResidual(24, 96, 40, 5, 2, True, True),
            InvertedResidual(40, 240, 40, 5, 1, True, True),
            InvertedResidual(40, 240, 40, 5, 1, True, True),
            InvertedResidual(40, 120, 48, 5, 1, True, True),
            InvertedResidual(48, 144, 48, 5, 1, True, True),
            InvertedResidual(48, 288, 96, 5, 2, True, True),
            InvertedResidual(96, 576, 96, 5, 1, True, True),
            InvertedResidual(96, 576, 96, 5, 1, True, True)
        )

        # 3. Head (Final Stages)
        self.conv_last = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(576),
            HSwish() # 사용자 정의 클래스 사용
        )

        # 4. Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            HSwish(), # 사용자 정의 클래스 사용
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bnecks(x)
        x = self.conv_last(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
