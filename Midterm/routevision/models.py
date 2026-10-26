"""
Models compared in the mid-term demo. All take a (B, 4, 64, 64) window; the
multimodal model additionally takes the (B, 8) side vector (6 design parameters +
patch row/column). Every model returns one logit per window (hotspot or not).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IN_CH, SIDE_DIM = 4, 8


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MLPNet(nn.Module):
    """Window average-pooled to 32x32 and flattened (4 x 32 x 32 = 4096 inputs)."""

    def __init__(self, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(IN_CH * 32 * 32, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x, side=None):
        return self.net(F.avg_pool2d(x, 2)).squeeze(1)


class CNNBackbone(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.b1, self.b2, self.b3 = conv_block(IN_CH, width), conv_block(width, 2 * width), conv_block(2 * width, 4 * width)
        self.out_dim = 4 * width

    def forward(self, x):
        x = F.max_pool2d(self.b1(x), 2)      # 32 x 32
        x = F.max_pool2d(self.b2(x), 2)      # 16 x 16
        return self.b3(x)                    # 16 x 16 feature map (Grad-CAM layer)


class CNNNet(nn.Module):
    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        self.backbone = CNNBackbone(width)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.out_dim, 1))

    def gradcam_layer(self):
        return self.backbone.b3

    def forward(self, x, side=None):
        f = self.backbone(x).mean(dim=(2, 3))
        return self.head(f).squeeze(1)


class MultimodalNet(nn.Module):
    """Image branch (CNN) + tabular branch (design parameters, patch position), fused by concatenation."""

    def __init__(self, width=32, dropout=0.3):
        super().__init__()
        self.backbone = CNNBackbone(width)
        self.side = nn.Sequential(nn.Linear(SIDE_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.backbone.out_dim + 32, 64), nn.ReLU(),
                                  nn.Linear(64, 1))

    def gradcam_layer(self):
        return self.backbone.b3

    def forward(self, x, side):
        f = self.backbone(x).mean(dim=(2, 3))
        return self.head(torch.cat([f, self.side(side)], 1)).squeeze(1)


class ViTNet(nn.Module):
    """ImageNet ViT-B/16 with a 4-channel patch embedding; windows are resized 64 -> 224."""

    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision import models
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        old = m.conv_proj
        new = nn.Conv2d(IN_CH, old.out_channels, old.kernel_size, old.stride)
        with torch.no_grad():
            w = old.weight                                        # (768, 3, 16, 16)
            new.weight.copy_(torch.cat([w, w.mean(1, keepdim=True)], 1) * 3 / IN_CH)
            new.bias.copy_(old.bias)
        m.conv_proj = new
        m.heads.head = nn.Linear(m.heads.head.in_features, 1)
        self.m = m

    def forward(self, x, side=None):
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        return self.m(x).squeeze(1)


class UNet(nn.Module):
    """Pixel-level regression of the whole 256x256 final routing-utilisation map.

    Encoder: ImageNet ResNet-18 with a 4-channel stem (same initialisation trick as the ViT). Decoder: U-Net
    style, upsampling with skip connections back to full resolution. use_params=True adds the second modality:
    the 6 design parameters modulate the bottleneck features through FiLM (feature-wise scale and shift).
    Output in [0, 1] (the utilisation labels are in [0, 1])."""

    def __init__(self, use_params=False, pretrained=True, n_params=6):
        super().__init__()
        from torchvision import models
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        old = r.conv1
        stem = nn.Conv2d(IN_CH, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            stem.weight.copy_(torch.cat([old.weight, old.weight.mean(1, keepdim=True)], 1) * 3 / IN_CH)
        self.stem = nn.Sequential(stem, r.bn1, r.relu)                       # 64  x 128
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4   # 64x64, 128x32, 256x16, 512x8
        self.use_params = use_params
        if use_params:
            self.film = nn.Sequential(nn.Linear(n_params, 128), nn.ReLU(), nn.Linear(128, 2 * 512))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)   # starts as identity
        self.d4 = conv_block(512 + 256, 256)
        self.d3 = conv_block(256 + 128, 128)
        self.d2 = conv_block(128 + 64, 64)
        self.d1 = conv_block(64 + 64, 64)
        self.d0 = conv_block(64 + IN_CH, 32)
        self.out = nn.Conv2d(32, 1, 1)

    @staticmethod
    def _up(x, skip):
        return torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1)

    def forward(self, x, params=None):
        s0 = self.stem(x)
        e1 = self.l1(self.pool(s0))
        e2 = self.l2(e1)
        e3 = self.l3(e2)
        e4 = self.l4(e3)
        if self.use_params:
            g, b = self.film(params).chunk(2, 1)
            e4 = e4 * (1 + g[:, :, None, None]) + b[:, :, None, None]
        d = self.d4(self._up(e4, e3))
        d = self.d3(self._up(d, e2))
        d = self.d2(self._up(d, e1))
        d = self.d1(self._up(d, s0))
        d = self.d0(self._up(d, x))
        return torch.sigmoid(self.out(d)).squeeze(1)


MODEL_ZOO = {'MLP': MLPNet, 'CNN': CNNNet, 'Multimodal': MultimodalNet, 'ViT': ViTNet}


def build(name, **kw):
    return MODEL_ZOO[name](**kw)
