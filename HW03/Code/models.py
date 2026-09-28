"""
Model definitions:
  - PlainCNN: a self-designed VGG-style network without shortcut connections.
  - build_resnet18: the classic ResNet-18 backbone (He et al., 2015), optionally
    initialised with ImageNet-pretrained weights for transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def conv_bn_relu(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class PlainCNN(nn.Module):
    """
    Input 3x32x32
      Block 1: [Conv3x3(32) -> BN -> ReLU] x2 -> MaxPool   -> 32x16x16
      Block 2: [Conv3x3(64) -> BN -> ReLU] x2 -> MaxPool   -> 64x8x8
      Block 3: [Conv3x3(128) -> BN -> ReLU] x2             -> 128x8x8
      Global Average Pooling -> Dropout -> Linear(num_classes)
    Block 3 keeps an 8x8 feature map so Grad-CAM heatmaps retain spatial detail.
    """

    def __init__(self, num_classes=10, width=32, dropout=0.3):
        super().__init__()
        w = width
        self.block1 = nn.Sequential(conv_bn_relu(3, w), conv_bn_relu(w, w), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(conv_bn_relu(w, 2 * w), conv_bn_relu(2 * w, 2 * w), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(conv_bn_relu(2 * w, 4 * w), conv_bn_relu(4 * w, 4 * w))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * w, num_classes)

    def features(self, x):
        return self.block3(self.block2(self.block1(x)))

    def forward(self, x):
        f = self.features(x)
        return self.fc(self.dropout(F.adaptive_avg_pool2d(f, 1).flatten(1)))

    def first_conv(self):
        return self.block1[0][0]

    def gradcam_layer(self):
        return self.block3


class ResNet18Classifier(nn.Module):
    """
    ResNet-18 with a new 10-class head. CIFAR-10 images (32x32) are upsampled to
    `input_size` before the backbone: ResNet-18 downsamples by 32x, so a raw 32x32
    input would leave a 1x1 final feature map, which both hurts ImageNet-pretrained
    features (designed for larger objects) and makes Grad-CAM meaningless.
    """

    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False, input_size=64):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = resnet18(weights=weights)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.input_size = input_size
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for name, p in self.net.named_parameters():
                if not name.startswith('fc.'):
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_backbone:
            # Frozen feature extractor: keep BatchNorm running statistics fixed too.
            self.net.eval()
            self.net.fc.train()
        return self

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        return self.net(x)

    def first_conv(self):
        return self.net.conv1

    def gradcam_layer(self):
        # At 64x64 input, layer4 is only 2x2; layer3 (4x4) is the deepest layer
        # that still gives a spatially meaningful Grad-CAM heatmap.
        return self.net.layer3


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))
