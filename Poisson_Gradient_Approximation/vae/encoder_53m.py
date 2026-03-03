import torch
import torch.nn.functional as F

## EfficientNet style encoder

class MBConvBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, expand_ratio=6, reduction=4):
    super().__init__()
    self.use_residual = (in_channels == out_channels)
    mid_channels = in_channels * expand_ratio

    # Expansion (1x1 conv)
    self.expand_conv = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
    self.inst_norm1 = torch.nn.InstanceNorm2d(mid_channels, affine=True)
    self.act1 = torch.nn.PReLU()

    # Depthwise (3x3, conv)
    self.dw_conv = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False)
    self.inst_norm2 = torch.nn.InstanceNorm2d(mid_channels, affine=True)
    self.act2 = torch.nn.PReLU()

    # Squeeze-and-Excitation (SE, sort of attention)
    self.se_reduce = torch.nn.Conv2d(mid_channels, mid_channels // reduction, kernel_size=1)
    self.se_expand = torch.nn.Conv2d(mid_channels // reduction, mid_channels, kernel_size=1)
    self.act3 = torch.nn.PReLU()
    self.sigmoid = torch.nn.Sigmoid()

    # Projection (1x1 conv)
    self.project_conv = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
    self.inst_norm3 = torch.nn.InstanceNorm2d(out_channels, affine=True)
    self.act4 = torch.nn.PReLU()

  def forward(self, x):
    identity = x

    # Expansion
    x = self.act1(self.inst_norm1(self.expand_conv(x)))

    # Depthwise
    x = self.act2(self.inst_norm2(self.dw_conv(x)))

    # Squeeze-and-Excitation
    se = F.adaptive_avg_pool2d(x, 1)
    se = self.act3(self.se_reduce(se))
    se = self.sigmoid(self.se_expand(se))
    x = x * se

    # Projection
    x = self.inst_norm3(self.project_conv(x))

    if self.use_residual:
      x = x + identity

    return self.act4(x)

class Encoder_53M(torch.nn.Module):
  def __init__(self, height, width, latent_features):
    super().__init__()

    # Stage 1: 128x128 -> 64x64
    self.stage1 = torch.nn.Sequential(
      MBConvBlock(3, 32),
      torch.nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
      torch.nn.InstanceNorm2d(32, affine=True),
      torch.nn.PReLU()
    )

    # Stage 2: 64x64 -> 32x32
    self.stage2 = torch.nn.Sequential(
      MBConvBlock(32, 64),
      torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
      torch.nn.InstanceNorm2d(64, affine=True),
      torch.nn.PReLU()
    )

    # Stage 3: 32x32 -> 16x16
    self.stage3 = torch.nn.Sequential(
      MBConvBlock(64, 128),
      torch.nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
      torch.nn.InstanceNorm2d(128, affine=True),
      torch.nn.PReLU()
    )

    # Stage 4: 16x16 -> 8x8
    self.stage4 = torch.nn.Sequential(
      MBConvBlock(128, 256),
      torch.nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
      torch.nn.InstanceNorm2d(256, affine=True),
      torch.nn.PReLU()
    )

    # Stage 5: 8x8 -> 4x4
    self.stage5 = torch.nn.Sequential(
      MBConvBlock(256, 384),
      torch.nn.Conv2d(384, 384, kernel_size=4, stride=2, padding=1),
      torch.nn.InstanceNorm2d(384, affine=True),
      torch.nn.PReLU()
    )

    n = (height // 32) * (width // 32)
    self.fc_lam = torch.nn.Sequential(
      torch.nn.Linear(n * 384, latent_features),
      torch.nn.LayerNorm(latent_features),
      torch.nn.PReLU(),
      torch.nn.Linear(latent_features, latent_features)
    )

  def forward(self, x):
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)

    x = x.flatten(1)

    # Assuming self.fc_lam(x) returns log values
    lam = torch.exp(self.fc_lam(x))
    return lam