import torch
import torch.nn.functional as F

class DownsampleBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, stride=2):
    super().__init__()
    self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.inst_norm = torch.nn.InstanceNorm2d(out_channels, affine=True)
    self.prelu = torch.nn.PReLU()

  def forward(self, x):
    return self.prelu(self.inst_norm(self.conv(x)))

class MBConvBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, expand_ratio=6, reduction=4):
    super().__init__()
    self.use_residual = (in_channels==out_channels)
    mid_channels = in_channels * expand_ratio

    # Expansion
    self.expand_conv = torch.nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
    self.inst_norm1 = torch.nn.InstanceNorm2d(mid_channels, affine=True)
    self.act1 = torch.nn.PReLU()

    # Depthwise
    self.dw_conv = torch.nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels,
                                   bias=False)
    self.inst_norm2 = torch.nn.InstanceNorm2d(mid_channels, affine=True)
    self.act2 = torch.nn.PReLU()

    # Squeeze-and-Excitation
    self.se_reduce = torch.nn.Conv2d(mid_channels, mid_channels // reduction, kernel_size=1)
    self.se_expand = torch.nn.Conv2d(mid_channels // reduction, mid_channels, kernel_size=1)
    self.act3 = torch.nn.PReLU()
    self.sigmoid = torch.nn.Sigmoid()

    # Projection
    self.project_conv = torch.nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
    self.inst_norm3 = torch.nn.InstanceNorm2d(out_channels, affine=True)
    self.act4 = torch.nn.PReLU()

  def forward(self, x):
    identity = x
    x = self.act1(self.inst_norm1(self.expand_conv(x)))
    x = self.act2(self.inst_norm2(self.dw_conv(x)))

    # SE optimization
    se = F.adaptive_avg_pool2d(x, 1)
    se = self.act3(self.se_reduce(se))
    se = self.sigmoid(self.se_expand(se))
    x = x * se

    x = self.inst_norm3(self.project_conv(x))
    if self.use_residual:
      x = x + identity
    return self.act4(x)


class Encoder_GRT_60M(torch.nn.Module):
  def __init__(self, height, width, latent_features, num_res_blocks=8):
    super().__init__()
    self.initial_conv = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)

    self.down = torch.nn.Sequential(
      DownsampleBlock(16, 32),
      DownsampleBlock(32, 64),
      DownsampleBlock(64, 128),
      DownsampleBlock(128, 256),
      DownsampleBlock(256, 320)
    )

    self.res_blocks = torch.nn.ModuleList([
      MBConvBlock(320, 320) for _ in range(num_res_blocks)
    ])

    n = (height // 32) * (width // 32)
    self.fc_mu = torch.nn.Sequential(
      torch.nn.Linear(n * 320, latent_features),
      torch.nn.LayerNorm(latent_features),
      torch.nn.PReLU(),
      torch.nn.Linear(latent_features, latent_features)
    )
    self.fc_var = torch.nn.Sequential(
      torch.nn.Linear(n * 320, latent_features),
      torch.nn.LayerNorm(latent_features),
      torch.nn.PReLU(),
      torch.nn.Linear(latent_features, latent_features)
    )

  def forward(self, x):
    x = self.initial_conv(x)
    x = self.down(x)

    for block in self.res_blocks:
      x = block(x)

    x = x.flatten(1)

    mu = self.fc_mu(x)
    log_var = self.fc_var(x)
    return mu, log_var