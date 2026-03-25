import torch
import torch.nn.functional as F

## Super-resolution style decoder

class StyleModulation(torch.nn.Module):
  def __init__(self, latent_dim, num_features):
    super().__init__()
    self.to_style = torch.nn.Linear(latent_dim, num_features * 2)

    with torch.no_grad():
      self.to_style.weight.zero_()
      self.to_style.bias.data[:num_features] = 1
      self.to_style.bias.data[num_features:] = 0

  def forward(self, x, z):
    style = self.to_style(z).unsqueeze(2).unsqueeze(3)
    gamma, beta = style.chunk(2, dim=1)

    out = F.instance_norm(x)
    return out * gamma + beta

class StyleMBConvBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, latent_dim, expand_ratio=6, reduction=4):
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
    self.inst_norm3 = StyleModulation(latent_dim, out_channels)
    self.act4 = torch.nn.PReLU()

  def forward(self, x, z):
    identity = x

    # Expansion
    x =self.act1(self.inst_norm1(self.expand_conv(x)))

    # Depthwise
    x = self.act2(self.inst_norm2(self.dw_conv(x)))

    # Squeeze-and-Excitation
    se = F.adaptive_avg_pool2d(x, 1)
    se = self.act3(self.se_reduce(se))
    se = self.sigmoid(self.se_expand(se))
    x = x * se

    # Projection
    x = self.inst_norm3(self.project_conv(x), z)

    if self.use_residual:
      x = x + identity

    return self.act4(x)

class PixelShuffleBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, upscale_factor=2):
    super().__init__()

    # For upsampling with a factor of 2 with PixelShuffle we need out_channels * 4
    self.conv = torch.nn.Conv2d(in_channels, out_channels * (upscale_factor**2), kernel_size=3, padding=1)
    self.shuffle = torch.nn.PixelShuffle(upscale_factor)
    self.prelu = torch.nn.PReLU()

  def forward(self, x):
    return self.prelu(self.shuffle(self.conv(x)))

class Decoder_GRT_53M(torch.nn.Module):
  def __init__(self, height, width, latent_features, num_res_blocks=8):
    super().__init__()
    self.height = height // 32
    self.width = width // 32
    n = self.height * self.width

    self.fc = torch.nn.Sequential(
      torch.nn.Linear(latent_features, latent_features),
      torch.nn.PReLU(),
      torch.nn.Linear(latent_features, latent_features),
      torch.nn.PReLU(),                                   # mapping network
      torch.nn.Linear(latent_features, 384),
      torch.nn.LayerNorm(384),
      torch.nn.PReLU(),
      torch.nn.Linear(384, n * 384),
      torch.nn.PReLU()
    )

    # Low resolution residual blocks
    self.res_blocks = torch.nn.ModuleList([
      StyleMBConvBlock(384, 384, latent_features) for _ in range(num_res_blocks)
    ])

    # Upsampling via PixelShuffle (4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128)
    self.up1 = PixelShuffleBlock(384, 256)
    self.up2 = PixelShuffleBlock(256, 128)
    self.up3 = PixelShuffleBlock(128, 64)
    self.up4 = PixelShuffleBlock(64, 32)
    self.up5 = PixelShuffleBlock(32, 32)

    self.final_conv = torch.nn.Conv2d(32, 3, kernel_size=3, padding=1)

  def forward(self, z):
    x = self.fc(z)
    x = x.view(x.size(0), -1, self.height, self.width)

    for block in self.res_blocks:
      x = block(x, z)

    x = self.up1(x)
    x = self.up2(x)
    x = self.up3(x)
    x = self.up4(x)
    x = self.up5(x)
    return torch.tanh(self.final_conv(x))