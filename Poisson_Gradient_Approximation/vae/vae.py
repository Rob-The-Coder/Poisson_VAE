import torch

from pathlib import Path
from typing import Optional
from utils import Model_Args

from vae import Encoder_36M, Decoder_36M, Encoder_53M, Decoder_53M
from utils import CustomPoissonSampling, GaussianReparametrizationTrick, Poisson_ELBO_Loss, Gaussian_ELBO_LOSS

class VAE(torch.nn.Module):
  def __init__(
    self,
    height,
    width,
    latent_dim,
    sampling: str,
    model_type: str = "36M"
  ):
    super().__init__()
    self.__latent_dim = latent_dim
    self.__height = height
    self.__width = width

    # Instantiating correct model type
    self.__model_type = model_type
    MODEL_MAP = {
      "36M": (Encoder_36M(height, width, latent_dim), Decoder_36M(height, width, latent_dim)),
      "53M": (Encoder_53M(height, width, latent_dim), Decoder_53M(height, width, latent_dim)),
    }
    if self.__model_type not in MODEL_MAP:
      supported = ", ".join(MODEL_MAP.keys())
      raise ValueError(
        f"Unsupported model type '{self.__model_type}'. Supported: {supported}"
      )
    self.encoder, self.decoder = MODEL_MAP[self.__model_type]

    self.__sampling = sampling
    SAMPLING_MAP = {
      "PGA": (Poisson_ELBO_Loss(), CustomPoissonSampling().apply),
      "GRP": (Gaussian_ELBO_LOSS(), GaussianReparametrizationTrick().apply),
    }
    if self.__sampling not in SAMPLING_MAP:
      supported = ", ".join(SAMPLING_MAP.keys())
      raise ValueError(
        f"Unsupported sampling method '{self.__sampling}'. Supported: {supported}"
      )
    self.__loss_function, self.__sampling_method = SAMPLING_MAP[self.__sampling]

  @staticmethod
  def __restore_vae(data):
    vae = VAE(data["height"], data["width"], data["latent_dim"], data["sampling"], data["type"])

    vae.load_state_dict(data["params"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(device)

    return vae

  @staticmethod
  def from_pretrained(model_args: Optional[Model_Args] = None, data: Optional[dict] = None):
    if model_args is None and data is None:
      raise ValueError("Either model_args or data must be provided!")

    if model_args is not None and data is not None:
      raise ValueError("Only one of model_args or data can be provided!")

    if model_args is not None:
      # Loading model from filesystem
      data = torch.load(Path(model_args.project_dir) / "models/" / model_args.vae_filename)

    return VAE.__restore_vae(data)

  @property
  def latent_dim(self):
    return self.__latent_dim

  def set_sampling(self, sampling: torch.autograd.Function):
    self.__sampling = sampling

  def encode(self, x):
    with torch.no_grad():
      return self.encoder(x)

  def decode(self, z):
    with torch.no_grad():
      return self.decoder(z)

  def generate_faces(self, num_faces, LAMBDA, device):
    z = torch.poisson(torch.full((num_faces, self.__latent_dim), LAMBDA, device=device, dtype=torch.float32))
    faces = self.decode(z)

    return faces

  def save_model(self, model_args: Model_Args):
    data = {
      "params": self.state_dict(),
      "height": self.__height,
      "width": self.__width,
      "latent_dim": self.__latent_dim,
      "sampling": self.__sampling,
      "type": self.__model_type
    }

    # Saving model on filesystem
    torch.save(data, Path(model_args.project_dir) / "models/" / model_args.vae_filename)

    return data

  def forward(self, x):
    lam = self.encoder(x)
    z = self.__sampling_method(lam)
    y = self.decoder(z)
    return lam, y