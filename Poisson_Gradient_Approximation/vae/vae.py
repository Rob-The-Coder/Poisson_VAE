import torch

from pathlib import Path
from typing import Optional
from utils import Model_Args
from vae import Encoder, Decoder


class VAE(torch.nn.Module):
  def __init__(
    self,
    height,
    width,
    latent_dim,
    sampling,
  ):

    super().__init__()
    self.__latent_dim = latent_dim
    self.__height = height
    self.__width = width

    self.encoder = Encoder(height, width, latent_dim)
    self.decoder = Decoder(height, width, latent_dim)
    self.__sampling = sampling

  @staticmethod
  def __restore_vae(data):
    vae = VAE(data["height"], data["width"], data["latent_dim"], data["sampling"])
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
      data = torch.load(Path(model_args.project_dir) / "models/" / model_args.vae_filename, weights_only=False, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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
      "sampling": self.__sampling
    }

    # Saving model on filesystem
    torch.save(data, Path(model_args.project_dir) / "models/" / model_args.vae_filename)

    return data

  def forward(self, x):
    lam = self.encoder(x)
    z = self.__sampling(lam)
    y = self.decoder(z)
    return lam, y