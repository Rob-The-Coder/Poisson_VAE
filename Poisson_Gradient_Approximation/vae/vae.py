import torch

from pathlib import Path
from typing import Optional

from core.vae_output import VAEOutput
from core.model_factory import ModelFactory
from core.model_args import ModelArgs
from utils.sampling import CustomPoissonSampling, GaussianReparametrizationTrick
from utils.loss import Poisson_ELBO_Loss, Gaussian_ELBO_Loss

class VAE(torch.nn.Module):
  def __init__(
    self,
    height,
    width,
    latent_dim: int,
    sampling: str = "PGA",
    model_type: str = "36M"
  ):
    super().__init__()
    self.__latent_dim = latent_dim
    self.__height = height
    self.__width = width

    self.__sampling = sampling
    SAMPLING_MAP = {
      "PGA": (Poisson_ELBO_Loss(), CustomPoissonSampling().apply, self.__forward_pga, self.__generate_pga),
      "GRT": (Gaussian_ELBO_Loss(), GaussianReparametrizationTrick().apply, self.__forward_grt, self.__generate_grt),
    }
    if self.__sampling not in SAMPLING_MAP:
      supported = ", ".join(SAMPLING_MAP.keys())
      raise ValueError(
        f"Unsupported sampling method '{self.__sampling}'. Supported: {supported}"
      )
    self.__loss_function, self.__sampling_method, self.__forward_logic, self.__generation_logic = SAMPLING_MAP[self.__sampling]

    # Instantiating correct model
    self.__model_type = model_type
    self.encoder, self.decoder = ModelFactory.create(self.__sampling, self.__model_type, self.__height, self.__width, self.__latent_dim)
    
  @staticmethod
  def __restore_vae(data):
    vae = VAE(data["height"], data["width"], data["latent_dim"], data["sampling"], data["type"])

    vae.load_state_dict(data["params"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(device)

    return vae

  @staticmethod
  def from_pretrained(model_args: Optional[ModelArgs] = None, data: Optional[dict] = None):
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

  def save_model(self, model_args: ModelArgs):
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

  # GENERATION LOGIC
  def __generate_pga(self, num_faces, device, **kwargs):
    lambda_ = kwargs.get("LAMBDA")
    if lambda_ is None:
      raise ValueError("LAMBDA cannot be None")

    z = torch.poisson(torch.full((num_faces, self.__latent_dim), lambda_, device=device, dtype=torch.float32))
    faces = self.decode(z)

    return faces

  def __generate_grt(self, num_faces, device, **kwargs):
    z = torch.randn(num_faces, self.__latent_dim, device=device)
    faces = self.decode(z)

    return faces

  def generate_faces(self, num_faces, device, **kwargs):
    return self.__generation_logic(num_faces, device, **kwargs)

  # LOSS LOGIC
  def compute_loss(self, x, out: VAEOutput, **kwargs):
    return self.__loss_function.compute_loss(x, out, **kwargs)

  # FORWARD LOGIC
  def __forward_pga(self, x):
    lam = self.encoder(x)
    z = self.__sampling_method(lam)
    y = self.decoder(z)

    return VAEOutput(reconstruction=y, p1=lam)

  def __forward_grt(self, x):
    mu, log_var = self.encoder(x)
    z = self.__sampling_method(mu, log_var)
    y = self.decoder(z)

    return VAEOutput(reconstruction=y, p1=mu, p2=log_var)

  def forward(self, x):
    return self.__forward_logic(x)