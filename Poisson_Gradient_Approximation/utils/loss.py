import torch
from abc import ABC, abstractmethod

from core.vae_output import VAEOutput

class ELBO_Loss(ABC):
  @abstractmethod
  def compute_loss(self, x, out: VAEOutput, **kwargs):
    pass

class Poisson_ELBO_Loss(ELBO_Loss):
  def compute_loss(self, x, out: VAEOutput, **kwargs):
    lambda_ = kwargs.get("lambda_")
    if lambda_ is None:
      raise ValueError("lambda_ cannot be None")

    lambda_ = torch.tensor(lambda_)
    rescale_ = kwargs.get("rescale", 1e-2)

    lam = out.p1
    y = out.reconstruction
    rec_error = (y - x).abs().mean()
    kl_div = (lam * (torch.log(lam) - torch.log(lambda_)) - lam + lambda_).mean()

    return (kl_div * rescale_), rec_error

class Gaussian_ELBO_Loss(ELBO_Loss):
  def compute_loss(self, x, out: VAEOutput, **kwargs):
    rescale_ = kwargs.get("rescale", 1e-2)
    mu = out.p1
    log_var = out.p2
    y = out.reconstruction

    rec_error = (y - x).abs().mean()
    kl_div = 0.5 * (torch.exp(log_var).sum() + (mu ** 2).sum() - log_var.sum())

    return kl_div * rescale_, rec_error