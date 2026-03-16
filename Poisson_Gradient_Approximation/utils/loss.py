import torch
from abc import ABC, abstractmethod

class CustomPoissonSampling(torch.autograd.Function):
  @staticmethod
  def forward(ctx, lam):
    sample = torch.poisson(lam)
    ctx.save_for_backward(sample, lam)
    return sample

  @staticmethod
  def backward(ctx, grad_output):
    sample, lam = ctx.saved_tensors
    grad_lam = grad_output * (0.5 + sample/(2*lam))
    return grad_lam

class GaussianReparametrizationTrick(torch.autograd.Function):
  @staticmethod
  def forward(ctx, mu, log_var):
    z = torch.randn_like(mu) * torch.exp(log_var / 2) + mu  # Reparametrization trick

    ctx.save_for_backward(z, mu, log_var)
    return z

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output

class ELBO_Loss(ABC):
  @abstractmethod
  def compute_loss(self):
    pass

class Poisson_ELBO_Loss(ELBO_Loss):
  def compute_loss(self, target, generated, lam, lambda_, rescale_=1e-2):
    lambda_ = torch.tensor(lambda_)

    rec_error = (generated - target).abs().mean()
    kl_div = (lam * (torch.log(lam) - torch.log(lambda_)) - lam + lambda_).mean()

    return (kl_div * rescale_), rec_error

class Gaussian_ELBO_Loss(ELBO_Loss):
  def compute_loss(self, target, generated, mu, log_var, rescale_=1e-2):
    rec_error = (generated - target).abs().mean()
    kl_div = 0.5 * (torch.exp(log_var).sum() + (mu ** 2).sum() - log_var.sum())

    return kl_div * rescale_, rec_error