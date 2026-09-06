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

class Reinforcement_ELBO_Loss(ELBO_Loss):
  def __init__(self):
    self.baseline = self.RunningMeanBaseline()

  class RunningMeanBaseline:
    def __init__(self, alpha: float = 0.99):
      self.alpha = alpha
      self.value = 0.0

    def __call__(self, rewards):
      mean = rewards.mean().item()
      self.value = self.alpha * self.value + (1 - self.alpha) * mean
      return torch.full_like(rewards, self.value)

  def compute_loss(self, x, out: VAEOutput, **kwargs):
    lambda_ = kwargs.get("lambda_")
    if lambda_ is None:
      raise ValueError("lambda_ cannot be None")

    lambda_ = torch.tensor(lambda_)
    rescale_ = kwargs.get("rescale", 1e-2)
    lam = out.p1
    zs = out.p2
    ys = out.reconstruction

    kl_div = (lam * (torch.log(lam) - torch.log(lambda_)) - lam + lambda_)

    free_bits = 0.5
    kl_div = torch.clamp(kl_div, min=free_bits).mean()

    rewards = []
    for y_k in ys:
      f_k = (y_k - x).flatten(1).abs().mean(dim=-1)
      f_k = torch.nan_to_num(f_k, nan=0.0)
      rewards.append(f_k)

    rewards = torch.stack(rewards, dim=0)
    #baselines = self.baseline(rewards)
    total = rewards.sum(0, keepdim=True)
    baselines = (total - rewards) / (len(zs) - 1)

    # Attaching rewards and baseline to each sample's grad_fn
    B, D = lam.shape
    for k, z_k in enumerate(zs):
      r_k = rewards[k].unsqueeze(-1).expand(B, D).detach()
      b_k = baselines[k].unsqueeze(-1).expand(B, D).detach()
      z_k.grad_fn._reward = r_k
      z_k.grad_fn._baseline = b_k

    recon = rewards.mean()
    #recon = (ys[0] - x).flatten(1).abs().mean()
    return kl_div * rescale_, recon