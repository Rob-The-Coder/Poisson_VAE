import torch

class CustomPoissonSampling(torch.autograd.Function):
  @staticmethod
  def forward(ctx, lam):
    z = torch.poisson(lam)

    ctx.save_for_backward(z, lam)
    return z

  @staticmethod
  def backward(ctx, grad_output):
    z, lam = ctx.saved_tensors
    grad_lam = grad_output * (0.5 + z / (2 * lam))

    return grad_lam

class GaussianReparametrizationTrick(torch.autograd.Function):
  @staticmethod
  def forward(ctx, mu, log_var):
    eps = torch.randn_like(mu)
    std = torch.exp(0.5 * log_var)
    z = eps * std + mu  # Reparametrization trick

    ctx.save_for_backward(eps, log_var)
    return z

  @staticmethod
  def backward(ctx, grad_output):
    eps, log_var = ctx.saved_tensors
    std = torch.exp(0.5 * log_var)

    grad_mu = grad_output
    grad_log_var = grad_output * eps * std * 0.5

    return grad_mu, grad_log_var

class ReinforcementLearningTrick(torch.autograd.Function):
  @staticmethod
  def forward(ctx, lam, k):
    z = torch.poisson(lam)

    ctx.save_for_backward(z, lam)
    ctx._reward = None
    ctx._baseline = None
    ctx._k = k
    return z

  @staticmethod
  def backward(ctx, grad_output):
    z, lam = ctx.saved_tensors
    reward = ctx._reward  # (K, B, latent_dim) or (B, latent_dim)
    baseline = ctx._baseline  # scalar
    k = ctx._k

    advantage = reward - baseline  # center the signal
    #advantage = torch.nan_to_num(advantage, nan=0.0)
    #advantage = advantage / advantage.std().clamp(min=1e-8)

    score = (z - lam) / lam.clamp(min=1e-5)  # ∂ log p(z|λ) / ∂λ  =  (z-λ)/λ
    grad_lam = (advantage * score) / k

    return grad_lam, None