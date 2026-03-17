import torch

class CustomPoissonSampling(torch.autograd.Function):
  @staticmethod
  def forward(ctx, lam):
    sample = torch.poisson(lam)
    ctx.save_for_backward(sample, lam)
    return sample

  @staticmethod
  def backward(ctx, grad_output):
    sample, lam = ctx.saved_tensors
    grad_lam = grad_output * (0.5 + sample / (2 * lam))
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