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
    grad_lam = grad_output * (0.5 + sample/(2*lam))
    return grad_lam


class ELBO_Loss():
  def compute_loss(self, target, generated, lam, lambda_, rescale_=1e-4):
    lambda_ = torch.tensor(lambda_)

    # rec_error = ((generated - target) ** 2).mean()
    rec_error = (generated - target).abs().mean()
    kl_div = (lam * (torch.log(lam) - torch.log(lambda_)) - lam + lambda_).mean()

    return (kl_div * rescale_), rec_error