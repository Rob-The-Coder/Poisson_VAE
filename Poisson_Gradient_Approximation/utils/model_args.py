from pathlib import Path

class Model_Args():
  def __init__(
      self,
      vae_filename: str,
      checkpoint_filename: str,
      project_dir: Path = Path("/home/schifano/Documents/Thesis/Poisson_Gradient_Approximation/")
  ):
    self.vae_filename = vae_filename
    self.checkpoint_filename = checkpoint_filename
    self.project_dir = project_dir