import torch
import matplotlib.pyplot as plt
import torchvision
import argparse

from vae import VAE
from utils import Model_Args

def show_faces(faces, title: str = ""):
  plt.rcParams['figure.dpi'] = 200
  g = torchvision.utils.make_grid(faces, normalize=True, value_range=(-1, 1))
  plt.imshow(g.permute(1, 2, 0).detach().cpu().numpy())
  plt.axis("off")
  plt.title(title, fontsize=15)
  plt.show()

def parse_args():
  parser = argparse.ArgumentParser(description="VAE face generation script")

  # Path
  parser.add_argument("--path", type=str, required=False, default="/home/schifano/Documents/Thesis/Poisson_Gradient_Approximation/archive/", help="Path of the image folder")
  parser.add_argument("--project_dir", type=str, required=False, default="/home/schifano/Documents/Thesis/Poisson_Gradient_Approximation/", help="Path of the project folder")

  # parameters
  parser.add_argument("--num_faces", type=int, required=False, default=32, help="Number of faces to generate")
  parser.add_argument("--vae_filename", type=str, default="VAE.pt")
  parser.add_argument("--lambda_poisson", type=float, default=10, help="Parametro LAMBDA")
  parser.add_argument("--title", type=str, default="", help="Title of the generated plot")

  return parser.parse_args()

if __name__=="__main__":
  args = parse_args()

  model_args = Model_Args(vae_filename=args.vae_filename, project_dir=args.project_dir)

  vae = VAE.from_pretrained(model_args)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  faces = vae.generate_faces(num_faces=32, LAMBDA=args.lambda_poisson, device=device)
  show_faces(faces, "")