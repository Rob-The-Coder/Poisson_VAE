import torch
import matplotlib.pyplot as plt
import torchvision
import math
import argparse

from Poisson_Gradient_Approximation.utils import CustomDataset
from vae import VAE
from utils import Model_Args

def show_faces(faces, title: str = ""):
  plt.rcParams['figure.dpi'] = 200

  nrow = 8
  if faces.size()[0] > 8:
    nrow = math.isqrt(faces.size()[0]-1) + 1

  g = torchvision.utils.make_grid(faces, nrow=nrow, normalize=True, value_range=(-1, 1))
  plt.imshow(g.permute(1, 2, 0).detach().cpu().numpy())
  plt.axis("off")
  plt.title(title, fontsize=15)
  plt.show()

def parse_args():
  parser = argparse.ArgumentParser(description="VAE face generation script")

  # Path
  parser.add_argument("--path", type=str, required=False, default="/home/schifano/Documents/Thesis/Poisson_Gradient_Approximation/archive/", help="Path to the images folder")
  parser.add_argument("--project_dir", type=str, required=False, default="/home/schifano/Documents/Thesis/Poisson_Gradient_Approximation/", help="Path to the project folder")

  # parameters
  parser.add_argument("--num_faces", type=int, required=False, default=32, help="Number of faces to generate")
  parser.add_argument("--vae_filename", type=str, default="VAE.pt")
  parser.add_argument("--lambda_poisson", type=float, default=10, help="LAMBDA parameter")
  parser.add_argument("--title", type=str, default="", help="Title of the generated plot")
  parser.add_argument("--interpolation", action="store_true", default=True, help="Whether to interpolate images")
  parser.add_argument("--height", type=int, default=64)
  parser.add_argument("--width", type=int, default=64)
  parser.add_argument("--start", type=int, default=700)
  parser.add_argument("--end", type=int, default=900)

  return parser.parse_args()

if __name__=="__main__":
  args = parse_args()

  model_args = Model_Args(vae_filename=args.vae_filename, checkpoint_filename="", project_dir=args.project_dir)

  vae = VAE.from_pretrained(model_args)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  faces = vae.generate_faces(num_faces=args.num_faces, LAMBDA=args.lambda_poisson, device=device)
  show_faces(faces, "")

  if args.interpolation:
    train_set = CustomDataset.get_train_set(args.height, args.width, args.path)

    x0 = train_set[args.start][0][None].to(device)
    x1 = train_set[args.end][0][None].to(device)

    z0 = vae.encode(x0)
    z1 = vae.encode(x1)
    beta = torch.linspace(0, 1, 8, device=device).view(-1, 1)

    z = (1 - beta) * z0 + beta * z1
    y = vae.decode(z)
    show_faces(y)