import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import math
import umap
import argparse

from rich import print
from rich.console import Console
from rich.table import Table
from decouple import config
from pathlib import Path

from utils import CelebA
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
  plt.title(title, fontsize=10)
  plt.show()

def parse_args():
  parser = argparse.ArgumentParser(description="VAE face generation script")

  # Path
  parser.add_argument("--path", type=str, required=False,
                      help="Path to images folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")
  parser.add_argument("--project_dir", type=str, required=False,
                      help="Path to the project folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")

  # File handling
  parser.add_argument("--vae_filename", type=str, required=False,
                      help="Name of the generated VAE file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE.pt")
  parser.add_argument("--vae_checkpoint", type=str, required=False,
                      help="Name of the generated training checkpoint file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE_checkpoint.pt")

  # parameters
  parser.add_argument("--num_faces", type=int, required=False, default=36, help="Number of faces to generate. Defaults to 36")
  parser.add_argument("--lambda_poisson", type=float, default=10, help="LAMBDA parameter. Defaults to 10")
  parser.add_argument("--title", type=str, default="", help="Title of the generated plot. By default is set to a blank string")

  parser.add_argument("--interpolation", action="store_true", default=True, help="Whether to interpolate images. Defaults to True")
  parser.add_argument("--height", type=int, default=64, help="Height of the image. Defaults to 64")
  parser.add_argument("--width", type=int, default=64, help="Width of the image. Defaults to 64")
  parser.add_argument("--start", type=int, default=700, help="When --interpolation is set to true, this is the starting image. Defaults to 700")
  parser.add_argument("--end", type=int, default=900, help="When --interpolation is set to true, this is the ending image. Defaults to 900")

  parser.add_argument("--clusterization", action="store_true", default=True, help="Whether to compute clusterization. Defaults to True")
  parser.add_argument("--batch_size", type=int, default=512, help="Batch size. Defaults to 512")
  parser.add_argument("--num_samples", type=int, default=5000, help="Sample number used to compute clusterization. Defaults to 5000")

  args = parser.parse_args()
  args.path = args.path or config("IMG_DIR", default=Path.cwd())
  args.project_dir = args.project_dir or config("PROJECT_DIR", default=Path.cwd())

  args.vae_filename = args.vae_filename or config("VAE_FILENAME", default="VAE.pt")
  args.vae_checkpoint = args.vae_checkpoint or config("VAE_CHECKPOINT", default="VAE_checkpoint.pt")
  return args

def print_args(args):
  console = Console()

  table = Table(title="VAE images generation configuration")

  table.add_column("Parameter", style="cyan")
  table.add_column("Value", style="magenta")

  for key, value in vars(args).items():
    table.add_row(key, str(value))

  console.print(table)

if __name__=="__main__":
  # Parsing args from command line
  args = parse_args()

  # Printing args
  print_args(args)

  # Checking the existence of paths
  project_dir = Path(args.project_dir)
  path = Path(args.path)

  if not project_dir.exists():
    print(f"[bold red][ERROR]: [/bold red] Path {project_dir} not found!")
    exit(1)

  if not path.exists():
    print(f"[bold red][ERROR]: [/bold red] Path {path} not found!")
    exit(1)

  model_args = Model_Args(vae_filename=args.vae_filename, checkpoint_filename="", project_dir=project_dir)

  vae = VAE.from_pretrained(model_args)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  print("\n[bold cyan][INFO]: [/bold cyan] Generating faces...")
  faces = vae.generate_faces(num_faces=args.num_faces, LAMBDA=args.lambda_poisson, device=device)
  show_faces(faces, args.title)

  if args.interpolation:
    print("\n[bold cyan][INFO]: [/bold cyan] Generating interpolation image...")
    train_set = CelebA.get_train_set(args.height, args.width, path)

    x0 = train_set[args.start][0][None].to(device)
    x1 = train_set[args.end][0][None].to(device)

    z0 = vae.encode(x0)
    z1 = vae.encode(x1)
    beta = torch.linspace(0, 1, 8, device=device).view(-1, 1)

    z = (1 - beta) * z0 + beta * z1
    y = vae.decode(z)
    show_faces(y)

  if args.clusterization:
    vae.eval()
    latents = []

    attr_df = CelebA.get_attributes(path)

    _, valid_loader = CelebA.get_dataloaders(
      height=args.height,
      width=args.width,
      batch_size=args.batch_size,
      path=path
    )

    # Computing the latents
    with torch.no_grad():
      for i, (batch, _) in enumerate(valid_loader):
        z = vae.encode(batch.to(device))
        latents.append(z.cpu().numpy())
        if len(np.concatenate(latents)) >= args.num_samples:
          break
    z_combined = np.concatenate(latents)[:args.num_samples]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Computing embedding for all attributes
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(z_combined)

    attributes_to_test = ['Male', 'Smiling', 'Eyeglasses', 'Young']
    for i, attr in enumerate(attributes_to_test):
      labels = attr_df[attr].values[:args.num_samples]
      sc = axes[i].scatter(embedding[:, 0], embedding[:, 1], c=labels, s=5, alpha=0.5, cmap='coolwarm')
      axes[i].set_title(f"Attribute: {attr}")
      plt.colorbar(sc, ax=axes[i])

    plt.tight_layout()
    plt.show()