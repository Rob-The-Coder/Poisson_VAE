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
from dataclasses import dataclass

from utils import CelebA
from vae import VAE
from core.model_args import ModelArgs

@dataclass
class GenerationArgs:
  images_dir: str
  project_dir: str
  vae_filename: str
  num_faces: int = 36
  lam: int = 10
  title: str = ""
  interpolation: bool = True
  height: int = 64
  width: int = 64
  start: int = 6000
  end: int = 80000
  clusterization: bool = True
  batch_size: int = 512
  num_samples: int = 5000
  latent_analysis: bool = True
  start_traversal: int = 0
  end_traversal: int = 50
  steps_number_traversal: int = 8
  threshold: float = 0.05
  start_alpha: int = -5
  end_alpha: int = 5
  steps_number_attributes: int = 8

def get_faces(faces, title: str = "", nrow: int = 8, square: bool = True):
  plt.rcParams['figure.dpi'] = 200
  fig, ax = plt.subplots()

  if square:
    nrow = math.isqrt(faces.size()[0] - 1) + 1

  g = torchvision.utils.make_grid(faces, nrow=nrow, normalize=True, value_range=(-1, 1))
  ax.imshow(g.permute(1, 2, 0).detach().cpu().numpy())
  ax.axis("off")
  ax.set_title(title, fontsize=10)
  return fig

def generate(args_dict: GenerationArgs):
  figures = []
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Checking the existence of paths
  project_dir = Path(args_dict.project_dir)
  images_dir = Path(args_dict.images_dir)

  if not project_dir.exists():
    print(f"[bold red][ERROR]: [/bold red] Path {project_dir} not found!")
    exit(1)

  if not images_dir.exists():
    print(f"[bold red][ERROR]: [/bold red] Path {images_dir} not found!")
    exit(1)

  model_args = ModelArgs(vae_filename=args_dict.vae_filename, checkpoint_filename="", project_dir=project_dir)
  vae = VAE.from_pretrained(model_args)
  vae.eval()

  print("\n[bold cyan][INFO]: [/bold cyan] Generating faces...")
  faces = vae.generate_faces(num_faces=args_dict.num_faces, device=device, LAMBDA=args_dict.lam)
  #figures.append(get_faces(faces, args_dict.title))
  yield get_faces(faces, args_dict.title)

  if args_dict.interpolation:
    print("\n[bold cyan][INFO]: [/bold cyan] Generating interpolation image...")
    train_set = CelebA.get_train_set(args_dict.height, args_dict.width, images_dir)

    x0 = train_set[args_dict.start][0][None].to(device)
    x1 = train_set[args_dict.end][0][None].to(device)

    z0 = vae(x0).p1
    z1 = vae(x1).p1
    beta = torch.linspace(0, 1, 8, device=device).view(-1, 1)

    z = (1 - beta) * z0 + beta * z1
    y = vae.decode(z)
    #figures.append(get_faces(y, "Interpolation", square=False))
    yield get_faces(y, "Interpolation", square=False)

  if args_dict.clusterization:
    print("\n[bold cyan][INFO]: [/bold cyan] Performing clusterization...")
    latents = []

    valid_set = CelebA.get_valid_set(args_dict.height, args_dict.width, images_dir)
    attr_df = CelebA.get_attributes(images_dir)
    attr_df = attr_df[attr_df['image_id'].isin(valid_set.img_partition)]

    _, valid_loader = CelebA.get_dataloaders(
      height=args_dict.height,
      width=args_dict.width,
      batch_size=args_dict.batch_size,
      images_dir=images_dir
    )

    # Computing the latents
    with torch.no_grad():
      for i, (batch, _) in enumerate(valid_loader):
        z = vae(batch.to(device)).p1
        latents.append(z.cpu().numpy())
        if len(np.concatenate(latents)) >= args_dict.num_samples:
          break
    z_combined = np.concatenate(latents)[:args_dict.num_samples]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Computing embedding for all attributes
    reducer = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.0)
    embedding = reducer.fit_transform(z_combined)

    attributes_to_test = ['Male', 'Smiling', 'Blond_Hair', 'Young']
    for i, attr in enumerate(attributes_to_test):
      labels = attr_df[attr].values[:args_dict.num_samples]
      sc = axes[i].scatter(embedding[:, 0], embedding[:, 1], c=labels, s=2, alpha=0.5, cmap='coolwarm')
      axes[i].set_title(f"Attribute: {attr}")
      axes[i].set_xlabel("U1")
      axes[i].set_ylabel("U2")
      axes[i].set_xticks([])
      axes[i].set_yticks([])
      plt.colorbar(sc, ax=axes[i])

    plt.tight_layout()
    #figures.append(fig)
    yield fig

  if args_dict.latent_analysis:
    print("\n[bold cyan][INFO]: [/bold cyan] Performing latent space analysis...")
    _, valid_loader = CelebA.get_dataloaders(
      height=args_dict.height,
      width=args_dict.width,
      batch_size=args_dict.batch_size,
      images_dir=images_dir
    )

    # Latent traversal
    # For each dimension of the latent space, starting from a base image (taken from the validation loader) progressively larger value are replaced
    all_rows = []
    steps = torch.linspace(args_dict.start_traversal, args_dict.end_traversal, args_dict.steps_number_traversal).float().to(device)
    x, _ = next(iter(valid_loader))
    base_z = vae(x[0:1].to(device)).p1

    for dim in range(vae.latent_dim):
      z_strip = base_z.repeat(args_dict.steps_number_traversal, 1).float().to(device)
      for i, step in enumerate(steps):
        z_strip[i, dim] = step

      # Checking difference to see whether the traversal is significant
      z_diff = base_z.repeat(2, 1)
      z_diff[0, dim] = steps[0]
      z_diff[1, dim] = steps[-1]

      # Image generation
      with torch.no_grad():
        decoded = vae.decode(z_strip)
        decoded_diffs = vae.decode(z_diff)

        diff = torch.abs(decoded[0] - decoded[args_dict.steps_number_traversal - 1]).mean().item()
        if diff > args_dict.threshold:
          all_rows.append(decoded)

    save_path = project_dir / "analysis" / f"traversal_{args_dict.vae_filename}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid = torch.cat(all_rows, dim=0)
    torchvision.utils.save_image(grid, save_path, nrow=args_dict.steps_number_traversal, normalize=True, value_range=(-1, 1))

    # Attribute directions
    def get_mean_latent(indices):
      latents = []
      for img_idx in indices:
        img = train_set[img_idx][0][None]
        with torch.no_grad():
          lam = vae(img).p1
          latents.append(lam)
      return torch.cat(latents).mean(dim=0)

    attr_df = CelebA.get_attributes(images_dir)
    train_set = CelebA.get_train_set(args_dict.height, args_dict.width, images_dir)

    attributes = ['Male', 'Smiling', 'Eyeglasses', 'Young', 'Bald', 'Blond_Hair', 'Oval_Face', 'Wearing_Hat', 'Pointy_Nose']
    manipulated_imgs = []
    # For each attribute the positive and "negative" attribute vector is computed by taking the mean of a number of samples. Then the actual
    # vector direction is computed as the difference between the two.
    for attr_name in attributes:
      pos_idx, neg_idx = train_set.get_train_idx(attr_df, attr_name)
      z_pos_mean = get_mean_latent(pos_idx)
      z_neg_mean = get_mean_latent(neg_idx)

      attr_vector = z_pos_mean - z_neg_mean

      alphas = torch.linspace(args_dict.start_alpha, args_dict.end_alpha, args_dict.steps_number_attributes).float().to(device)
      z_strip = base_z.repeat(args_dict.steps_number_attributes, 1).float().to(device)

      for i, alpha in enumerate(alphas):
        z_strip[i, :] = base_z + alpha * attr_vector

      with torch.no_grad():
        decoded = vae.decode(z_strip)
        manipulated_imgs.append(decoded)

    grid = torch.cat(manipulated_imgs, dim=0)
    fig = get_faces(grid, title="Attributes direction applied to image", nrow=args_dict.steps_number_attributes, square=False)
    fig.savefig(project_dir / "analysis" / f"attributes_manipulation_{args_dict.vae_filename}.png")
    #figures.append(fig)
    yield fig

    # Lambda distribution
    # Check lambda distribution on real sample gotten from the validation loader against the "true" distribution
    fig, axes = plt.subplots(1, 2, figsize=(25, 10))
    axes = axes.flatten()

    lambdas = []
    with torch.no_grad():
      for x, _ in valid_loader:
        lam = vae(x.to(device)).p1
        lambdas.append(lam.cpu())
        if len(lambdas) > 10: break
    lambdas = torch.cat(lambdas, dim=0).numpy()
    lambdas = np.floor(lambdas)

    z = torch.poisson(torch.full((args_dict.batch_size * 10, vae.latent_dim), args_dict.lam, device=device, dtype=torch.float32))
    distributions = [lambdas.flatten(), z.flatten()]

    for i, distribution in enumerate(distributions):
      axes[i].hist(distribution, bins=250, color='skyblue', edgecolor='black')
      axes[i].set_title(f"$\lambda$ distribution {"in the latent space" if i==0 else "from torch.poisson"}")
      axes[i].set_xlabel("$\lambda$")
      axes[i].set_ylabel("Frequency")
      axes[i].set_xlim(left=0, right=20)
      axes[i].set_xticks(range(21))
      axes[i].grid(True, alpha=0.3)

    fig.savefig(project_dir / "analysis" / "lambda_distribution.png")
    #figures.append(fig)
    yield fig

    # Checking for "dead" dimensions (es. mean lambda < 0.1)
    mean_lambdas = np.mean(lambdas, axis=0)
    dead_dims = np.sum(mean_lambdas < 0.1)
    print(f"Latent dimensions 'dead' (λ < 0.1): {dead_dims}/{vae.latent_dim}")

    # Style mixing
    lam_a = vae(train_set[603][0][None].to(device)).p1
    lam_b = vae(train_set[80000][0][None].to(device)).p1

    mid = vae.latent_dim // 2
    mix_1 = torch.cat([lam_a[:, :mid], lam_b[:, mid:]], dim=1)
    mix_2 = torch.cat([lam_b[:, :mid], lam_a[:, mid:]], dim=1)

    res_a = vae.decode(lam_a)
    res_b = vae.decode(lam_b)
    res_mix1 = vae.decode(mix_1)
    res_mix2 = vae.decode(mix_2)

    grid = torch.cat([train_set[603][0][None], res_a, res_mix1, res_mix2, res_b, train_set[80000][0][None]], dim=0)
    fig = get_faces(grid, title="Style mixing", nrow=6, square=False)
    #figures.append(fig)
    yield get_faces(grid, title="Style mixing", nrow=6, square=False)

    # Pure reconstruction to test decoder capabilities
    x, _ = next(iter(valid_loader))
    y = vae(x[0:36].to(device)).reconstruction
    # figures.append(get_faces(y, "Pure reconstruction", square=True))
    yield get_faces(y, "Pure reconstruction", square=True)

def parse_args():
  parser = argparse.ArgumentParser(description="VAE face generation script")

  # Path
  parser.add_argument("--images_dir", type=str, required=False, help="Path to images folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")
  parser.add_argument("--project_dir", type=str, required=False, help="Path to the project folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")

  # File handling
  parser.add_argument("--vae_filename", type=str, required=False, help="Name of the generated VAE file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE.pt")
  parser.add_argument("--vae_checkpoint", type=str, required=False, help="Name of the generated training checkpoint file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE_checkpoint.pt")

  # parameters
  parser.add_argument("--num_faces", type=int, required=False, default=36, help="Number of faces to generate. Defaults to 36")
  parser.add_argument("--lam", type=float, default=10, help="LAMBDA parameter. Defaults to 10")
  parser.add_argument("--title", type=str, default="", help="Title of the generated plot. By default is set to a blank string")

  parser.add_argument("--interpolation", type=bool, default=True, help="Whether to interpolate images. Defaults to True")
  parser.add_argument("--height", type=int, default=64, help="Height of the image. Defaults to 64")
  parser.add_argument("--width", type=int, default=64, help="Width of the image. Defaults to 64")
  parser.add_argument("--start", type=int, default=700, help="When --interpolation is set to true, this is the starting image. Defaults to 6000")
  parser.add_argument("--end", type=int, default=900, help="When --interpolation is set to true, this is the ending image. Defaults to 80000")

  parser.add_argument("--clusterization", type=bool, default=True, help="Whether to compute clusterization. Defaults to True")
  parser.add_argument("--batch_size", type=int, default=512, help="Batch size used to compute clusters. Defaults to 512")
  parser.add_argument("--num_samples", type=int, default=5000, help="Samples number used to compute clusterization. Defaults to 5000")

  parser.add_argument("--latent_analysis", type=bool, default=True, help="Whether to perform latent space analysis. Defaults to True")
  parser.add_argument("--start_traversal", type=int, default=0, help="Starting traversal value used to modify the images generated by the model. Defaults to 0")
  parser.add_argument("--end_traversal", type=int, default=50, help="Ending traversal value used to modify the images generated by the model. Defaults to 50")
  parser.add_argument("--steps_number_traversal", type=int, default=8, help="Steps number used to traverse the latent space. Defaults to 8")
  parser.add_argument("--threshold", type=float, default=0.05, help="Threshold to accept whether a dimension is significant. Defaults to 0.05")
  parser.add_argument("--start_alpha", type=int, default=-5, help="Starting alpha value used to apply the attribute. Defaults to -5")
  parser.add_argument("--end_alpha", type=int, default=5, help="Ending alpha value used to apply the attribute. Defaults to 5")
  parser.add_argument("--steps_number_attribute", type=int, default=8, help="Steps number used to apply attributes. Defaults to 8")

  args = parser.parse_args()
  args.images_dir = args.images_dir or config("IMG_DIR", default=Path.cwd())
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

  gen_args = GenerationArgs(**{k: v for k, v in vars(args).items() if k in GenerationArgs.__dataclass_fields__})

  for fig in generate(gen_args):
    fig.show()