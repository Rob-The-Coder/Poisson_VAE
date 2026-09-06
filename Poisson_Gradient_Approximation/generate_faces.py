import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import math
import umap
import argparse

from matplotlib.lines import Line2D
from narwhals.stable import v1
from rich import print
from rich.console import Console
from rich.table import Table
from decouple import config
from pathlib import Path
from dataclasses import dataclass

from torch.utils.data import Subset, DataLoader
from torcheval.metrics import FrechetInceptionDistance
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn

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
    yield get_faces(y, "Interpolation", square=False)

  if args_dict.clusterization:
    print("\n[bold cyan][INFO]: [/bold cyan] Performing clusterization...")

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
    latents = []
    with torch.no_grad():
      for i, (batch, _) in enumerate(valid_loader):
        z = vae(batch.to(device)).p1
        latents.append(z.cpu().numpy())
        if len(np.concatenate(latents)) >= args_dict.num_samples:
          break
    z_combined = np.concatenate(latents)[:args_dict.num_samples]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    axes2 = axes2.flatten()

    # Computing embedding for all attributes
    reducer = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.0)
    embedding = reducer.fit_transform(z_combined)
    '''
    attr_labels = {
      'Male': ('Female', 'Male'),
      'Smiling': ('Not Smiling', 'Smiling'),
      'Blond_Hair': ('Not Blond', 'Blond Hair'),
      'Young': ('Not Young', 'Young'),
    }

    attributes_to_test = ['Male', 'Smiling', 'Blond_Hair', 'Young']
    for i, attr in enumerate(attributes_to_test):
      x = embedding[:, 0]
      y = embedding[:, 1]
      labels = attr_df[attr].values[:args_dict.num_samples]

      x_min, x_max = np.percentile(x, [10, 90])
      y_min, y_max = np.percentile(y, [10, 90])
      pad_x = (x_max - x_min) * 0.1
      pad_y = (y_max - y_min) * 0.1

      colors = np.where(labels==1, 'red', 'blue')
      neg_label, pos_label = attr_labels[attr]

      legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, label=neg_label),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label=pos_label),
      ]

      axes[i].scatter(x, y, c=colors, s=2, alpha=0.5)
      axes[i].set_title(f"Attribute: {attr}")
      axes[i].set_xlabel("U1")
      axes[i].set_ylabel("U2")
      axes[i].set_xticks([])
      axes[i].set_yticks([])
      axes[i].set_xlim(x_min - pad_x, x_max + pad_x)
      axes[i].set_ylim(y_min - pad_y, y_max + pad_y)
      axes[i].legend(handles=legend_elements, loc='upper right', markerscale=1)

      axes2[i].scatter(embedding[:, 0], embedding[:, 1], c=colors, s=2, alpha=0.5)
      axes2[i].set_title(f"Attribute: {attr}")
      axes2[i].set_xlabel("U1")
      axes2[i].set_ylabel("U2")
      axes2[i].set_xticks([])
      axes2[i].set_yticks([])
      axes2[i].legend(handles=legend_elements, loc='upper right', markerscale=1)

    plt.tight_layout()
    yield fig
    yield fig2
    '''

    # --- [Configurazione Palette a 4 Colori] ---
    colors_palette = {
      'Male': '#1f77b4',  # Blu
      'Smiling': '#ff7f0e',  # Arancione
      'Blond_Hair': '#2ca02c',  # Verde
      'Young': '#d62728'  # Rosso
    }
    color_default = '#e0e0e0'  # Grigio per i punti senza nessuno dei 4 attributi

    attributes_to_test = ['Male', 'Smiling', 'Blond_Hair', 'Young']

    # 1. Calcoliamo la frequenza reale di ogni attributo nel tuo sample corrente
    # Contiamo quanti 1 ci sono per ogni colonna
    counts = {attr: attr_df[attr].values[:args_dict.num_samples].sum() for attr in attributes_to_test}

    # 2. Ordiniamo gli attributi dal PIÙ COMUNE al PIÙ RARO
    # In questo modo il più raro viene disegnato per ultimo e sovrascrive gli altri, rimanendo visibile
    priority_order = sorted(attributes_to_test, key=lambda x: counts[x], reverse=True)

    # Stampo l'ordine calcolato in console (comodo per il tuo debugging)
    print(f"Ordine di disegno (dal comune al raro): {priority_order}")

    # 3. Assegnazione dinamica dei colori
    point_colors = np.full(args_dict.num_samples, color_default, dtype=object)
    for attr in priority_order:
      mask = attr_df[attr].values[:args_dict.num_samples]==1
      point_colors[mask] = colors_palette[attr]

    # --- [4. Creazione del Grafico Singolo] ---
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

    x = embedding[:, 0]
    y = embedding[:, 1]

    # Disegniamo prima lo sfondo grigio e poi i punti colorati per dare massima definizione
    bg_mask = (point_colors==color_default)
    fg_mask = ~bg_mask

    ax.scatter(x[bg_mask], y[bg_mask], c=color_default, s=3, alpha=0.15, label='Altri Attributi')
    ax.scatter(x[fg_mask], y[fg_mask], c=point_colors[fg_mask], s=5, alpha=0.75)

    # Creazione della legenda manuale seguendo l'ordine di priorità visiva
    legend_elements = [
      Line2D([0], [0], marker='o', color='w', markerfacecolor=colors_palette[attr], markersize=9, label=attr)
      for attr in priority_order
    ]
    legend_elements.append(
      Line2D([0], [0], marker='o', color='w', markerfacecolor=color_default, markersize=9, label='Altri'))

    # Pulizia assi e Zoom ottimizzato con percentili per escludere gli outlier di UMAP
    ax.set_title("Mappa dello Spazio Latente per Attributo Dominante (CelebA)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("U1", fontsize=12)
    ax.set_ylabel("U2", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, shadow=True)

    plt.tight_layout()
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

        diff = torch.abs(decoded[0] - decoded[args_dict.steps_number_traversal - 1]).mean().item()
        if diff > args_dict.threshold:
          all_rows.append(decoded)

    save_path = project_dir / "analysis" / f"traversal_{args_dict.vae_filename}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid = torch.cat(all_rows, dim=0)
    torchvision.utils.save_image(grid, save_path, nrow=args_dict.steps_number_traversal, normalize=True, value_range=(-1, 1))

    # Attribute directions
    attr_df = CelebA.get_attributes(images_dir)
    valid_set = CelebA.get_valid_set(args_dict.height, args_dict.width, images_dir)

    def get_mean_latent(indices):
      subset = Subset(valid_set, indices)
      loader = DataLoader(subset, batch_size=256, shuffle=False, num_workers=2)

      latents_sum = None
      total_samples = 0
      with torch.no_grad():
        for batch in loader:
          imgs = batch[0].to(device)
          lam = vae(imgs).p1

          if latents_sum is None:
            latents_sum = lam.sum(dim=0)
          else:
            latents_sum += lam.sum(dim=0)

          total_samples += imgs.size(0)

      return latents_sum / total_samples

    attributes = ['Male', 'Smiling', 'Eyeglasses', 'Young', 'Bald', 'Blond_Hair', 'No_Beard']
    vectors = {}
    manipulated_imgs = []
    # For each attribute the "positive" and "negative" attribute vector is computed by taking the mean of a number of samples. Then the actual
    # vector direction is computed as the difference between the two.
    #vectors = torch.load("analysis/vectors.pt")
    for attr_name in attributes:
      print(f"Computing vectors for attribute: {attr_name}")
      pos_idx, neg_idx = valid_set.get_partition_idx(attr_df, attr_name)
      z_pos_mean = get_mean_latent(pos_idx)
      z_neg_mean = get_mean_latent(neg_idx)

      attr_vector = z_pos_mean - z_neg_mean
      vectors[attr_name] = attr_vector
      print("OK")

    #torch.save(vectors, 'analysis/vectors.pt')

    for vector in vectors:
      alphas = torch.linspace(args_dict.start_alpha, args_dict.end_alpha, args_dict.steps_number_attributes).float().to(device)
      z_strip = base_z.repeat(args_dict.steps_number_attributes, 1).float().to(device)

      for i, alpha in enumerate(alphas):
        z_strip[i, :] = base_z + alpha * vectors[vector]

      with torch.no_grad():
        decoded = vae.decode(z_strip)
        manipulated_imgs.append(decoded)

    grid = torch.cat(manipulated_imgs, dim=0)
    fig = get_faces(grid, title="", nrow=args_dict.steps_number_attributes, square=False)
    fig.savefig(project_dir / "analysis" / f"attributes_manipulation_{args_dict.vae_filename}.png")
    yield fig

    '''
    for vector in vectors:
      z = torch.poisson(torch.full((args_dict.num_faces, vae.latent_dim), args_dict.lam, device=device, dtype=torch.float32))
      faces = vae.decode(z - 4 * vectors[vector])
      yield get_faces(faces, f"Generation with {vector} attribute applied, using $\\alpha = -4$")
    '''

    alpha = 3
    z = torch.poisson(torch.full((args_dict.steps_number_attributes, vae.latent_dim), args_dict.lam, device=device, dtype=torch.float32))
    #z = torch.randn(args_dict.steps_number_attributes, vae.latent_dim, device=device)
    row1 = z
    row2 = z + alpha * vectors["Smiling"]
    row3 = z + alpha * vectors["Young"]
    row4 = z + alpha * vectors["Smiling"] + alpha * vectors["Young"]

    with torch.no_grad():
      img1 = vae.decode(row1)
      img2 = vae.decode(row2)
      img3 = vae.decode(row3)
      img4 = vae.decode(row4)

    grid = torch.cat([img1, img2, img3, img4], dim=0)
    fig = get_faces(grid, title="", nrow=args_dict.steps_number_attributes,square=False)
    yield fig

    row2 = z + alpha * vectors["Male"]
    row3 = z + alpha * vectors["No_Beard"]
    row4 = z + alpha * vectors["Male"] + alpha * vectors["No_Beard"]

    with torch.no_grad():
      img1 = vae.decode(row1)
      img2 = vae.decode(row2)
      img3 = vae.decode(row3)
      img4 = vae.decode(row4)

    grid = torch.cat([img1, img2, img3, img4], dim=0)
    fig = get_faces(grid, title="", nrow=args_dict.steps_number_attributes, square=False)
    yield fig

    row2 = z + alpha * vectors["Blond_Hair"]
    row3 = z + alpha * vectors["Eyeglasses"]
    row4 = z + alpha * vectors["Blond_Hair"] + alpha * vectors["Eyeglasses"]

    with torch.no_grad():
      img1 = vae.decode(row1)
      img2 = vae.decode(row2)
      img3 = vae.decode(row3)
      img4 = vae.decode(row4)

    grid = torch.cat([img1, img2, img3, img4], dim=0)
    fig = get_faces(grid, title="", nrow=args_dict.steps_number_attributes, square=False)
    yield fig

    # Rescaling sensitivity
    def get_model(model_name):
      ma = ModelArgs(vae_filename=model_name, checkpoint_filename="", project_dir=project_dir)
      v = VAE.from_pretrained(ma)
      v.eval()

      return v

    z = torch.poisson(torch.full((4, vae.latent_dim), 4.0, device=device, dtype=torch.float32))
    models = ["VAE_checkpoint_60M_L4_300epochs_LR_RES5_LAT512.pt", "VAE_checkpoint_60M_L4_300epochs_LR_RES1_LAT512.pt", "VAE_checkpoint_60M_L4_300epochs_LR_RES5e-1_LAT512.pt", "VAE_checkpoint_60M_L4_300epochs_LR_RES1e-1_LAT512.pt", "VAE_checkpoint_60M_L4_300epochs_LR_RES5e-2_LAT512.pt"]
    imgs = []

    with torch.no_grad():
      for model_name in models:
        v = get_model(model_name)
        imgs.append(v.decode(z))

    grid = torch.cat(imgs, dim=0)
    fig = get_faces(grid, title="", nrow=4, square=False)
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
    lambdas = torch.cat(lambdas, dim=0).cpu().numpy()
    lambdas = np.round(lambdas)

    z = torch.poisson(torch.full((args_dict.batch_size * 10, vae.latent_dim), args_dict.lam, device=device, dtype=torch.float32))
    distributions = [lambdas.flatten(), z.cpu().flatten()]

    max_freq = max(
      np.histogram(distributions[0], bins=250)[0].max(),
      np.histogram(distributions[1], bins=250)[0].max()
    )

    for i, distribution in enumerate(distributions):
      axes[i].hist(distribution, bins=250, color='skyblue', edgecolor='black')
      axes[i].set_title(f"$\lambda$ distribution {'in the latent space' if i==0 else 'from torch.poisson'}")
      axes[i].set_xlabel("$\lambda$")
      axes[i].set_ylabel("Frequency")
      axes[i].set_xlim(left=0, right=20)
      axes[i].set_ylim(0, max_freq * 1.1)
      axes[i].set_xticks(range(21))
      axes[i].grid(True, alpha=0.3)

    fig.savefig(project_dir / "analysis" / "lambda_distribution.png")
    yield fig

    # Checking for "dead" dimensions (es. mean lambda < 0.1)
    mean_lambdas = np.mean(lambdas, axis=0)
    dead_dims = np.sum(mean_lambdas < 0.1)
    print(f"Latent dimensions 'dead' (λ < 0.1): {dead_dims}/{vae.latent_dim}")

  if True:
    # Style mixing
    train_set = CelebA.get_train_set(args_dict.height, args_dict.width, images_dir)


    idx_first = 603
    idx_last = 80000
    lam_a = vae(train_set[idx_first][0][None].to(device)).p1
    lam_b = vae(train_set[idx_last][0][None].to(device)).p1

    results = []
    split_points = [64, 128, 192, 256, 320, 384]
    # original and reconstruction
    results.append(train_set[idx_first][0][None].to(device))
    results.append(vae.decode(lam_a))

    # mixes
    for split in split_points:
      mixed_lam = torch.cat([lam_a[:, :split], lam_b[:, split:]], dim=1)
      results.append(vae.decode(mixed_lam))

    # reconstruction and original
    results.append(vae.decode(lam_b))
    results.append(train_set[idx_last][0][None].to(device))

    grid = torch.cat(results, dim=0)
    yield get_faces(grid, title="Style mixing", nrow=10, square=False)

  if True:
    _, valid_loader = CelebA.get_dataloaders(
      height=args_dict.height,
      width=args_dict.width,
      batch_size=args_dict.batch_size,
      images_dir=images_dir
    )

    # Pure reconstruction to test decoder capabilities
    x, _ = next(iter(valid_loader))
    y = vae(x[0:16].to(device)).reconstruction

    try:
      _ = y.ndim
      y = y
    except AttributeError:
      y = y[0]

    yield get_faces(y, "", square=True)

  if True:
    print("Computing FID...")
    train_set = CelebA.get_train_set(args_dict.height, args_dict.width, images_dir)

    train_loader_fid = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False, drop_last=True, num_workers=2)
    fid = FrechetInceptionDistance(device=device)

    MAX_IMAGES = 10000
    processed_images = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),  # Mostra "X di Y" immagini elaborate
        TimeRemainingColumn(),  # Mostra il tempo stimato rimanente
      ) as progress:
      fid_task = progress.add_task("[cyan]FID ", total=MAX_IMAGES)

      for batch, _ in train_loader_fid:
        current_batch_size = batch.size(0)
        if processed_images + current_batch_size > MAX_IMAGES:
          remaining = MAX_IMAGES - processed_images

          if remaining <= 0:
            break
          batch = batch[:remaining]
          current_batch_size = remaining

        # real images batch
        batch = batch.to(device)

        # generating images for comparison
        generated_images = vae.generate_faces(num_faces=current_batch_size, LAMBDA=args_dict.lam, device=device)

        # Denormalize images from [-1, 1] to [0, 1] since FID expects images in this format
        real_images = (batch + 1) / 2
        generated_images = (generated_images + 1) / 2

        fid.update(real_images, is_real=True)
        fid.update(generated_images, is_real=False)

        progress.update(fid_task, advance=current_batch_size)
        processed_images += current_batch_size
        if processed_images >= MAX_IMAGES:
          break

      # Computing FID score
      fid_score = fid.compute()
      print(f"FID Score: {fid_score.item()}")

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