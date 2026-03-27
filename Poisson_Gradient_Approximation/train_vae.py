import argparse
from typing import Callable

from torchinfo import summary
from rich.console import Console
from rich.table import Table
from decouple import config
from pathlib import Path
from dataclasses import dataclass, fields

from utils import CelebA
from core.model_args import ModelArgs
from vae import VAE, VAE_Trainer

@dataclass
class TrainingArgs:
  images_dir: str
  project_dir: str
  vae_filename: str
  vae_checkpoint: str
  height: int = 64
  width: int = 64
  batch_size: int = 128
  lr: float = 1e-4
  rescale: float = 1e-2
  lam: int = 10
  latent_dim: int = 128
  type: str = "36M"
  sampling: str = "PGA"
  optimizer: str = "AdamW"
  resume: bool = False
  epochs_to_checkpoint: int = 10
  epochs_to_monitor: int = 0
  epochs: int = 100
  optimize: bool = True
  clip_gradients: bool = False

def train(args_dict: TrainingArgs, callback: Callable = None):
  console = Console(record=True)
  table = Table(title="VAE Training Configuration")

  table.add_column("Parameter", style="cyan")
  table.add_column("Value", style="magenta")

  for field in fields(args_dict):
    table.add_row(field.name, str(getattr(args_dict, field.name)))

  console.print(table)

  # Checking the existence of paths
  images_dir = Path(args_dict.images_dir)
  project_dir = Path(args_dict.project_dir)

  if not images_dir.exists():
    console.print(f"[bold red][ERROR]: [/bold red] Path {images_dir} not found!")
    exit(1)

  if not project_dir.exists():
    console.print(f"[bold red][ERROR]: [/bold red] Path {project_dir} not found!")
    exit(1)

  model_args = ModelArgs(vae_filename=args_dict.vae_filename, checkpoint_filename=args_dict.vae_checkpoint, project_dir=project_dir)

  train_loader, _ = CelebA.get_dataloaders(
    height=args_dict.height,
    width=args_dict.width,
    batch_size=args_dict.batch_size,
    images_dir=images_dir
  )

  if args_dict.resume:
    console.print("\n[bold cyan][INFO]: [/bold cyan] Recovering model from checkpoint...")

    trainer = VAE_Trainer.from_checkpoint(model_args)
    trainer.set_train_loader(train_loader)
    trainer.explain_checkpoint()
  else:
    console.print("\n[bold cyan][INFO]: [/bold cyan] Instantiating model and trainer...")
    vae = VAE(height=args_dict.height, width=args_dict.width, latent_dim=args_dict.latent_dim, sampling=args_dict.sampling, model_type=args_dict.type)

    console.print("\n[bold green][DEBUG]: [/bold green] Printing summary of encoder...")
    console.print(str(summary(vae.encoder, input_size=(train_loader.batch_size, 3, args_dict.height, args_dict.width))))

    console.print("\n[bold green][DEBUG]: [/bold green] Printing summary of decoder...")
    console.print(str(summary(vae.decoder, input_size=(train_loader.batch_size, vae.latent_dim))))

    trainer = VAE_Trainer(
      vae=vae,
      train_loader=train_loader,
      create_optimizer=(args_dict.optimizer, args_dict.lr),
      gradient_clipping=args_dict.clip_gradients,
      LAMBDA=args_dict.lam,
      RESCALE=args_dict.rescale
    )

  try:
    console.print("\n[bold cyan][INFO]: [/bold cyan] Starting training...")

    trainer.train(
      model_args=model_args,
      EPOCHS=args_dict.epochs,
      epochs_to_create_checkpoint=args_dict.epochs_to_checkpoint,
      epochs_to_monitor=args_dict.epochs_to_monitor,
      optimize=args_dict.optimize,
      callback=callback
    )
  except KeyboardInterrupt:
    console.print("\n[bold red][ERROR]: [/bold red] Training was interrupted. Saving a last checkpoint...")
    trainer.create_checkpoint(model_args=model_args)
  finally:
    monitor_path = model_args.project_dir / "training" / ("monitor_" + model_args.vae_filename)
    monitor_path.mkdir(parents=True, exist_ok=True)

    console.save_html((monitor_path / "training_log.html").absolute().as_posix())

def parse_args():
  parser = argparse.ArgumentParser(description="VAE training script")

  # Path
  parser.add_argument("--images_dir", type=str, required=False, help="Path to images folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")
  parser.add_argument("--project_dir", type=str, required=False, help="Path to the project folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")

  # File handling
  parser.add_argument("--vae_filename", type=str, required=False, help="Name of the generated VAE file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE.pt")
  parser.add_argument("--vae_checkpoint", type=str, required=False, help="Name of the generated training checkpoint file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE_checkpoint.pt")

  # Hyperparameters
  parser.add_argument("--height", type=int, default=64, help="Height of the image. Defaults to 64")
  parser.add_argument("--width", type=int, default=64, help="Width of the image. Defaults to 64")
  parser.add_argument("--batch_size", type=int, default=128, help="Batch size. Defaults to 128")
  parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate. Defaults to 1e-4")
  parser.add_argument("--rescale", type=float, default=1e-2, help="RESCALE parameter. Defaults to 1e-2")
  parser.add_argument("--lam", type=float, default=10, help="LAMBDA parameter. Defaults to 10")
  parser.add_argument("--latent_dim", type=int, default=128, help="Dimension of the latent space. Defaults to 128")

  # Training - Hardware/Optimization
  parser.add_argument("--type", type=str, choices=["36M", "53M", "60M"], default="36M", help="Decide which version of the model to use. Defaults to 36M")
  parser.add_argument("--sampling", type=str, choices=["PGA", "GRT"], default="PGA", help="Decide which sampling strategy to adopt. Defaults to PGA")
  parser.add_argument("--optimizer", type=str, choices=["AdamW", "Adam", "SGD"], default="AdamW", help="Decide which type of optimizer to use. Defaults to AdamW")
  parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint. If not used defaults to False")
  parser.add_argument("--epochs_to_checkpoint", type=int, default=10, help="Number of epochs to create a checkpoint. Defaults to 10")
  parser.add_argument("--epochs_to_monitor", type=int, default=0, help="Numb128er of epochs to monitor the training process. Defaults to 0")
  parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train. Defaults to 100")
  parser.add_argument("--optimize", type=bool, default=True, help="Enables JIT and AMP. Defaults to True. Use --optimize False to disable")
  parser.add_argument("--clip_gradients", action="store_false", default=False, dest="clip_gradients", help="Enables gradient clipping. Defaults to False, use --clip_gradients True to enable")

  args = parser.parse_args()
  args.images_dir = args.images_dir or config("IMG_DIR", default=Path.cwd())
  args.project_dir = args.project_dir or config("PROJECT_DIR", default=Path.cwd())

  args.vae_filename = args.vae_filename or config("VAE_FILENAME", default="VAE.pt")
  args.vae_checkpoint = args.vae_checkpoint or config("VAE_CHECKPOINT", default="VAE_checkpoint.pt")
  return args

if __name__ == "__main__":
  # Parsing args from command line
  args = parse_args()

  train_args = TrainingArgs(**{k: v for k, v in vars(args).items() if k in TrainingArgs.__dataclass_fields__})
  train(train_args)