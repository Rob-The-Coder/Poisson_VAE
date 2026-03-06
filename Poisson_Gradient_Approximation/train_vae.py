import argparse

from torchinfo import summary
from rich import print
from rich.console import Console
from rich.table import Table
from decouple import config
from pathlib import Path

from utils import CustomPoissonSampling, CelebA, ELBO_Loss, Model_Args
from vae import VAE, VAE_Trainer

def parse_args():
  parser = argparse.ArgumentParser(description="VAE training script")

  # Path
  parser.add_argument("--path", type=str, required=False, help="Path to images folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")
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
  parser.add_argument("--type", type=str, choices=["36M", "53M"], default="36M", help="Decide which version of the model to use. Defaults to 36M")
  parser.add_argument("--optimizer", type=str, choices=["AdamW", "Adam", "SGD"], default="AdamW", help="Decide which type of optimizer to use. Defaults to AdamW")
  parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint. If not used defaults to False")
  parser.add_argument("--epochs_to_checkpoint", type=int, default=10, help="Number of epochs to create a checkpoint. Defaults to 10")
  parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train. Defaults to 100")
  parser.add_argument("--optimize", action="store_true", default=True, help="Enables JIT and AMP. Defaults to True. Use --optimize False to disable")
  parser.add_argument("--clip_gradients", action="store_false", default=False, dest="clip", help="Enables gradient clipping. Defaults to False, use --clip_gradients True to enable")

  args = parser.parse_args()
  args.path = args.path or config("IMG_DIR", default=Path.cwd())
  args.project_dir = args.project_dir or config("PROJECT_DIR", default=Path.cwd())

  args.vae_filename = args.vae_filename or config("VAE_FILENAME", default="VAE.pt")
  args.vae_checkpoint = args.vae_checkpoint or config("VAE_CHECKPOINT", default="VAE_checkpoint.pt")
  return args

def print_args(args):
  console = Console()

  table = Table(title="VAE Training Configuration")

  table.add_column("Parameter", style="cyan")
  table.add_column("Value", style="magenta")

  for key, value in vars(args).items():
    table.add_row(key, str(value))

  console.print(table)

if __name__ == "__main__":
  # Parsing args from command line
  args = parse_args()

  # Printing args
  print_args(args)

  # Checking the existence of paths
  path = Path(args.path)
  project_dir = Path(args.project_dir)

  if not path.exists():
    print(f"[bold red][ERROR]: [/bold red] Path {path} not found!")
    exit(1)

  if not project_dir.exists():
    print(f"[bold red][ERROR]: [/bold red] Path {project_dir} not found!")
    exit(1)

  model_args = Model_Args(vae_filename=args.vae_filename, checkpoint_filename=args.vae_checkpoint, project_dir=project_dir)

  train_loader, _ = CelebA.get_dataloaders(
    height = args.height,
    width = args.width,
    batch_size = args.batch_size,
    path = path
  )

  elbo_loss = ELBO_Loss()

  if args.resume:
    print("\n[bold cyan][INFO]: [/bold cyan] Recovering model from checkpoint...")

    trainer = VAE_Trainer.from_checkpoint(model_args)
    trainer.set_train_loader(train_loader)
    trainer.set_loss_function(elbo_loss)
    trainer.explain_checkpoint()
  else:
    print("\n[bold cyan][INFO]: [/bold cyan] Instantiating model and trainer...")

    vae = VAE(height=args.height, width=args.width, latent_dim=args.latent_dim, sampling=CustomPoissonSampling.apply, model_type=args.type)

    print("\n[bold green][DEBUG]: [/bold green] Printing summary of model...")
    summary(vae, input_size=(train_loader.batch_size, 3, args.height, args.width))

    trainer = VAE_Trainer(
      vae=vae,
      loss_function=elbo_loss,
      train_loader=train_loader,
      create_optimizer=(args.optimizer, args.lr),
      gradient_clipping=args.clip,
      LAMBDA=args.lam,
      RESCALE=args.rescale
    )

  try:
    print("\n[bold cyan][INFO]: [/bold cyan] Starting training...")

    trainer.train(
      model_args=model_args,
      EPOCHS=args.epochs,
      epochs_to_create_checkpoint=args.epochs_to_checkpoint,
      optimize=args.optimize,
    )
  except KeyboardInterrupt:
    print("\n[bold red][ERROR]: [/bold red] Training was interrupted. Saving a last checkpoint...")
    trainer.create_checkpoint(model_args=model_args)