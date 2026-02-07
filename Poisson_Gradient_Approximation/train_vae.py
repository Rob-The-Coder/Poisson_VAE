import argparse

from torchinfo import summary

from utils import CustomPoissonSampling, CustomDataset, ELBO_Loss, VAE_Trainer, Model_Args
from vae import VAE

def parse_args():
  parser = argparse.ArgumentParser(description="VAE training script")

  # Path
  parser.add_argument("--path", type=str, required=False, default="/home/schifano/Documents/Thesis/Poisson_Gradient_Approximation/archive/", help="Path to images folder")
  parser.add_argument("--project_dir", type=str, required=False, default="/home/schifano/Documents/Thesis/Poisson_Gradient_Approximation/", help="Path to the project folder")

  # Hyperparameters
  parser.add_argument("--vae_filename", type=str, default="VAE.pt")
  parser.add_argument("--vae_checkpoint", type=str, default="VAE_checkpoint.pt")
  parser.add_argument("--height", type=int, default=64)
  parser.add_argument("--width", type=int, default=64)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--rescale", type=float, default=1e-2, help="RESCALE parameter")
  parser.add_argument("--lambda_poisson", type=float, default=10, help="LAMBDA parameter")
  parser.add_argument("--latent_dim", type=int, default=128)

  # Training - Hardware/Optimization
  parser.add_argument("--resume", action="store_true", default=False)
  parser.add_argument("--epochs_to_show_faces", type=int, default=10)
  parser.add_argument("--epochs_to_checkpoint", type=int, default=10)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--optimize", action="store_true", default=True, help="Enables JIT and AMP")
  parser.add_argument("--clip_gradients", action="store_false", dest="clip", help="Disables gradient clipping")

  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()

  model_args = Model_Args(vae_filename=args.vae_filename, checkpoint_filename=args.vae_checkpoint, project_dir=args.project_dir)

  train_loader, _ = CustomDataset.get_dataloaders(
    height = args.height,
    width = args.width,
    batch_size = args.batch_size,
    path = args.path
  )

  elbo_loss = ELBO_Loss()

  if args.resume:
    print("\nRecovering model from checkpoint...")

    trainer = VAE_Trainer.from_checkpoint(model_args)
    trainer.set_train_loader(train_loader)
    trainer.set_loss_function(elbo_loss)
    trainer.explain_checkpoint()
  else:
    print("\nInstantiating model and trainer...")

    vae = VAE(height=args.height, width=args.width, latent_dim=args.latent_dim, sampling=CustomPoissonSampling.apply)
    summary(vae, input_size=(train_loader.batch_size, 3, args.height, args.width))

    trainer = VAE_Trainer(
      vae=vae,
      loss_function=elbo_loss,
      train_loader=train_loader,
      create_optimizer=("AdamW", args.lr),
      gradient_clipping=args.clip,
      LAMBDA=args.lambda_poisson,
      RESCALE=args.rescale
    )

  try:
    print("\nStarting training...")

    trainer.train(
      model_args=model_args,
      EPOCHS=args.epochs,
      epochs_to_create_checkpoint=5,
      epochs_to_show_faces=1,
      optimize=args.optimize,
      download_model = False
    )
  except KeyboardInterrupt:
    print("\nTraining was interrupted. Saving a last checkpoint...")
    trainer.create_checkpoint(model_args=model_args)