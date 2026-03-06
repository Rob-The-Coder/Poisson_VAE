import argparse
import streamlit as st

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
  if 'training_queue' not in st.session_state:
    st.session_state.training_queue = []

  st.title("VAE training GUI")

  with st.form("training_schedule"):
    st.subheader("Training scheduler")

    # Logic
    col1, col2, col3 = st.columns(3)
    with col1:
      resume = st.toggle("Resume training", value=False, help="Resume training from checkpoint. If not used defaults to False")
    with col2:
      optimize = st.toggle("Use optimization", value=True, help="Enables JIT and AMP. Defaults to True")
    with col3:
      clip_gradients = st.toggle("Gradient clipping", value=False, help="Enables gradient clipping. Defaults to False")

    # Path and file handling
    col1, col2 = st.columns(2)
    with col1:
      path = st.text_input("Image folder path", value=None, help="Path to images folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")
      vae_filename = st.text_input("VAE filename", value=None, help="Name of the generated VAE file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE.pt")
    with col2:
      project_dir = st.text_input("Project directory", value=None, help="Path to the project folder, if not specified will use the directory specified in the .env file. If both are not specified it will default to the current directory")
      vae_checkpoint = st.text_input("VAE checkpoint", value=None, help="Name of the generated training checkpoint file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE_checkpoint.pt")

    # Hyperparameters
    col1, col2, col3 = st.columns(3)
    with col1:
      height = st.number_input("Height", value=64, help="Height of the image. Defaults to 64")
    with col2:
      width = st.number_input("Width", value=64, help="Width of the image. Defaults to 64")
    with col3:
      batch_size = st.number_input("Batch_size", value=128, help="Batch size. Defaults to 128")

    col1, col2 = st.columns(2)
    with col1:
      lr = st.number_input("Learning rate", value=1e-4, format="%.4f", help="Learning rate. Defaults to 1e-4")
    with col2:
      rescale = st.number_input("Rescale hyperparameter", value=1e-2, format="%.2f", help="RESCALE parameter. Defaults to 1e-2")

    col1, col2 = st.columns(2)
    with col1:
      lam = st.number_input("Lambda parameter of the Poisson distribution", value=10, help="Lambda parameter. Defaults to 10")
    with col2:
      latent_dim = st.number_input("Latent space dimension", value=128, help="Dimension of the latent space. Defaults to 128")

    # Training
    col1, col2 = st.columns(2)
    with col1:
      model_type = st.segmented_control("Model type", options=["36M", "53M"], default="36M", help="Decide which version of the model to use. Defaults to 36M")
    with col2:
      optimizer = st.segmented_control("Optimizer", options=["AdamW", "Adam", "SGD"], default="AdamW", help="Decide which type of optimizer to use. Defaults to AdamW")

    col1, col2 = st.columns(2)
    with col1:
      epochs = st.number_input("Training epochs", value=100, help="Number of epochs to train. Defaults to 100")
    with col2:
      epochs_to_checkpoint = st.number_input("Epochs to create a checkpoint", value=10, help="Number of epochs to create a checkpoint. Defaults to 10")

    add_btn = st.form_submit_button("Add to training queue", width="stretch")
    if add_btn:
      path = path or config("IMG_DIR", default=Path.cwd())
      project_dir = project_dir or config("PROJECT_DIR", default=Path.cwd())

      vae_filename = vae_filename or config("VAE_FILENAME", default="VAE.pt")
      vae_checkpoint = vae_checkpoint or config("VAE_CHECKPOINT", default="VAE_checkpoint.pt")
      st.session_state.training_queue.append({
        "Resume training": resume,
        "Use optimization": optimize,
        "Clip gradients": clip_gradients,
        "Image folder path": path,
        "Project directory": project_dir,
        "VAE filename": vae_filename,
        "VAE checkpoint": vae_checkpoint,
        "Height": height,
        "Width": width,
        "Batch size": batch_size,
        "Learning rate": lr,
        "Rescale parameter": rescale,
        "Lambda parameter": lam,
        "latent space dimension": latent_dim,
        "Model type": model_type,
        "Optimizer": optimizer,
        "Training epochs": epochs,
        "Epochs to checkpoint": epochs_to_checkpoint,
      })

  if st.session_state.training_queue:
    st.write("---")
    with st.container(border=True):
      st.subheader("Training queue")
      for i, config in enumerate(st.session_state.training_queue):
        st.markdown(f"#### Job {i + 1}")

        markdown=("| :blue[Parameter] | :violet[Value] |\n"
                  "|-----------|-------|\n")
        for key, value in config.items():
          markdown += f"| {key} | {str(value)} |\n"

        st.markdown(markdown, width="stretch")

      if st.button("START TRAINING", type="primary", width="stretch"):
        for i, config in enumerate(st.session_state.training_queue):
          st.info(f"Executing job {i + 1}/{len(st.session_state.training_queue)}...")

          st.success(f"Job {i + 1} completed!")