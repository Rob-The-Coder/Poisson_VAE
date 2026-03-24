import streamlit as st
import subprocess

from decouple import config
from pathlib import Path

def get_checkpoints(base_dir):
  checkpoint_path = Path(base_dir) / "checkpoints"
  if not checkpoint_path.exists():
    return []

  checkpoints = [f.name for f in checkpoint_path.glob("*.pt")]
  return sorted(checkpoints)

img_path = config("IMG_DIR", default=Path.cwd())
project_dir = config("PROJECT_DIR", default=Path.cwd())

if 'training_queue' not in st.session_state:
  st.session_state.training_queue = []

st.title("VAE training GUI")

with st.container(border=True):
  st.subheader("Training scheduler")

  # Logic - resume is outside the form due to Streamlit handling logic
  col1, col2, col3 = st.columns(3)
  with col1:
    resume = st.toggle("Resume training", value=False, help="Resume training from checkpoint. If not used defaults to False")

  with st.form("training_schedule", border=False):
    # Logic - rest of the toggles
    with col2:
      optimize = st.toggle("Use optimization", value=True, help="Enables JIT and AMP. Defaults to True")
    with col3:
      clip_gradients = st.toggle("Gradient clipping", value=False, disabled=resume, help="Enables gradient clipping. Defaults to False")

    # Path and file handling
    col1, col2 = st.columns(2)
    with col1:
      vae_filename = st.text_input("VAE filename", value=None, help="Name of the generated VAE file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE.pt")
    with col2:
      if resume:
        available_checkpoints = get_checkpoints(project_dir)

        if not available_checkpoints:
          st.warning(f"⚠️ No checkpoints found in {project_dir}. Check again the project directory in the **.env** file!")
          vae_checkpoint = st.text_input("VAE checkpoint", value=None, help="Name of the checkpoint file used to resume training. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE_checkpoint.pt")
        else:
          vae_checkpoint = st.selectbox("Select which checkpoint to resume", options=available_checkpoints, help="List of the available checkpoints in **.pt** format to resume")
      else:
        vae_checkpoint = st.text_input("VAE checkpoint", value=None, help="Name of the generated training checkpoint file. if not specified will use the name specified in the .env file. If both are not specified it will default to VAE_checkpoint.pt")

    # Hyperparameters
    col1, col2, col3 = st.columns(3)
    with col1:
      height = st.number_input("Height", value=64, help="Height of the image. Defaults to 64")
    with col2:
      width = st.number_input("Width", value=64, help="Width of the image. Defaults to 64")
    with col3:
      batch_size = st.number_input("Batch size", value=128, help="Batch size. Defaults to 128")

    col1, col2 = st.columns(2)
    with col1:
      lr = st.number_input("Learning rate", value=1e-4, format="%.4f", disabled=resume, help="Learning rate. Defaults to 1e-4")
    with col2:
      rescale = st.number_input("Rescale hyperparameter", value=1e-2, format="%.2f", disabled=resume, help="RESCALE parameter. Defaults to 1e-2")

    col1, col2 = st.columns(2)
    with col1:
      lam = st.number_input("Lambda parameter of the Poisson distribution", value=10, disabled=resume, help="Lambda parameter. Defaults to 10")
    with col2:
      latent_dim = st.number_input("Latent space dimension", value=128, disabled=resume, help="Dimension of the latent space. Defaults to 128")

    # Training
    col1, col2, col3 = st.columns(3)
    with col1:
      model_type = st.segmented_control("Model type", options=["36M", "53M"], default="36M", disabled=resume, help="Decide which version of the model to use. Defaults to 36M")
    with col2:
      sampling = st.segmented_control("Sampling strategy", options=["PGA", "GRT"], default="PGA", help="Decide which sampling strategy to adopt. Defaults to PGA")
    with col3:
      optimizer = st.segmented_control("Optimizer", options=["AdamW", "Adam", "SGD"], default="AdamW", disabled=resume, help="Decide which type of optimizer to use. Defaults to AdamW")

    col1, col2, col3 = st.columns(3)
    with col1:
      epochs = st.number_input("Training epochs", value=100, help="Number of epochs to train. Defaults to 100")
    with col2:
      epochs_to_checkpoint = st.number_input("Epochs to create a checkpoint", value=10, help="Number of epochs to create a checkpoint. Defaults to 10")
    with col3:
      epochs_to_monitor = st.number_input("Epochs to monitor training", value=0, help="Number of epochs to monitor the training process. Defaults to 0")

    add_btn = st.form_submit_button("Add to training queue", width="stretch")
    if add_btn:
      vae_filename = vae_filename or config("VAE_FILENAME", default="VAE.pt")
      vae_checkpoint = vae_checkpoint or config("VAE_CHECKPOINT", default="VAE_checkpoint.pt")
      st.session_state.training_queue.append({
        "Resume training": resume,
        "Use optimization": optimize,
        "Clip gradients": clip_gradients,
        "Image folder path": img_path,
        "Project directory": project_dir,
        "VAE filename": vae_filename,
        "VAE checkpoint": vae_checkpoint,
        "Height": height,
        "Width": width,
        "Batch size": batch_size,
        "Learning rate": lr,
        "Rescale parameter": rescale,
        "Lambda parameter": lam,
        "Latent space dimension": latent_dim,
        "Model type": model_type,
        "Sampling": sampling,
        "Optimizer": optimizer,
        "Training epochs": epochs,
        "Epochs to checkpoint": epochs_to_checkpoint,
        "Epochs to monitor": epochs_to_monitor
      })

if st.session_state.training_queue:
  st.divider()
  with st.container(border=True):
    st.subheader("Training queue")
    for i, config in enumerate(st.session_state.training_queue):
      col1, col2, col3 = st.columns(3)
      with col1:
        st.markdown(f"#### Job {i + 1}")
      with col2:
        st.space("stretch")
      with col3:
        if st.button(label="", icon=":material/delete:", width="stretch", type="primary", key=f"btn{i}"):
          del st.session_state.training_queue[i - 1]

      markdown= "| :blue[Parameter] | :violet[Value] |\n|-----------|-------|\n"
      for key, value in config.items():
        markdown += f"| {key} | {str(value)} |\n"
      st.markdown(markdown, width="stretch")

    if st.button("START TRAINING", type="primary", width="stretch"):
      command=f""
      for i, config in enumerate(st.session_state.training_queue):
        command += (f"python3 train_vae.py --images_dir '{config['Image folder path']}' --project_dir '{config['Project directory']}' "
                  f"--vae_filename '{config['VAE filename']}' --vae_checkpoint '{config['VAE checkpoint']}' --height {config['Height']} "
                  f"--width {config['Width']} --batch_size {config['Batch size']} --lr {config['Learning rate']} --rescale {config['Rescale parameter']} "
                  f"--lam {config['Lambda parameter']} --latent_dim {config['Latent space dimension']} --type '{config['Model type']}' --sampling {config['Sampling']} "
                  f"--optimizer '{config['Optimizer']}' --epochs_to_checkpoint {config['Epochs to checkpoint']} --epochs_to_monitor {config['Epochs to monitor']} "
                  f"--epochs {config['Training epochs']} ")

        if not config['Use optimization']:
          command += "--optimize False "
        if config['Resume training']:
          command += "--resume "
        if clip_gradients:
          command += "--clip_gradients True "
        command += ";"

      subprocess.Popen(command, shell=True)
      st.success(f"Training command successfully created and launched!")