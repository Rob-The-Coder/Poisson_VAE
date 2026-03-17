import streamlit as st
import io

from decouple import config
from pathlib import Path
from dataclasses import asdict

from generate_faces import GenerationArgs, generate

def get_models(base_dir):
  model_path = Path(base_dir) / "models"
  if not model_path.exists():
    return []

  models = [f.name for f in model_path.glob("*.pt")]
  return sorted(models)

img_path = config("IMG_DIR", default=Path.cwd())
project_dir = config("PROJECT_DIR", default=Path.cwd())

st.title("VAE generation GUI")
with st.container(border=True):
  st.subheader("Faces generation")

  # Face generation
  col1, col2, col3 = st.columns(3)
  with col1:
    available_models = get_models(project_dir)
    vae_filename = st.selectbox("Select which model to use", options=available_models, help="List of the available models in **.pt** format to use")
  with col2:
    num_faces = st.number_input("Number of faces to generate", value=36, help="Number of faces to generate. Defaults to 36")
  with col3:
    lam = st.number_input("Lambda parameter", value=10, help="Lambda parameter. Defaults to 10")

  title = st.text_area("Image title", value="", help="Title of the generated plot. By default is set to a blank string")

  # Interpolation
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Faces interpolation")
  with col2:
    st.space("xxsmall")
    interpolate = st.toggle("Interpolate", value=True, help="Whether to interpolate images. Defaults to True")

  col1, col2 = st.columns(2)
  with col1:
    height = st.number_input("Height", value=64, disabled=not interpolate, help="Height of the image. Defaults to 64")
  with col2:
    width = st.number_input("Width", value=64, disabled=not interpolate, help="Width of the image. Defaults to 64")

  start, end = st.slider("Choose images to interpolate", min_value=0, max_value=162669, value=(6000, 80000), disabled=not interpolate, help="Starting and ending images used to interpolate. Defaults to respectively 6000 and 80000")

  # Clusterization
  col1, col2 = st.columns(2)
  with col1:
    st.subheader("Clusterization")
  with col2:
    st.space("xxsmall")
    clusterize = st.toggle("Clusterize", value=True, help="Whether to compute clusterization. Defaults to True")

  col1, col2 = st.columns(2)
  with col1:
    batch_size = st.number_input("Batch size", value=512, disabled=not clusterize, help="Batch size used to compute clusters. Defaults to 512")
  with col2:
    num_samples = st.number_input("Number of samples", value=5000, disabled=not clusterize, help="Samples number used to compute clusterization. Defaults to 5000")

  btn = st.button("Generate", width="stretch")

if btn:
  with st.container(border=True):
    vae_filename = vae_filename or config("VAE_FILENAME", default="VAE.pt")

    args_obj = GenerationArgs(
      images_dir=img_path,
      project_dir=project_dir,
      vae_filename=vae_filename,
      num_faces=num_faces,
      lam=lam,
      title=title,
      interpolation=interpolate,
      height=height,
      width=width,
      start=start,
      end=end,
      clusterization=clusterize,
      batch_size=batch_size,
      num_samples=num_samples
    )

    markdown = "| :blue[Parameter] | :violet[Value] |\n|-----------|-------|\n"
    for key, value in asdict(args_obj).items():
      markdown += f"| {key} | {str(value)} |\n"
    st.markdown(markdown, width="stretch")

    with st.spinner("🎨 Generando grafici..."):
      figs = generate(args_obj)

      for i, fig in enumerate(figs):
        st.divider()
        st.pyplot(fig)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')

        st.download_button(
          label=f"Download plot {i + 1}",
          data=buf.getvalue(),
          file_name=f"plot.png",
          mime="image/png",
          key=f"btn_{i}"
        )