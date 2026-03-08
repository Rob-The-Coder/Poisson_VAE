import streamlit as st

# Gui configuration
st.set_page_config(page_title="Poisson VAE Lab", layout="centered")

# Page selection
training_page = st.Page("gui/training_gui.py", title="Train Model", icon="🏋️")
generation_page = st.Page("gui/generation_gui.py", title="Generate Faces", icon="🖼️")
pg = st.navigation([training_page, generation_page])

pg.run()