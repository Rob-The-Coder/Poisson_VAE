import streamlit as st

# Gui configuration
st.set_page_config(page_title="Poisson VAE Lab", layout="centered")

# Page selection
training_page = st.Page("gui/training_gui.py", title="Train model", icon="🏋️")
generation_page = st.Page("gui/generation_gui.py", title="Generate faces", icon="🖼️")
pg = st.navigation({"VAE": [training_page, generation_page]})

pg.run()