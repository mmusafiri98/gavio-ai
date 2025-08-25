import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="🎵 MusicGen App", page_icon="🎶", layout="centered")

st.title("🎶 Générateur de musique avec MusicGen (CPU friendly)")

# Charger le modèle MusicGen
@st.cache_resource
def load_model():
    return pipeline("text-to-audio", model="facebook/musicgen-small")

music_pipe = load_model()

# Interface utilisateur
prompt = st.text_area("📝 Décris ta chanson :", "A calm jazz melody with piano and saxophone.")

duration = st.slider("⏱️ Durée (secondes)", 5, 30, 10)

if st.button("🎼 Générer"):
    with st.spinner("🎵 Génération de la musique..."):
        result = music_pipe(prompt, forward_params={"max_new_tokens": duration * 50})  # environ
        audio_bytes = result["audio"]

        # Sauvegarder en fichier
        filename = "musicgen_output.wav"
        with open(filename, "wb") as f:
            f.write(audio_bytes)

        st.success("✅ Musique générée !")
        st.audio(filename, format="audio/wav")

        # Bouton de téléchargement
        with open(filename, "rb") as f:
            st.download_button(
                label="⬇️ Télécharger la musique",
                data=f,
                file_name=filename,
                mime="audio/wav"
            )
