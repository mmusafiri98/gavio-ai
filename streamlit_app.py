import streamlit as st
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import time
import zipfile
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="🎬 Text-to-Video Simple",
    page_icon="🎬",
    layout="wide"
)

# CSS simple
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🎬 Générateur Text-to-Video Simple</h1>', unsafe_allow_html=True)

# Fonction pour charger le modèle (simplifiée)
@st.cache_resource
def load_simple_model():
    try:
        from diffusers import DiffusionPipeline
        
        with st.spinner("🔄 Chargement du modèle..."):
            model_id = "damo-vilab/text-to-video-ms-1.7b"
            
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                st.sidebar.success("✅ GPU utilisé")
            else:
                st.sidebar.info("💻 CPU utilisé")
                
            return pipe
            
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        return None

# Interface utilisateur
col1, col2 = st.columns([3, 1])

with col1:
    prompt = st.text_area(
        "📝 Décrivez votre vidéo:",
        value="un uomo che corre",
        height=80
    )

with col2:
    num_frames = st.slider("🎞️ Frames", 8, 24, 16)
    seed = st.number_input("🎲 Seed", value=42, min_value=0)

# Génération
if st.button("🚀 Générer"):
    if prompt.strip():
        pipe = load_simple_model()
        
        if pipe:
            try:
                with st.spinner("🎬 Génération en cours..."):
                    # Configuration
                    generator = torch.manual_seed(seed)
                    
                    # Génération
                    with torch.inference_mode():
                        result = pipe(
                            prompt,
                            num_frames=num_frames,
                            generator=generator,
                            num_inference_steps=20,
                            guidance_scale=7.0
                        )
                    
                    # Récupération des frames
                    if hasattr(result, 'frames') and result.frames:
                        frames = result.frames[0]
                    else:
                        frames = result if isinstance(result, list) else [result]
                    
                    st.success(f"✅ {len(frames)} frames générées!")
                    
                    # Affichage des frames
                    st.markdown("**🖼️ Frames générées:**")
                    
                    # Grille d'affichage
                    cols = st.columns(4)
                    for i, frame in enumerate(frames[:12]):  # Afficher max 12 frames
                        with cols[i % 4]:
                            if isinstance(frame, Image.Image):
                                st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                            else:
                                st.image(Image.fromarray(frame), caption=f"Frame {i+1}", use_column_width=True)
                    
                    # Créer un ZIP avec toutes les frames
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for i, frame in enumerate(frames):
                            # Convertir en PIL si nécessaire
                            if not isinstance(frame, Image.Image):
                                frame = Image.fromarray(frame)
                            
                            # Sauvegarder dans le ZIP
                            img_buffer = BytesIO()
                            frame.save(img_buffer, format='PNG')
                            zip_file.writestr(f"frame_{i+1:03d}.png", img_buffer.getvalue())
                    
                    zip_buffer.seek(0)
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label="⬇️ Télécharger toutes les frames (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"frames_{int(time.time())}.zip",
                        mime="application/zip"
                    )
                    
                    # Instructions pour créer la vidéo
                    st.markdown("---")
                    st.markdown("**🎬 Pour créer une vidéo à partir des frames:**")
                    st.code("""
# Avec ffmpeg (si installé)
ffmpeg -r 8 -i frame_%03d.png -vcodec libx264 -pix_fmt yuv420p output.mp4

# Ou avec Python
from PIL import Image
import imageio

frames = []
for i in range(1, num_frames+1):
    img = Image.open(f'frame_{i:03d}.png')
    frames.append(np.array(img))

imageio.mimsave('video.mp4', frames, fps=8)
                    """)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de la génération: {e}")
    else:
        st.warning("⚠️ Veuillez entrer un prompt")

# Informations
st.markdown("---")
st.markdown("""
**ℹ️ Cette version simplifiée:**
- Génère des frames individuelles
- Évite les problèmes de dépendances vidéo
- Permet de télécharger les frames en ZIP
- Les frames peuvent être assemblées en vidéo séparément

**💡 Conseils:**
- Commencez avec 16 frames pour tester
- Utilisez des descriptions détaillées
- Ajustez le seed pour varier les résultats
""")

# Status système
with st.expander("🖥️ Informations système"):
    st.write(f"GPU disponible: {'✅' if torch.cuda.is_available() else '❌'}")
    if torch.cuda.is_available():
        st.write(f"GPU: {torch.cuda.get_device_name()}")
    st.write(f"PyTorch: {torch.__version__}")
