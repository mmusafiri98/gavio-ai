import streamlit as st
import torch
from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
import tempfile
import os
import time
from io import BytesIO
import base64

# Gestion flexible de cv2 pour éviter les erreurs
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("⚠️ OpenCV non disponible. Utilisation d'une alternative pour la génération vidéo.")

# Alternative sans cv2
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="🎬 Text-to-Video Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .info-box {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🎬 Générateur Text-to-Video</h1>', unsafe_allow_html=True)

# Configuration dans la sidebar
st.sidebar.header("⚙️ Configuration")

# Fonction pour initialiser le modèle
@st.cache_resource
def load_model():
    """Charge le modèle text-to-video"""
    try:
        with st.spinner("🔄 Chargement du modèle (peut prendre quelques minutes...)"):
            model_id = "damo-vilab/text-to-video-ms-1.7b"
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                variant="fp16" if torch.cuda.is_available() else None
            )
            
            # Optimisations
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")
                st.sidebar.success("✅ GPU détecté et utilisé")
            else:
                st.sidebar.warning("⚠️ CPU utilisé (génération lente)")
                
            # Optimisations mémoire
            try:
                pipe.enable_model_cpu_offload()
                pipe.enable_xformers_memory_efficient_attention()
            except:
                pass
                
            return pipe
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None

# Fonction pour convertir frames en vidéo (avec alternatives)
def frames_to_video(frames, fps=8, output_path="output.mp4"):
    """Convertit une liste de frames PIL en vidéo MP4 avec plusieurs méthodes"""
    if not frames:
        return None
    
    try:
        # Méthode 1: Avec imageio (plus compatible)
        if IMAGEIO_AVAILABLE:
            frame_arrays = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_array = np.array(frame)
                else:
                    frame_array = frame
                frame_arrays.append(frame_array)
            
            # Créer la vidéo avec imageio
            imageio.mimsave(output_path, frame_arrays, fps=fps, format='mp4')
            return output_path
            
        # Méthode 2: Avec cv2 (si disponible)
        elif CV2_AVAILABLE:
            frame_arrays = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_array = np.array(frame)
                else:
                    frame_array = frame
                frame_arrays.append(frame_array)
            
            height, width, channels = frame_arrays[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for frame_array in frame_arrays:
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            return output_path
            
        # Méthode 3: GIF animé comme alternative
        else:
            gif_path = output_path.replace('.mp4', '.gif')
            pil_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    pil_frames.append(frame)
                else:
                    pil_frames.append(Image.fromarray(frame))
            
            # Sauvegarder comme GIF animé
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000/fps),
                loop=0
            )
            return gif_path
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la création vidéo: {e}")
        return None

# Fonction pour générer la vidéo
def generate_video(pipe, prompt, num_frames=16, fps=8, seed=None):
    """Génère une vidéo à partir du prompt"""
    try:
        # Configuration du générateur pour la reproductibilité
        generator = None
        if seed is not None:
            generator = torch.manual_seed(seed)
        
        # Génération
        with st.spinner(f"🎬 Génération de {num_frames} frames..."):
            progress_bar = st.progress(0)
            
            # Génération de la vidéo
            with torch.inference_mode():
                result = pipe(
                    prompt,
                    num_frames=num_frames,
                    generator=generator,
                    num_inference_steps=25,
                    guidance_scale=7.5
                )
                
            progress_bar.progress(50)
            
            # Récupération des frames
            if hasattr(result, 'frames') and result.frames:
                frames = result.frames[0]  # Premier batch
            elif isinstance(result, list):
                frames = result
            else:
                frames = [result]
                
            progress_bar.progress(75)
            
            # Conversion en vidéo
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                video_path = frames_to_video(frames, fps=fps, output_path=tmp_file.name)
            
            progress_bar.progress(100)
            
            return frames, video_path
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la génération: {e}")
        return None, None

# Interface utilisateur principale
col1, col2 = st.columns([2, 1])

with col1:
    # Zone de texte pour le prompt
    prompt = st.text_area(
        "📝 Décrivez la vidéo que vous voulez générer:",
        value="un uomo che corre",
        height=100,
        help="Décrivez en détail ce que vous voulez voir dans la vidéo"
    )
    
    # Exemples de prompts
    st.markdown("💡 **Exemples de prompts:**")
    example_prompts = [
        "un chat qui joue avec une balle",
        "une voiture qui roule sur une route de campagne",
        "un oiseau qui vole dans le ciel bleu",
        "des vagues qui s'écrasent sur la plage",
        "une personne qui danse sous la pluie"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(example_prompts):
        if cols[i % 3].button(f"📋 {example}", key=f"example_{i}"):
            st.experimental_rerun()

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**📊 Paramètres de génération:**")
    
    num_frames = st.slider("🎞️ Nombre de frames", 8, 32, 16, help="Plus de frames = vidéo plus longue")
    fps = st.slider("⚡ FPS (images/seconde)", 4, 16, 8, help="Vitesse de la vidéo")
    
    use_seed = st.checkbox("🎲 Utiliser un seed fixe", help="Pour des résultats reproductibles")
    seed = st.number_input("Seed", value=42, min_value=0) if use_seed else None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Informations système
    st.markdown("**🖥️ Système:**")
    st.write(f"GPU disponible: {'✅' if torch.cuda.is_available() else '❌'}")
    if torch.cuda.is_available():
        st.write(f"GPU: {torch.cuda.get_device_name()}")

# Chargement du modèle
pipe = load_model()

# Bouton de génération
if st.button("🚀 Générer la Vidéo", disabled=(pipe is None)):
    if prompt.strip():
        # Génération de la vidéo
        frames, video_path = generate_video(
            pipe, prompt, 
            num_frames=num_frames, 
            fps=fps, 
            seed=seed
        )
        
        if frames and video_path:
            st.success("✅ Vidéo générée avec succès!")
            
            # Affichage de la vidéo/GIF
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🎬 Résultat généré:**")
                
                # Déterminer le type de fichier
                is_gif = video_path.endswith('.gif')
                
                if is_gif:
                    # Afficher le GIF
                    with open(video_path, 'rb') as gif_file:
                        gif_bytes = gif_file.read()
                        st.image(gif_bytes)
                        
                    # Bouton de téléchargement pour GIF
                    st.download_button(
                        label="⬇️ Télécharger le GIF",
                        data=gif_bytes,
                        file_name=f"animation_{int(time.time())}.gif",
                        mime="image/gif"
                    )
                else:
                    # Afficher la vidéo MP4
                    with open(video_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    
                    # Bouton de téléchargement pour MP4
                    st.download_button(
                        label="⬇️ Télécharger la vidéo",
                        data=video_bytes,
                        file_name=f"video_{int(time.time())}.mp4",
                        mime="video/mp4"
                    )
            
            with col2:
                st.markdown("**🖼️ Frames individuelles:**")
                
                # Afficher quelques frames
                num_display = min(4, len(frames))
                frame_cols = st.columns(2)
                
                for i in range(num_display):
                    with frame_cols[i % 2]:
                        st.image(
                            frames[i * len(frames) // num_display], 
                            caption=f"Frame {i * len(frames) // num_display + 1}",
                            use_column_width=True
                        )
            
            # Nettoyage du fichier temporaire
            try:
                os.unlink(video_path)
            except:
                pass
                
    else:
        st.warning("⚠️ Veuillez entrer un prompt pour générer la vidéo")

# Footer avec informations
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🎬 Générateur Text-to-Video utilisant le modèle <strong>damo-vilab/text-to-video-ms-1.7b</strong></p>
    <p>💡 <em>Astuce: Utilisez des descriptions détaillées pour de meilleurs résultats</em></p>
</div>
""", unsafe_allow_html=True)

# Instructions d'installation
with st.expander("📋 Instructions d'installation"):
    st.code("""
# Installation des dépendances
pip install streamlit torch diffusers transformers opencv-python pillow

# Pour lancer l'application
streamlit run app.py
    """, language="bash")
    
    st.markdown("""
    **Requirements système:**
    - GPU NVIDIA recommandé (8GB+ VRAM)
    - RAM: 16GB+ recommandé
    - Python 3.8+
    """)
