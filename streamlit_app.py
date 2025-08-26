import streamlit as st
from gradio_client import Client

# Initialiser le client du modèle
client = Client("Qwen/Qwen2.5-Coder-Artifacts")

st.set_page_config(layout="wide")
st.title("Interface Streamlit pour Qwen2.5-Coder")

# Layout avec colonnes
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Prompt")
    prompt = st.text_area("Entrez votre demande de code ici :")
    if st.button("Générer le code"):
        if prompt:
            with st.spinner("Génération du code..."):
                # Appel du modèle
                code_result = client.predict(
                    api_name="/demo_card_click",
                    data=[prompt]
                )
            st.session_state["generated_code"] = code_result

with col2:
    st.header("Code généré")
    code = st.session_state.get("generated_code", "")
    st.code(code, language="python" if "python" in code.lower() else "html")

    st.header("Résultat / Aperçu")
    if code.strip() != "":
        try:
            # Si le code est du HTML, l'afficher
            if code.strip().startswith("<") or "html" in code.lower():
                st.components.v1.html(code, height=500)
            # Sinon, exécuter le code Python et afficher le résultat
            else:
                exec_locals = {}
                exec(code, {}, exec_locals)
                if 'output' in exec_locals:
                    st.write(exec_locals['output'])
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du code : {e}")
