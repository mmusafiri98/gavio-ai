from gradio_client import Client

# Initialiser le client du modèle
client = Client("Qwen/Qwen2.5-Coder-7B-Instruct")

# Exemple de prompt
prompt = "Écris une fonction Python pour trier une liste."

# Appel au modèle
code_result = client.predict(prompt_text=prompt)
print(code_result)

