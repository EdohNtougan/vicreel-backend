from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from TTS.api import TTS
import os

app = FastAPI(title="VicReel Backend", description="Coqui TTS API for VicReel", version="1.0")

# 📂 Dossier pour stocker les fichiers audio générés
os.makedirs("outputs", exist_ok=True)

# 🧠 Charger un modèle TTS français de Coqui (ex. voix féminine)
MODEL_NAME = "tts_models/fr/ljspeech/vits"
tts = TTS(model_name=MODEL_NAME, progress_bar=False, gpu=False)

@app.get("/")
def home():
    return {"message": "Bienvenue sur le backend VicReel (Coqui TTS API)"}

@app.get("/generate/")
def generate(text: str):
    """Génère une voix depuis le texte."""
    if not text:
        raise HTTPException(status_code=400, detail="Le paramètre 'text' est requis.")
    
    output_path = f"outputs/result.wav"
    tts.tts_to_file(text=text, file_path=output_path)
    
    return FileResponse(output_path, media_type="audio/wav", filename="vicreel_voice.wav")
