from TTS.api import TTS
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print("Téléchargement et chargement de", model_name)
tts = TTS(model_name, progress_bar=True, gpu=False)
print("Modèle téléchargé avec succès !")