from TTS.api import TTS
model_name = "tts_models/fr/mai/tacotron2-DDC"
vocoder_name = "vocoder_models/fr/mai/hifigan"
print("Téléchargement et chargement de", model_name)
tts = TTS(model_name, vocoder_path=vocoder_name, progress_bar=True, gpu=False)
print("Modèle téléchargé avec succès !")
