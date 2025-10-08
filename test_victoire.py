from TTS.api import TTS
import os

model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
print("Chargement XTTS-v2...")
tts = TTS(model_name, progress_bar=True, gpu=False)

# Correction : Liste et utilise un speaker_id par défaut (XTTS en a besoin)
speakers = tts.speakers if hasattr(tts, 'speakers') else ["default"]
speaker_id = speakers[0] if speakers else None  # Premier speaker built-in (ex. "Ana Florence")
print(f"Speakers disponibles : {speakers}")
print(f"Utilise speaker_id : {speaker_id}")

print("Génération...")

text = "Bonjour, VicReel victoire ! Test TTS réussi."
output_file = "output/victoire.wav"
os.makedirs("output", exist_ok=True)
tts.tts_to_file(text, file_path=output_file, language="fr", speaker_id=speaker_id)  # Ajoute speaker_id
size = os.path.getsize(output_file) / 1024 if os.path.exists(output_file) else 0
print(f"Taille : {size:.2f} Ko")

if size > 10:
    print("Victoire ! Télécharge output/victoire.wav.")
else:
    print("Problème persistant : copie log.")