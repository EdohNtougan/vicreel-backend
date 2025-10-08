from TTS.api import TTS
from functools import partial
import os

model = "tts_models/multilingual/multi-dataset/xtts_v2"
wav = "quick_test.wav"
text = "Test audio VicReel: une génération rapide."
print("Loading model (this can take a minute)...")
tts = TTS(model)
print("Speakers:", getattr(tts, "speakers", [])[:5])
# Example: use first available speaker if present
speaker = tts.speakers[0] if getattr(tts, "speakers", None) else None
print("Using speaker:", speaker)
# Correct call: use partial to pass kwargs into run_in_executor if needed — but here we call directly
if speaker:
    # call directly synchronously to isolate issues
    tts.tts_to_file(text=text, file_path=wav, speaker=speaker, language="fr")
else:
    tts.tts_to_file(text=text, file_path=wav, language="fr")
print("File size (bytes):", os.path.getsize(wav))