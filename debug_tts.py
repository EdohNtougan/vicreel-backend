# python debug_tts.py
import os, traceback
from TTS.api import TTS

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
OUT = "debug_test.wav"

print("Python:", os.getenv("PYTHONPATH"))
try:
    print("Chargement du modèle :", MODEL)
    tts = TTS(MODEL, progress_bar=True, gpu=False)
    print("Instance TTS chargée")
    # print attributes
    for attr in ("speakers","voices","languages","supported_languages"):
        val = getattr(tts, attr, None)
        if val:
            print(f"{attr}: (len={len(val)}) sample: {val[:5] if isinstance(val, (list,tuple)) else val}")
        else:
            print(f"{attr}: None or empty")
    # choose first speaker if exists
    speakers = getattr(tts, 'speakers', None) or getattr(tts, 'voices', None) or []
    speaker = speakers[0] if speakers else None
    print("Using speaker:", speaker)
    print("Synthesizing a short test to", OUT)
    tts.tts_to_file(text="Test VicReel debug", file_path=OUT, language="fr", speaker=speaker) if speaker else tts.tts_to_file(text="Test VicReel debug", file_path=OUT)
    print("Synthesis complete, checking file size...")
    print("exists:", os.path.exists(OUT), "size:", os.path.getsize(OUT) if os.path.exists(OUT) else "N/A")
except Exception:
    traceback.print_exc()
