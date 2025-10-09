from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=False)
speakers = tts.speakers if hasattr(tts, 'speakers') else []
print(f"Nombre total speakers : {len(speakers)}")
print("Premiers 10 :", speakers[:10])