import os import uuid import asyncio import logging from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status from fastapi.responses import FileResponse, JSONResponse from fastapi.middleware.cors import CORSMiddleware from fastapi.security.api_key import APIKeyHeader from TTS.api import TTS from pydub import AudioSegment from functools import partial from typing import Optional

Configuration

API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025") DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2") DEFAULT_LANGUAGE = os.getenv("VICREEL_DEFAULT_LANGUAGE", "fr") OUTPUT_DIR = os.getenv("VICREEL_OUTPUT_DIR", "outputs") MAX_CONCURRENCY = int(os.getenv("VICREEL_MAX_CONCURRENCY", "1"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

Logging

logging.basicConfig(level=logging.INFO) logger = logging.getLogger("vicreel")

app = FastAPI(title="VicReel - Coqui TTS API") app.add_middleware(CORSMiddleware, allow_origins=[""], allow_methods=[""], allow_headers=["*"])

API Key header dependency

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def verify_api_key(api_key: Optional[str] = Depends(api_key_header)): """Validate API key. If no API key is configured on the server (empty string), accept all requests. If an API key is configured, header must be present and match.""" if not API_KEY: return True if not api_key or api_key != API_KEY: raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key") return True

TTS manager with lazy load and concurrency control

class TTSManager: def init(self, default_model: str): self.default_model = default_model self._models: dict[str, TTS] = {} self._lock = asyncio.Lock()

async def get(self, model_name: Optional[str] = None) -> TTS:
    name = model_name or self.default_model
    if name in self._models:
        return self._models[name]
    async with self._lock:
        if name in self._models:
            return self._models[name]
        loop = asyncio.get_event_loop()
        logger.info(f"Loading model {name} (this may take some time)")
        # Use partial to call the TTS constructor in a threadpool
        ctor = partial(TTS, name)
        tts_inst = await loop.run_in_executor(None, ctor)
        self._models[name] = tts_inst
        logger.info(f"Model {name} loaded")
        return tts_inst

async def synth_to_wav(self, text: str, wav_path: str, model_name: Optional[str] = None, language: str = DEFAULT_LANGUAGE, speaker_wav: Optional[str] = None):
    tts = await self.get(model_name)
    loop = asyncio.get_event_loop()
    effective_model = (model_name or self.default_model).lower()

    # Prepare kwargs according to model type
    kwargs = {"text": text, "file_path": wav_path}
    if "xtts" in effective_model:
        kwargs["language"] = language
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        else:
            speakers = getattr(tts, 'speakers', []) or getattr(tts, 'voices', [])
            if speakers:
                # pick the first available speaker/voice
                kwargs["speaker"] = speakers[0]
                logger.info(f"Using default speaker: {kwargs['speaker']}")
            else:
                # Fallback: don't set speaker, let model decide
                logger.warning("No speakers/voices attribute found on model; proceeding without explicit speaker")

    logger.debug(f"Prepared tts kwargs: {kwargs}")
    func = partial(getattr(tts, 'tts_to_file', tts.tts_to_file), **kwargs)
    await loop.run_in_executor(None, func)

    # Validate generated file
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
        raise RuntimeError(f"Generated file {wav_path} is empty or too small")
    logger.info(f"Generated audio at {wav_path}")

tts_manager = TTSManager(DEFAULT_MODEL) _inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

def _convert_wav_to_mp3(src: str, dst: str): audio = AudioSegment.from_wav(src) audio.export(dst, format="mp3")

def _safe_remove(path: str): try: if os.path.exists(path): os.remove(path) logger.debug(f"Removed temporary file {path}") except Exception as e: logger.warning(f"Could not delete {path}: {e}")

@app.get("/") def root(): return {"message": "VicReel Coqui TTS API", "default_model": DEFAULT_MODEL, "default_language": DEFAULT_LANGUAGE}

@app.get("/health") def health(): return {"status": "ok", "loaded_models": list(tts_manager._models.keys())}

@app.post("/models/download", dependencies=[Depends(verify_api_key)]) async def download_model(body: dict): model = body.get("model") if not model: raise HTTPException(status_code=400, detail="model required in body") await tts_manager.get(model) return {"status": "ok", "model": model}

@app.post("/tts", dependencies=[Depends(verify_api_key)]) async def tts_endpoint(request: Request, background_tasks: BackgroundTasks): # Accept JSON or form content_type = request.headers.get("content-type", "") if "application/json" in content_type: body = await request.json() text = body.get("text") fmt = (body.get("format") or "wav").lower() model = body.get("model", None) language = body.get("language", DEFAULT_LANGUAGE) speaker_wav = body.get("speaker_wav", None) else: form = await request.form() text = form.get("text") fmt = (form.get("format") or "wav").lower() model = form.get("model", None) language = form.get("language", DEFAULT_LANGUAGE) speaker_wav = form.get("speaker_wav", None)

if not text:
    raise HTTPException(status_code=400, detail="text is required")

if fmt not in ("wav", "mp3"):
    raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

job_id = uuid.uuid4().hex
wav_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
out_path = wav_path if fmt == "wav" else os.path.join(OUTPUT_DIR, f"{job_id}.mp3")

# Concurrency control
async with _inference_semaphore:
    try:
        await tts_manager.synth_to_wav(text=text, wav_path=wav_path, model_name=model, language=language, speaker_wav=speaker_wav)

        if fmt != "wav":
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _convert_wav_to_mp3, wav_path, out_path)
            # Remove the intermediate WAV after response is prepared
            background_tasks.add_task(_safe_remove, wav_path)
            # Remove the MP3 after response is served
            background_tasks.add_task(_safe_remove, out_path)
            return FileResponse(out_path, media_type="audio/mpeg", filename=os.path.basename(out_path))
        else:
            # Remove WAV after response is served
            background_tasks.add_task(_safe_remove, wav_path)
            return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS error")
        # Return a safe error message to clients; the logs keep the full detail
        raise HTTPException(status_code=500, detail="Internal TTS error")

