import os
import uuid
import asyncio
import logging
from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from TTS.api import TTS
from pydub import AudioSegment
from functools import partial

# config
API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025")
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_LANGUAGE = os.getenv("VICREEL_DEFAULT_LANGUAGE", "fr")
OUTPUT_DIR = os.getenv("VICREEL_OUTPUT_DIR", "outputs")
MAX_CONCURRENCY = int(os.getenv("VICREEL_MAX_CONCURRENCY", "1"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("vicreel")

app = FastAPI(title="VicReel - Coqui TTS API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

# TTS manager with lazy load and concurrency control
class TTSManager:
    def __init__(self, default_model):
        self.default_model = default_model
        self._models = {}
        self._lock = asyncio.Lock()

    async def get(self, model_name: str | None = None):
        name = model_name or self.default_model
        if name in self._models:
            return self._models[name]
        async with self._lock:
            if name in self._models:
                return self._models[name]
            loop = asyncio.get_event_loop()
            logger.info(f"Loading model {name} (may take time)")
            tts_inst = await loop.run_in_executor(None, TTS, name)
            self._models[name] = tts_inst
            logger.info(f"Model {name} loaded")
            return tts_inst

    async def synth_to_wav(self, text: str, wav_path: str, model_name: str | None = None, language: str = DEFAULT_LANGUAGE, speaker_wav: str | None = None):
        tts = await self.get(model_name)
        loop = asyncio.get_event_loop()
        kwargs = {"text": text, "file_path": wav_path}
        if "xtts" in (model_name or self.default_model).lower():
            kwargs["language"] = language
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            else:
                speakers = getattr(tts, 'speakers', [])
                if speakers:
                    kwargs["speaker"] = speakers[0]
                    logger.info(f"Using default speaker: {kwargs['speaker']}")
                else:
                    raise ValueError("No speakers available for XTTS. Provide speaker_wav.")
        logger.debug(f"Prepared kwargs: {kwargs}")
        func = partial(tts.tts_to_file, **kwargs)
        await loop.run_in_executor(None, func)
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(f"Generated file {wav_path} is empty or too small")
        logger.info(f"Generated audio at {wav_path}")

tts_manager = TTSManager(DEFAULT_MODEL)
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

def _convert_wav_to_mp3(src, dst):
    audio = AudioSegment.from_wav(src)
    audio.export(dst, format="mp3")

def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning(f"Could not delete {path}: {e}")

@app.get("/")
def root():
    return {"message": "VicReel Coqui TTS API", "default_model": DEFAULT_MODEL, "default_language": DEFAULT_LANGUAGE}

@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(tts_manager._models.keys())}

@app.post("/models/download", dependencies=[Depends(verify_api_key)])
async def download_model(body: dict):
    model = body.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model required in body")
    await tts_manager.get(model)
    return {"status": "ok", "model": model}

@app.post("/tts", dependencies=[Depends(verify_api_key)])
async def tts_endpoint(request: Request, background_tasks: BackgroundTasks):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text")
        fmt = body.get("format", "wav")
        model = body.get("model", None)
        language = body.get("language", DEFAULT_LANGUAGE)
        speaker_wav = body.get("speaker_wav", None)
    else:
        form = await request.form()
        text = form.get("text")
        fmt = form.get("format", "wav")
        model = form.get("model", None)
        language = form.get("language", DEFAULT_LANGUAGE)
        speaker_wav = form.get("speaker_wav", None)

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    job_id = uuid.uuid4().hex
    wav_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    out_path = wav_path if fmt == "wav" else os.path.join(OUTPUT_DIR, f"{job_id}.{fmt}")

    await _inference_semaphore.acquire()
    try:
        await tts_manager.synth_to_wav(text=text, wav_path=wav_path, model_name=model, language=language, speaker_wav=speaker_wav)
        if fmt != "wav":
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _convert_wav_to_mp3, wav_path, out_path)
            background_tasks.add_task(_safe_remove, wav_path)
            background_tasks.add_task(_safe_remove, out_path)
            return FileResponse(out_path, media_type="audio/mpeg", filename=os.path.basename(out_path))
        else:
            background_tasks.add_task(_safe_remove, wav_path)
            return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _inference_semaphore.release()