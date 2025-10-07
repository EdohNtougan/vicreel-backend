# app.py
import os
import uuid
import asyncio
import logging
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Header, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
from pydub import AudioSegment

# -------------------------
# Configuration via env
# -------------------------
VICREEL_API_KEY = os.getenv("VICREEL_API_KEY")  # set this in Codespace env
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/fr/ljspeech/vits")
MODELS_DIR = os.getenv("VICREEL_MODELS_DIR", "models")
OUTPUT_DIR = os.getenv("VICREEL_OUTPUT_DIR", "outputs")
MAX_CONCURRENCY = int(os.getenv("VICREEL_MAX_CONCURRENCY", "1"))  # limit parallel inferences

# create folders
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vicreel")

app = FastAPI(title="VicReel - Coqui TTS API")

# allow all origins for dev (change in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Simple API key dependency
# -------------------------
def verify_api_key(x_api_key: str | None = Header(None)):
    if VICREEL_API_KEY:
        if not x_api_key or x_api_key != VICREEL_API_KEY:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")

# -------------------------
# TTS Manager (lazy load)
# -------------------------
class TTSManager:
    def __init__(self, default_model=DEFAULT_MODEL):
        self.default_model = default_model
        self.models = {}  # model_name -> TTS instance
        self._lock = asyncio.Lock()

    async def get_tts(self, model_name: str | None = None):
        name = model_name or self.default_model
        if name in self.models:
            return self.models[name]
        # ensure only one coroutine tries to load at once
        async with self._lock:
            if name in self.models:
                return self.models[name]
            loop = asyncio.get_event_loop()
            logger.info(f"Loading model: {name} (this may take time the first call)")
            # run blocking TTS(...) in thread pool
            tts_instance = await loop.run_in_executor(None, TTS, name)
            self.models[name] = tts_instance
            logger.info(f"Model loaded: {name}")
            return tts_instance

    async def synth_to_file(self, text: str, out_wav: str, model_name: str | None = None):
        tts = await self.get_tts(model_name)
        loop = asyncio.get_event_loop()
        # blocking tts.tts_to_file -> run in executor
        await loop.run_in_executor(None, tts.tts_to_file, text, out_wav)

tts_manager = TTSManager()

# -------------------------
# Concurrency limiter
# -------------------------
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# -------------------------
# Helpers
# -------------------------
def convert_wav_to_format(src_wav: str, dst_path: str, fmt: str = "mp3"):
    # uses pydub -> ffmpeg required
    audio = AudioSegment.from_wav(src_wav)
    audio.export(dst_path, format=fmt)

def safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Removed {path}")
    except Exception as e:
        logger.warning(f"Failed to remove {path}: {e}")

# -------------------------
# Endpoints
# -------------------------
@app.get("/", tags=["health"])
def root():
    return {"message": "VicReel Coqui TTS API", "model_default": tts_manager.default_model}

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

@app.get("/models", tags=["models"])
async def list_models():
    # show which models are loaded
    return {"loaded_models": list(tts_manager.models.keys()), "default": tts_manager.default_model}

@app.post("/models/download", tags=["models"], dependencies=[Depends(verify_api_key)])
async def download_model(payload: dict):
    """
    Trigger an explicit download/loading of a given model name to reduce first-request latency.
    JSON body: { "model": "tts_models/fr/ljspeech/vits" }
    """
    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model is required in body")
    await tts_manager.get_tts(model)
    return {"status": "ok", "model": model}

@app.post("/tts", tags=["tts"], dependencies=[Depends(verify_api_key)])
async def generate_tts(request: Request, background_tasks: BackgroundTasks):
    """
    Accepts JSON { "text": "...", "model": "optional model name", "format":"mp3" }
    Or form-data with same fields.
    Returns file stream (mp3/wav).
    """
    # read body (JSON or form)
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text")
        model = body.get("model")
        out_fmt = body.get("format", "mp3")
    else:
        form = await request.form()
        text = form.get("text")
        model = form.get("model")
        out_fmt = form.get("format", "mp3")

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    job_id = str(uuid.uuid4())
    wav_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    out_path = wav_path if out_fmt == "wav" else os.path.join(OUTPUT_DIR, f"{job_id}.{out_fmt}")

    # limit concurrency
    await _inference_semaphore.acquire()
    try:
        # synth (blocking) executed in thread pool by TTS manager
        await tts_manager.synth_to_file(text=text, out_wav=wav_path, model_name=model)

        if out_fmt != "wav":
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, convert_wav_to_format, wav_path, out_path, out_fmt)
            # schedule cleanup of both files
            background_tasks.add_task(safe_remove, wav_path)
            background_tasks.add_task(safe_remove, out_path)
            media_type = "audio/mpeg" if out_fmt == "mp3" else "application/octet-stream"
            return FileResponse(out_path, media_type=media_type, filename=os.path.basename(out_path))
        else:
            # wav: schedule cleanup
            background_tasks.add_task(safe_remove, wav_path)
            return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    except Exception as e:
        logger.exception("Error during TTS generation")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _inference_semaphore.release()
