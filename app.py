import os
import uuid
import asyncio
import logging
import json
import inspect
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from TTS.api import TTS
from pydub import AudioSegment
from functools import partial

# -------------------
# Configuration
# -------------------
API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025")
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_LANGUAGE = os.getenv("VICREEL_DEFAULT_LANGUAGE", "fr")
OUTPUT_DIR = os.getenv("VICREEL_OUTPUT_DIR", "outputs")
MAX_CONCURRENCY = int(os.getenv("VICREEL_MAX_CONCURRENCY", "1"))
# Maximum text length limit (default 4000, override via env)
MAX_TEXT_LENGTH = int(os.getenv("VICREEL_MAX_TEXT_LENGTH", "4000"))

# Path to aliases file (prioritized). You can override with env var.
ALIASES_FILE_PATH = os.getenv("VICREEL_SPEAKER_ALIASES_FILE", "/app/config/speaker_aliases.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vicreel")

# FastAPI App
app = FastAPI(title="VicReel - Coqui TTS API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# API Key dependency
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Validate API key. If no API key is configured on the server (empty string), accept all requests.
    If an API key is configured, header must be present and match."""
    if not API_KEY:
        return True
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True


# -------------------
# TTS Manager
# -------------------
class TTSManager:
    def __init__(self, default_model: str):
        self.default_model = default_model
        self._models: dict[str, TTS] = {}
        self._lock = asyncio.Lock()

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

    async def get_all_speakers(self, model_name: Optional[str] = None):
        tts = await self.get(model_name)
        return getattr(tts, 'speakers', []) or getattr(tts, 'voices', []) or []

    async def synth_to_wav(
        self,
        text: str,
        wav_path: str,
        model_name: Optional[str] = None,
        language: str = DEFAULT_LANGUAGE,
        speaker_wav: Optional[str] = None,
        speaker: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ):
        """
        Synthesizes text to WAV file.
        - speaker_wav: path to a reference wav for xtts voice cloning
        - speaker: speaker id (from model) to use
        - options: additional kwargs to pass to tts.tts_to_file (filtered)
        """
        tts = await self.get(model_name)
        loop = asyncio.get_event_loop()
        effective_model = (model_name or self.default_model).lower()

        # Base kwargs
        kwargs: Dict[str, Any] = {"text": text, "file_path": wav_path}

        # XTTS specific: language and speaker selection
        if "xtts" in effective_model:
            kwargs["language"] = language
            # priority: speaker_wav -> explicit speaker id -> default speakers list
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            elif speaker:
                kwargs["speaker"] = speaker
            else:
                speakers = getattr(tts, 'speakers', []) or getattr(tts, 'voices', [])
                if speakers:
                    kwargs["speaker"] = speakers[0]
                    logger.info(f"Using default speaker: {kwargs['speaker']}")
                else:
                    logger.warning("No speakers/voices attribute found on model; proceeding without explicit speaker")
        else:
            # For non-XTTS models, allow passing a speaker if supported
            if speaker:
                kwargs["speaker"] = speaker

        # Merge 'options' into kwargs after filtering supported keys to avoid TypeError
        if options:
            func = getattr(tts, 'tts_to_file', None)
            if func is None:
                logger.warning("Model does not expose tts_to_file; ignoring options.")
            else:
                try:
                    sig = inspect.signature(func)
                    supported = set(sig.parameters.keys())
                    # Keep only supported option keys
                    filtered = {k: v for k, v in options.items() if k in supported}
                    if filtered:
                        kwargs.update(filtered)
                    else:
                        logger.debug(f"No provided options were supported by tts_to_file: {list(options.keys())}")
                except Exception as e:
                    logger.exception(f"Failed to inspect tts_to_file signature: {e}")

        logger.debug(f"Prepared tts kwargs: {kwargs}")

        # Run blocking tts_to_file in executor
        func = getattr(tts, 'tts_to_file', None)
        if func is None:
            raise RuntimeError("Model object does not implement tts_to_file")
        await loop.run_in_executor(None, partial(func, **kwargs))

        # Validate generated file
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(f"Generated file {wav_path} is empty or too small")
        logger.info(f"Generated audio at {wav_path}")


tts_manager = TTSManager(DEFAULT_MODEL)
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)


# -------------------
# Aliases: prefer file-based aliases if available, otherwise read from env
# -------------------
def _is_valid_alias_map(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, str):
            return False
    return True

def _load_aliases_from_file(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if _is_valid_alias_map(data):
            logger.info(f"Loaded {len(data)} speaker aliases from file {path}")
            return data
        else:
            logger.warning(f"Aliases file {path} has invalid format; expected mapping str->str. Ignoring.")
    except FileNotFoundError:
        logger.info(f"Aliases file not found at {path}; will fallback to environment variable or empty mapping.")
    except json.JSONDecodeError as e:
        logger.exception(f"Aliases file {path} contains invalid JSON: {e}")
    except Exception as e:
        logger.exception(f"Failed to load aliases file {path}: {e}")
    return {}

def _load_aliases_from_env() -> Dict[str, str]:
    raw = os.getenv("VICREEL_SPEAKER_ALIASES", "") or ""
    if not raw:
        logger.info("No VICREEL_SPEAKER_ALIASES env var set; using empty alias map.")
        return {}
    try:
        data = json.loads(raw)
        if _is_valid_alias_map(data):
            logger.info(f"Loaded {len(data)} speaker aliases from VICREEL_SPEAKER_ALIASES env var")
            return data
        else:
            logger.warning("VICREEL_SPEAKER_ALIASES env var JSON has invalid format; expected mapping str->str. Ignoring.")
    except json.JSONDecodeError as e:
        logger.exception(f"Failed to parse VICREEL_SPEAKER_ALIASES env var as JSON: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error parsing VICREEL_SPEAKER_ALIASES env var: {e}")
    return {}

def _save_aliases_to_file(path: str, aliases: Dict[str, str]):
    """
    Atomic save: write to temp file then os.replace -> avoids corrupt file.
    If filesystem is read-only or not writable, raise/log but keep changes in-memory.
    """
    try:
        dirp = os.path.dirname(path)
        if dirp and not os.path.exists(dirp):
            os.makedirs(dirp, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(aliases, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        logger.info(f"Persisted {len(aliases)} aliases to {path}")
    except PermissionError:
        logger.warning(f"Permission denied writing aliases file {path}; changes will remain in-memory.")
    except Exception as e:
        logger.exception(f"Failed to write aliases file {path}: {e}")

# Load aliases: priority -> file, then env var, else empty map
SPEAKER_ALIASES = _load_aliases_from_file(ALIASES_FILE_PATH)
if not SPEAKER_ALIASES:
    SPEAKER_ALIASES = _load_aliases_from_env()
# At this point SPEAKER_ALIASES is a validated dict (possibly empty)
logger.info(f"Effective speaker aliases loaded: {len(SPEAKER_ALIASES)} entries (file priority: {ALIASES_FILE_PATH})")


def _apply_alias(speaker_id: str) -> str:
    """Return display name for a speaker id (alias if exists)."""
    if not speaker_id:
        return speaker_id
    return SPEAKER_ALIASES.get(speaker_id, speaker_id)


# -------------------
# Helpers
# -------------------
def _convert_wav_to_mp3(src: str, dst: str):
    audio = AudioSegment.from_wav(src)
    audio.export(dst, format="mp3")


def _safe_remove(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.debug(f"Removed temporary file {path}")
    except Exception as e:
        logger.warning(f"Could not delete {path}: {e}")


# -------------------
# Routes
# -------------------
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


@app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(model: Optional[str] = None):
    """
    Return available voices/speakers for the model (speaker id + display name).
    """
    tts = await tts_manager.get(model)
    speakers = getattr(tts, 'speakers', []) or getattr(tts, 'voices', []) or []
    result = [{"id": s, "display_name": _apply_alias(s)} for s in speakers]
    return {"model": model or DEFAULT_MODEL, "voices": result, "count": len(result)}


@app.get("/languages", dependencies=[Depends(verify_api_key)])
async def list_languages(model: Optional[str] = None):
    """
    Return languages supported by the model when available.
    """
    tts = await tts_manager.get(model)
    langs = getattr(tts, 'languages', None) or getattr(tts, 'supported_languages', None)
    if not langs:
        # fallback to commonly supported languages for XTTS-v2
        langs = ["fr", "en", "es", "de", "it", "pt", "nl", "ru", "zh", "ja"]
    return {"model": model or DEFAULT_MODEL, "languages": langs}


@app.post("/voices/aliases", dependencies=[Depends(verify_api_key)])
async def update_aliases(body: Dict[str, str]):
    """
    Update speaker aliases at runtime (protected by API key).
    Body : {"speaker_id": "Display Name", ...}
    This will update in-memory mapping and persist to the aliases file if writable.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body expected as mapping speaker_id->display_name")
    # Merge into in-memory aliases
    SPEAKER_ALIASES.update(body)
    logger.info(f"Updated {len(body)} speaker aliases (in-memory)")

    # Try to persist to file (if path is writable/available)
    try:
        _save_aliases_to_file(ALIASES_FILE_PATH, SPEAKER_ALIASES)
    except Exception:
        logger.warning("Could not persist aliases to file; changes remain in-memory only")

    return {"status": "ok", "aliases_count": len(SPEAKER_ALIASES)}


@app.post("/tts", dependencies=[Depends(verify_api_key)])
async def tts_endpoint(request: Request, background_tasks: BackgroundTasks):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text")
        fmt = (body.get("format") or "wav").lower()
        model = body.get("model", None)
        language = body.get("language", DEFAULT_LANGUAGE)
        speaker_wav = body.get("speaker_wav", None)
        # Accept speaker as either real id or alias; if alias provided, map it back to real id
        speaker_input = body.get("speaker", None)
        options = body.get("options", None) or {}
    else:
        form = await request.form()
        text = form.get("text")
        fmt = (form.get("format") or "wav").lower()
        model = form.get("model", None)
        language = form.get("language", DEFAULT_LANGUAGE)
        speaker_wav = form.get("speaker_wav", None)
        speaker_input = form.get("speaker", None)
        options = {}

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # ---------- NEW: enforce maximum text length ----------
    if len(text) > MAX_TEXT_LENGTH:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Le texte fourni dépasse la limite de {MAX_TEXT_LENGTH} caractères.",
                "actual_length": len(text)
            }
        )
    # -----------------------------------------------------

    if fmt not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    # map alias -> real id if necessary
    speaker = None
    if speaker_input:
        # if the provided value matches a recorded alias, map back to real id
        rev_map = {v: k for k, v in SPEAKER_ALIASES.items()}
        speaker = rev_map.get(speaker_input, speaker_input)

    job_id = uuid.uuid4().hex
    wav_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    out_path = wav_path if fmt == "wav" else os.path.join(OUTPUT_DIR, f"{job_id}.mp3")

    await _inference_semaphore.acquire()
    try:
        # pass speaker and options into synth_to_wav (new signature)
        await tts_manager.synth_to_wav(
            text=text,
            wav_path=wav_path,
            model_name=model,
            language=language,
            speaker_wav=speaker_wav,
            speaker=speaker,
            options=options
        )

        if fmt != "wav":
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _convert_wav_to_mp3, wav_path, out_path)
            background_tasks.add_task(_safe_remove, wav_path)
            background_tasks.add_task(_safe_remove, out_path)
            return FileResponse(out_path, media_type="audio/mpeg", filename=os.path.basename(out_path))
        else:
            background_tasks.add_task(_safe_remove, wav_path)
            return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("TTS error")
        raise HTTPException(status_code=500, detail="Internal TTS error")
    finally:
        _inference_semaphore.release()
