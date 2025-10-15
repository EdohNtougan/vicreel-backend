# app.py — VicReel (version avec timeout, purge, metrics, rate-limit, executor pool)
import os
import uuid
import asyncio
import logging
import json
import inspect
import time
import re
from typing import Optional, Dict, Any, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from TTS.api import TTS
from pydub import AudioSegment
from functools import partial

# -------------------
# Configuration (env-overridable)
# -------------------
API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025")
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_LANGUAGE = os.getenv("VICREEL_DEFAULT_LANGUAGE", "fr")
OUTPUT_DIR = os.getenv("VICREEL_OUTPUT_DIR", "outputs")
MAX_CONCURRENCY = int(os.getenv("VICREEL_MAX_CONCURRENCY", "1"))

# Text length limit
MAX_TEXT_LENGTH = int(os.getenv("VICREEL_MAX_TEXT_LENGTH", "4000"))

# Timeout for a single synth call (seconds)
SYNTH_TIMEOUT_SECONDS = int(os.getenv("VICREEL_SYNTH_TIMEOUT", "60"))

# Executor workers for CPU-bound TTS calls
EXECUTOR_WORKERS = int(os.getenv("VICREEL_EXECUTOR_WORKERS", "4"))

# File cleanup settings
PERSIST_OUTPUT_MAX_AGE = int(os.getenv("VICREEL_OUTPUT_MAX_AGE", str(60 * 10)))  # seconds; default 10 minutes
PERSIST_OUTPUT_MAX_FILES = int(os.getenv("VICREEL_OUTPUT_MAX_FILES", "100"))  # max files to keep

# Rate limiting (per API key): window seconds and max requests in window
RATE_LIMIT_WINDOW = int(os.getenv("VICREEL_RATE_LIMIT_WINDOW", "60"))  # seconds
RATE_LIMIT_MAX = int(os.getenv("VICREEL_RATE_LIMIT_MAX", "30"))  # requests per window per key

# Path to aliases file (prioritized). You can override with env var.
ALIASES_FILE_PATH = os.getenv("VICREEL_SPEAKER_ALIASES_FILE", "/app/config/speaker_aliases.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# Logging - basic + structured JSON helper
# -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vicreel")

def log_event(level: str, payload: dict):
    """Write structured JSON log (keeps old behavior available)."""
    try:
        entry = {"timestamp": int(time.time()), **payload}
        if level.lower() == "info":
            logger.info(json.dumps(entry, ensure_ascii=False))
        elif level.lower() == "warning":
            logger.warning(json.dumps(entry, ensure_ascii=False))
        elif level.lower() == "error":
            logger.error(json.dumps(entry, ensure_ascii=False))
        else:
            logger.debug(json.dumps(entry, ensure_ascii=False))
    except Exception:
        logger.exception("Failed to write structured log")

# -------------------
# FastAPI app
# -------------------
app = FastAPI(title="VicReel - Coqui TTS API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    """Validate API key. If no API key is configured on the server (empty string), accept all requests."""
    if not API_KEY:
        return True
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
    return True

# -------------------
# Executor pool (shared)
# -------------------
_executor = ThreadPoolExecutor(max_workers=EXECUTOR_WORKERS)

# -------------------
# Simple in-memory metrics
# -------------------
METRICS = {
    "requests_total": 0,
    "success_total": 0,
    "error_total": 0,
    "timeouts_total": 0,
    "total_duration_seconds": 0.0,
}

# -------------------
# Simple in-memory rate limiter (per API key)
# Uses sliding window with deque timestamps
# -------------------
_rate_limit_store: Dict[str, deque] = {}
_rate_limit_lock = asyncio.Lock()

async def check_rate_limit(api_key_value: str) -> Optional[JSONResponse]:
    """
    Returns a JSONResponse if request must be rejected (429), otherwise None.
    For anonymous clients (no API key), we use a shared key "ANON".
    """
    key = api_key_value or "ANON"
    now = time.time()
    async with _rate_limit_lock:
        dq = _rate_limit_store.get(key)
        if dq is None:
            dq = deque()
            _rate_limit_store[key] = dq
        # prune timestamps older than window
        while dq and dq[0] <= now - RATE_LIMIT_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            # too many requests
            retry_after = int(RATE_LIMIT_WINDOW - (now - dq[0])) if dq else RATE_LIMIT_WINDOW
            log_event("warning", {"event": "rate_limited", "key": key, "count": len(dq)})
            return JSONResponse(status_code=429, content={"error": "rate limit exceeded", "retry_after": retry_after})
        dq.append(now)
    return None

# -------------------
# TTS Manager with executor usage and extended synth signature
# -------------------
class TTSManager:
    def __init__(self, default_model: str, executor: ThreadPoolExecutor):
        self.default_model = default_model
        self._models: dict[str, TTS] = {}
        self._lock = asyncio.Lock()
        self._executor = executor

    async def get(self, model_name: Optional[str] = None) -> TTS:
        name = model_name or self.default_model
        if name in self._models:
            return self._models[name]
        async with self._lock:
            if name in self._models:
                return self._models[name]
            loop = asyncio.get_event_loop()
            logger.info(f"Loading model {name} (this may take some time)")
            ctor = partial(TTS, name)
            tts_inst = await loop.run_in_executor(self._executor, ctor)
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
        tts = await self.get(model_name)
        loop = asyncio.get_event_loop()
        effective_model = (model_name or self.default_model).lower()

        # Build kwargs
        kwargs: Dict[str, Any] = {"text": text, "file_path": wav_path}
        if "xtts" in effective_model:
            kwargs["language"] = language
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
            if speaker:
                kwargs["speaker"] = speaker

        # Merge options (filtering supported kwargs)
        if options:
            func = getattr(tts, 'tts_to_file', None)
            if func:
                try:
                    sig = inspect.signature(func)
                    supported = set(sig.parameters.keys())
                    filtered = {k: v for k, v in options.items() if k in supported}
                    if filtered:
                        kwargs.update(filtered)
                except Exception:
                    logger.exception("Failed to inspect tts_to_file signature; ignoring options")
            else:
                logger.warning("Model has no tts_to_file; ignoring options")

        log_event("info", {"event": "synth_prepare", "model": model_name or self.default_model, "kwargs_keys": list(kwargs.keys()), "text_len": len(text)})

        # run in executor (use pool)
        func = getattr(tts, 'tts_to_file', None)
        if func is None:
            raise RuntimeError("Model object does not implement tts_to_file")
        await loop.run_in_executor(self._executor, partial(func, **kwargs))

        # Validate file
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
            raise RuntimeError(f"Generated file {wav_path} is empty or too small")
        log_event("info", {"event": "synth_done", "wav_path": wav_path, "size": os.path.getsize(wav_path)})

tts_manager = TTSManager(DEFAULT_MODEL, _executor)
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# -------------------
# Aliases loader/persistence (enhanced)
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
            logger.info(f"Loaded {len(data)} speaker aliases (real_id->display) from file {path}")
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

# Helper: slugify / create stable alias keys from display names
_slugify_re = re.compile(r"[^a-z0-9_]+")
def _make_alias_key(display_name: str, existing_keys: Optional[set] = None) -> str:
    base = display_name.strip().lower()
    # replace non-alnum with underscore
    base = re.sub(r"\s+", "_", base)
    base = _slugify_re.sub("_", base)
    base = base.strip("_")
    if not base:
        base = "alias"
    # ensure unique
    if not existing_keys:
        return base
    key = base
    n = 1
    while key in existing_keys:
        n += 1
        key = f"{base}_{n}"
    return key

# The authoritative mapping loaded from file/env: real_speaker_id -> display_name (alias)
SPEAKER_ALIASES: Dict[str, str] = {}
# Derived indexes for quick lookups (built at startup and on updates)
# alias_key -> real_id
ALIASKEY_TO_REAL: Dict[str, str] = {}
# real_id -> alias_key
REAL_TO_ALIASKEY: Dict[str, str] = {}
# alias_key -> display_name
ALIASKEY_TO_DISPLAY: Dict[str, str] = {}

def _build_alias_indexes():
    """
    From SPEAKER_ALIASES (real_id -> display_name) build:
      - ALIASKEY_TO_REAL
      - REAL_TO_ALIASKEY
      - ALIASKEY_TO_DISPLAY
    so we expose ONLY alias_key to clients.
    """
    ALIASKEY_TO_REAL.clear()
    REAL_TO_ALIASKEY.clear()
    ALIASKEY_TO_DISPLAY.clear()
    existing = set()
    # keep deterministic order (sorted by real id)
    for real in sorted(SPEAKER_ALIASES.keys()):
        display = SPEAKER_ALIASES[real]
        key = _make_alias_key(display, existing)
        existing.add(key)
        ALIASKEY_TO_REAL[key] = real
        REAL_TO_ALIASKEY[real] = key
        ALIASKEY_TO_DISPLAY[key] = display
    logger.info(f"Built alias indexes: {len(ALIASKEY_TO_REAL)} aliases available")

# Load initial aliases (file preferred)
SPEAKER_ALIASES = _load_aliases_from_file(ALIASES_FILE_PATH)
if not SPEAKER_ALIASES:
    SPEAKER_ALIASES = _load_aliases_from_env()
_build_alias_indexes()
logger.info(f"Effective speaker aliases loaded: {len(SPEAKER_ALIASES)} entries (file priority: {ALIASES_FILE_PATH})")

def _apply_alias_to_display(real_id: str) -> str:
    """Return display alias for a real speaker id (falls back to real_id if none)."""
    if not real_id:
        return real_id
    return SPEAKER_ALIASES.get(real_id, real_id)

# -------------------
# Helpers: convert & safe remove & purge old files
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

def cleanup_old_files(directory: str, max_age_seconds: int = PERSIST_OUTPUT_MAX_AGE, max_files: int = PERSIST_OUTPUT_MAX_FILES):
    """
    Remove files older than max_age_seconds. If there are more than max_files,
    remove oldest until count <= max_files.
    """
    try:
        now = time.time()
        entries = []
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                try:
                    mtime = os.path.getmtime(path)
                    size = os.path.getsize(path)
                    entries.append((path, mtime, size))
                except Exception:
                    continue
        # Remove old files
        removed = 0
        for path, mtime, _ in entries:
            if now - mtime > max_age_seconds:
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
        # Enforce max files by removing oldest
        entries = sorted(entries, key=lambda x: x[1])  # oldest first
        if len(entries) - removed > max_files:
            excess = (len(entries) - removed) - max_files
            for i in range(excess):
                path = entries[i][0]
                try:
                    os.remove(path)
                except Exception:
                    pass
        if removed:
            log_event("info", {"event": "cleanup_old_files", "removed": removed})
    except Exception:
        logger.exception("cleanup_old_files failed")

# -------------------
# Speaker resolution (accept only alias_key or display_name; hide real ids)
# -------------------
def _normalize(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

async def resolve_speaker_input(speaker_input: Optional[str], model_name: Optional[str] = None) -> Optional[str]:
    """
    Resolve a client-provided speaker_input to the real speaker id expected by Coqui TTS.
    The client should ideally send the alias_key (the ID shown by /voices), but we also accept
    the display_name (alias text) or fuzzy variants (case-insensitive).
    Return real_speaker_id (string) if resolved, or None if no speaker requested.
    Raises HTTPException(400) with available alias sample if the input cannot be resolved.
    """
    if not speaker_input:
        return None

    low = _normalize(speaker_input)

    # 1) If input matches alias key exactly -> map to real id
    if low in ALIASKEY_TO_REAL:
        return ALIASKEY_TO_REAL[low]

    # 2) If input exactly matches a display name (case-insensitive)
    for ak, display in ALIASKEY_TO_DISPLAY.items():
        if _normalize(display) == low:
            return ALIASKEY_TO_REAL[ak]

    # 3) Try fuzzy match against alias display names (prefix/contains)
    for ak, display in ALIASKEY_TO_DISPLAY.items():
        dnorm = _normalize(display)
        if dnorm.startswith(low) or low in dnorm:
            return ALIASKEY_TO_REAL[ak]

    # 4) If model present, try to match against model's real speaker ids (case-insensitive)
    available = []
    try:
        tts_obj = await tts_manager.get(model_name) if model_name else None
        available = getattr(tts_obj, 'speakers', []) or getattr(tts_obj, 'voices', []) or []
    except Exception:
        available = []

    if available:
        # exact real id match
        for r in available:
            if r == speaker_input:
                return r
        for r in available:
            if _normalize(r) == low or low in _normalize(r) or _normalize(r).startswith(low):
                return r

    # 5) Not resolved -> return None (caller will error with available sample)
    return None

# -------------------
# Routes
# -------------------
@app.get("/")
def root():
    return {"message": "VicReel Coqui TTS API", "default_model": DEFAULT_MODEL, "default_language": DEFAULT_LANGUAGE}

@app.get("/health")
def health():
    return {"status": "ok", "loaded_models": list(tts_manager._models.keys())}

@app.get("/metrics")
def metrics_endpoint():
    # return metrics simple JSON
    return METRICS

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
    Return available voices for the model — BUT OBFUSCATE real ids:
      - id: alias_key (stable machine id exposed to clients)
      - display_name: user-friendly alias (what user sees)
    Clients must send alias_key to /tts to select a voice.
    """
    # ensure model loaded to reflect real availability
    try:
        tts = await tts_manager.get(model)
        available = getattr(tts, 'speakers', []) or getattr(tts, 'voices', []) or []
    except Exception:
        available = []

    # Present only aliases (intersection of available real ids and our alias map)
    result = []
    # If some real speakers are present but not aliased, we still provide a default display name
    for ak, real in ALIASKEY_TO_REAL.items():
        if not available or real in available:
            result.append({"id": ak, "display_name": ALIASKEY_TO_DISPLAY.get(ak, real)})
    # Also include any available real speakers not in alias file (but hide real id: create ad-hoc alias)
    if available:
        # find reals without alias
        missing = [r for r in available if r not in REAL_TO_ALIASKEY]
        existing_keys = set(ALIASKEY_TO_REAL.keys())
        for r in missing:
            # create temporary alias key from real id (not persisted)
            display = r
            tmp_key = _make_alias_key(display, existing_keys)
            existing_keys.add(tmp_key)
            result.append({"id": tmp_key, "display_name": display})
    return {"model": model or DEFAULT_MODEL, "voices": result, "count": len(result)}

@app.get("/languages", dependencies=[Depends(verify_api_key)])
async def list_languages(model: Optional[str] = None):
    """
    Return languages supported by the model when available.
    """
    try:
        tts = await tts_manager.get(model)
        langs = getattr(tts, 'languages', None) or getattr(tts, 'supported_languages', None)
    except Exception:
        langs = None
    if not langs:
        langs = ["fr", "en", "es", "de", "it", "pt", "nl", "ru", "zh", "ja"]
    return {"model": model or DEFAULT_MODEL, "languages": langs}

@app.post("/voices/aliases", dependencies=[Depends(verify_api_key)])
async def update_aliases(body: Dict[str, str]):
    """
    Update speaker aliases at runtime (protected by API key).
    Expected body format: { "<real_speaker_id>": "<display_name>", ... }
    This updates in-memory mapping and persists to ALIASES_FILE_PATH if possible.
    After update we rebuild alias indexes which change what /voices returns.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body expected as mapping real_speaker_id->display_name")
    # Validate entries are strings
    for k, v in body.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise HTTPException(status_code=400, detail="All keys and values must be strings")
    # Merge into authoritative mapping
    SPEAKER_ALIASES.update(body)
    # Persist
    try:
        _save_aliases_to_file(ALIASES_FILE_PATH, SPEAKER_ALIASES)
    except Exception:
        logger.warning("Could not persist aliases to file; changes remain in-memory only")
    # Rebuild indexes
    _build_alias_indexes()
    logger.info(f"Updated speaker aliases (total {len(SPEAKER_ALIASES)})")
    return {"status": "ok", "aliases_count": len(SPEAKER_ALIASES)}

@app.post("/tts", dependencies=[Depends(verify_api_key)])
async def tts_endpoint(request: Request, background_tasks: BackgroundTasks):
    # Quick rate-limit check (per API key)
    api_key_value = request.headers.get("x-api-key")
    rl_resp = await check_rate_limit(api_key_value)
    if rl_resp:
        return rl_resp

    # Kick off a quick cleanup in executor so we don't block synthesis
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, cleanup_old_files, OUTPUT_DIR, PERSIST_OUTPUT_MAX_AGE, PERSIST_OUTPUT_MAX_FILES)

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text")
        fmt = (body.get("format") or "wav").lower()
        model = body.get("model", None)
        language = body.get("language", DEFAULT_LANGUAGE)
        speaker_wav = body.get("speaker_wav", None)
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

    # enforce maximum text length
    if len(text) > MAX_TEXT_LENGTH:
        log_event("warning", {"event": "text_too_long", "length": len(text), "limit": MAX_TEXT_LENGTH})
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Le texte fourni dépasse la limite de {MAX_TEXT_LENGTH} caractères.",
                "actual_length": len(text)
            }
        )

    if fmt not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    # Resolve speaker_input to real_id using aliases (client should send alias_key)
    resolved_speaker_real: Optional[str] = None
    if speaker_input:
        resolved_speaker_real = await resolve_speaker_input(speaker_input, model)
        if not resolved_speaker_real:
            # Provide helpful sample of available aliases (first 40)
            sample = []
            for ak, display in list(ALIASKEY_TO_DISPLAY.items())[:40]:
                sample.append({"alias_id": ak, "display_name": display})
            raise HTTPException(status_code=400, detail={
                "error": "Requested speaker not found or ambiguous",
                "requested": speaker_input,
                "available_sample": sample,
                "hint": "Call GET /voices to get the alias_id to use, or update aliases via /voices/aliases"
            })

    job_id = uuid.uuid4().hex
    wav_path = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    out_path = wav_path if fmt == "wav" else os.path.join(OUTPUT_DIR, f"{job_id}.mp3")

    # Metrics: count request
    METRICS["requests_total"] += 1
    start_ts = time.perf_counter()

    # Concurrency control + timeout/cancellation
    await _inference_semaphore.acquire()
    try:
        try:
            # Use asyncio.wait_for to enforce synth timeout
            await asyncio.wait_for(
                tts_manager.synth_to_wav(
                    text=text,
                    wav_path=wav_path,
                    model_name=model,
                    language=language,
                    speaker_wav=speaker_wav,
                    speaker=resolved_speaker_real,
                    options=options
                ),
                timeout=SYNTH_TIMEOUT_SECONDS
            )
            duration = time.perf_counter() - start_ts
            METRICS["success_total"] += 1
            METRICS["total_duration_seconds"] += duration
            log_event("info", {"event": "tts_success", "duration": duration, "text_length": len(text), "model": model or DEFAULT_MODEL})
        except asyncio.TimeoutError:
            METRICS["timeouts_total"] += 1
            log_event("error", {"event": "tts_timeout", "timeout_seconds": SYNTH_TIMEOUT_SECONDS, "text_length": len(text)})
            raise HTTPException(status_code=504, detail=f"TTS synthesis timed out after {SYNTH_TIMEOUT_SECONDS} seconds")
        except HTTPException:
            raise
        except Exception as e:
            METRICS["error_total"] += 1
            log_event("error", {"event": "tts_error", "error": str(e)})
            logger.exception("TTS error (synthesis)")
            raise HTTPException(status_code=500, detail="Internal TTS error")

        # convert if necessary and return (we keep cleanup via background tasks)
        if fmt != "wav":
            await loop.run_in_executor(_executor, _convert_wav_to_mp3, wav_path, out_path)
            background_tasks.add_task(_safe_remove, wav_path)
            background_tasks.add_task(_safe_remove, out_path)
            return FileResponse(out_path, media_type="audio/mpeg", filename=os.path.basename(out_path))
        else:
            background_tasks.add_task(_safe_remove, wav_path)
            return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))

    finally:
        _inference_semaphore.release()
