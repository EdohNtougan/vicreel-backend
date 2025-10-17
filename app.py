# app.py — VicReel (version with alias reconciliation, timeout, purge, metrics, rate-limit, executor pool)
import os
import uuid
import asyncio
import logging
import json
import inspect
import time
import re
import unicodedata
import difflib
import traceback
from typing import Optional, Dict, Any, List
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue

from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydub import AudioSegment
from functools import partial

# NOTE: TTS import is done lazily inside worker or manager to avoid heavy import at module load

# -------------------
# Configuration (env-overridable)
# -------------------
API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025")
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
DEFAULT_LANGUAGE = os.getenv("VICREEL_DEFAULT_LANGUAGE", "fr")
OUTPUT_DIR = os.getenv("VICREEL_OUTPUT_DIR", "outputs")
MAX_CONCURRENCY = int(os.getenv("VICREEL_MAX_CONCURRENCY", "1"))

# Text length limit
MAX_TEXT_LENGTH = int(os.getenv("VICREEL_MAX_TEXT_LENGTH", "3000"))

# Timeout for a single synth call (seconds)
SYNTH_TIMEOUT_SECONDS = int(os.getenv("VICREEL_SYNTH_TIMEOUT", "300"))

# Executor workers for CPU-bound TTS calls (and for non-TTS tasks)
EXECUTOR_WORKERS = int(os.getenv("VICREEL_EXECUTOR_WORKERS", "4"))

# File cleanup settings
PERSIST_OUTPUT_MAX_AGE = int(os.getenv("VICREEL_OUTPUT_MAX_AGE", str(60 * 10)))  # seconds; default 10 minutes
PERSIST_OUTPUT_MAX_FILES = int(os.getenv("VICREEL_OUTPUT_MAX_FILES", "100"))  # max files to keep

# Rate limiting (per API key): window seconds and max requests in window
RATE_LIMIT_WINDOW = int(os.getenv("VICREEL_RATE_LIMIT_WINDOW", "60"))  # seconds
RATE_LIMIT_MAX = int(os.getenv("VICREEL_RATE_LIMIT_MAX", "30"))  # requests per window per key

# Path to aliases file (prioritized). You can override with env var.
ALIASES_FILE_PATH = os.getenv("VICREEL_SPEAKER_ALIASES_FILE", "config/speaker_aliases.json")
RESOLVED_ALIASES_FILE_PATH = os.getenv("VICREEL_RESOLVED_ALIASES_FILE", "config/resolved_aliases.json")

# Use subprocess-based synthesis for isolation (1 = enabled). Can set to "0" to disable.
USE_SUBPROCESS = os.getenv("VICREEL_USE_SUBPROCESS", "1") == "1"

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
# Executor pool (shared) - used for non-TTS tasks and fallback executor
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
    key = api_key_value or "ANON"
    now = time.time()
    async with _rate_limit_lock:
        dq = _rate_limit_store.get(key)
        if dq is None:
            dq = deque()
            _rate_limit_store[key] = dq
        while dq and dq[0] <= now - RATE_LIMIT_WINDOW:
            dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            retry_after = int(RATE_LIMIT_WINDOW - (now - dq[0])) if dq else RATE_LIMIT_WINDOW
            log_event("warning", {"event": "rate_limited", "key": key, "count": len(dq)})
            return JSONResponse(status_code=429, content={"error": "rate limit exceeded", "retry_after": retry_after})
        dq.append(now)
    return None

# -------------------
# Helper: child worker synth function (runs in separate process)
# -------------------
def _worker_synthesize_process(params: Dict[str, Any], result_queue: "Queue"):
    """
    Child process entrypoint. Runs TTS synthesis synchronously and reports result via result_queue.
    params: dict containing 'model', 'kwargs' (kwargs for tts_to_file)
    Puts {'ok': True} on success, or {'ok': False, 'error': "...", 'trace': "..."} on failure.
    """
    try:
        # Import TTS lazily inside child process to isolate memory
        from TTS.api import TTS as _TTS_child  # local alias
        model_name = params.get("model")
        kwargs = params.get("kwargs", {})
        # Construct TTS instance (will load model or reuse cached in child)
        tts = _TTS_child(model_name)
        # call to tts_to_file (blocking)
        tts.tts_to_file(**kwargs)
        result_queue.put({"ok": True})
    except Exception as e:
        tb = traceback.format_exc()
        try:
            result_queue.put({"ok": False, "error": str(e), "trace": tb})
        except Exception:
            # If queue.put fails, there's nothing we can do
            pass

# -------------------
# TTS Manager with process-isolated synthesis (fallback to executor)
# -------------------
class TTSManager:
    def __init__(self, default_model: str, executor: ThreadPoolExecutor):
        self.default_model = default_model
        self._models: dict[str, Any] = {}  # store TTS instances if we create in-process
        self._lock = asyncio.Lock()
        self._executor = executor

    async def get(self, model_name: Optional[str] = None):
        name = model_name or self.default_model
        if name in self._models:
            return self._models[name]
        async with self._lock:
            if name in self._models:
                return self._models[name]
            loop = asyncio.get_event_loop()
            logger.info(f"Loading model {name} (this may take some time)")
            # Create model instance in-process (for introspection: speakers/languages)
            def ctor():
                from TTS.api import TTS
                return TTS(name)
            tts_inst = await loop.run_in_executor(self._executor, ctor)
            self._models[name] = tts_inst
            logger.info(f"Model {name} loaded")
            # Try to reconcile aliases now that model speakers are available (if function present)
            try:
                available = getattr(tts_inst, 'speakers', []) or getattr(tts_inst, 'voices', []) or []
                if available:
                    reconcile_aliases_with_model_available(available)
            except Exception:
                logger.exception("Alias reconciliation after model load failed")
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
        Isolated synthesis:
         - Default: spawn a child process which imports TTS and runs tts_to_file.
         - Fallback: if subprocess not possible, run tts.tts_to_file inside executor (thread).
        """
        # Build kwargs for child/fallback call
        kwargs: Dict[str, Any] = {"text": text, "file_path": wav_path}
        effective_model = (model_name or self.default_model).lower()

        if "xtts" in effective_model:
            kwargs["language"] = language
            if speaker_wav:
                kwargs["speaker_wav"] = speaker_wav
            elif speaker:
                kwargs["speaker"] = speaker
            else:
                # try to pick default speaker
                tts_obj = await self.get(model_name)
                speakers = getattr(tts_obj, 'speakers', []) or getattr(tts_obj, 'voices', [])
                if speakers:
                    kwargs["speaker"] = speakers[0]
                    logger.info(f"Using default speaker: {kwargs['speaker']}")
                else:
                    logger.warning("No speakers/voices attribute found on model; proceeding without explicit speaker")
        else:
            if speaker:
                kwargs["speaker"] = speaker

        # Merge options filtering supported kwargs if possible
        if options:
            # Try to get signature from in-process model if available
            func_sig_supported = None
            try:
                # attempt to find supported keys via model instance (if loaded)
                tts_obj = self._models.get(model_name or self.default_model)
                func = getattr(tts_obj, 'tts_to_file', None) if tts_obj else None
                if func:
                    sig = inspect.signature(func)
                    func_sig_supported = {k for k in sig.parameters.keys()}
            except Exception:
                func_sig_supported = None
            if func_sig_supported:
                filtered = {k: v for k, v in options.items() if k in func_sig_supported}
            else:
                filtered = options
            if filtered:
                kwargs.update(filtered)

        log_event("info", {"event": "synth_prepare", "model": model_name or self.default_model, "kwargs_keys": list(kwargs.keys()), "text_len": len(text)})
        logger.info(f"Synth kwargs: {kwargs}")

        # If configured, run in child process for isolation
        if USE_SUBPROCESS:
            result_q = Queue()
            params = {"model": model_name or self.default_model, "kwargs": kwargs}
            proc = Process(target=_worker_synthesize_process, args=(params, result_q))
            proc.start()
            loop = asyncio.get_event_loop()
            try:
                # Wait for child result with timeout using run_in_executor to avoid blocking event loop
                # We'll call result_q.get() inside a thread and wrap with asyncio.wait_for
                get_in_thread = partial(result_q.get, True)
                result = await asyncio.wait_for(loop.run_in_executor(None, get_in_thread), timeout=SYNTH_TIMEOUT_SECONDS)
                # result is expected to be dict {"ok": True} or {"ok": False, "error": "..."}
                if not isinstance(result, dict) or not result.get("ok", False):
                    err = result.get("error", "<no error provided>") if isinstance(result, dict) else str(result)
                    trace = result.get("trace", None) if isinstance(result, dict) else None
                    raise RuntimeError(f"Child synthesis failed: {err}\n{trace or ''}")
                # success: ensure file exists and size OK
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
                    raise RuntimeError(f"Generated file {wav_path} is empty or too small")
                log_event("info", {"event": "synth_done", "wav_path": wav_path, "size": os.path.getsize(wav_path)})
            except asyncio.TimeoutError:
                # kill process
                try:
                    proc.terminate()
                except Exception:
                    logger.exception("Failed to terminate child process after timeout")
                # attempt to join
                try:
                    proc.join(2)
                except Exception:
                    pass
                raise RuntimeError(f"Synthesis timed out after {SYNTH_TIMEOUT_SECONDS} seconds")
            finally:
                # consume/join/close
                try:
                    result_q.close()
                except Exception:
                    pass
                try:
                    proc.join(timeout=1)
                except Exception:
                    pass
        else:
            # Fallback: run in executor (thread)
            try:
                from TTS.api import TTS
                tts = self._models.get(model_name or self.default_model)
                if not tts:
                    # create a tts instance in thread
                    def ctor_tts():
                        return TTS(model_name or self.default_model)
                    loop = asyncio.get_event_loop()
                    tts = await loop.run_in_executor(self._executor, ctor_tts)
                    # store for future introspection
                    self._models[model_name or self.default_model] = tts
                func = getattr(tts, 'tts_to_file', None)
                if func is None:
                    raise RuntimeError("Model object does not implement tts_to_file")
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self._executor, partial(func, **kwargs))
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 1000:
                    raise RuntimeError(f"Generated file {wav_path} is empty or too small")
                log_event("info", {"event": "synth_done", "wav_path": wav_path, "size": os.path.getsize(wav_path)})
            except Exception as e:
                raise

tts_manager = TTSManager(DEFAULT_MODEL, _executor)
_inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

# -------------------
# Aliases loader/persistence (enhanced & tolerant)
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
            logger.info(f"Loaded {len(data)} speaker aliases (raw) from file {path}")
            logger.info(f"Loaded aliases: {data}")  # Added for debug
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

# Raw mapping as-loaded from file/env. Keys may be either real speaker ids OR human names.
RAW_SPEAKER_ALIASES: Dict[str, str] = {}
# Authoritative mapping used for selection: real_speaker_id -> display_name
SPEAKER_ALIASES_REAL: Dict[str, str] = {}

# Derived indexes for client-facing exposure
ALIASKEY_TO_REAL: Dict[str, str] = {}
REAL_TO_ALIASKEY: Dict[str, str] = {}
ALIASKEY_TO_DISPLAY: Dict[str, str] = {}

_slugify_re = re.compile(r"[^a-z0-9_]+")

def _make_alias_key(display_name: str, existing_keys: Optional[set] = None) -> str:
    base = display_name.strip().lower()
    base = re.sub(r"\s+", "_", base)
    base = _slugify_re.sub("_", base)
    base = base.strip("_")
    if not base:
        base = "alias"
    if not existing_keys:
        return base
    key = base
    n = 1
    while key in existing_keys:
        n += 1
        key = f"{base}_{n}"
    return key


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", ' ', s)
    return s.strip()


def _build_alias_indexes_from_real_map():
    ALIASKEY_TO_REAL.clear()
    REAL_TO_ALIASKEY.clear()
    ALIASKEY_TO_DISPLAY.clear()
    existing = set()
    for real in sorted(SPEAKER_ALIASES_REAL.keys()):
        display = SPEAKER_ALIASES_REAL[real]
        key = _make_alias_key(display, existing)
        existing.add(key)
        ALIASKEY_TO_REAL[key] = real
        REAL_TO_ALIASKEY[real] = key
        ALIASKEY_TO_DISPLAY[key] = display
    logger.info(f"Built alias indexes: {len(ALIASKEY_TO_REAL)} aliases available")


def reconcile_aliases_with_model_available(available: List[str]):
    """
    Try to reconcile RAW_SPEAKER_ALIASES (which may be keyed by human names) with the
    model-provided available speaker ids. This fills SPEAKER_ALIASES_REAL.
    """
    logger.info(f"Reconciling {len(RAW_SPEAKER_ALIASES)} raw aliases against {len(available)} model speakers")
    authoritative: Dict[str, str] = {}
    normalized_available = {a: _normalize_text(a) for a in available}
    # also build reverse normalized -> real mapping for quick lookup
    norm_to_real = {}
    for real, norm in normalized_available.items():
        norm_to_real.setdefault(norm, real)

    for raw_key, display in RAW_SPEAKER_ALIASES.items():
        key_norm = _normalize_text(raw_key)
        chosen = None
        # 1) if raw_key exactly matches a real id (case-sensitive or case-insensitive)
        if raw_key in available:
            chosen = raw_key
        else:
            for a in available:
                if _normalize_text(a) == key_norm:
                    chosen = a
                    break
        # 2) contains/startswith
        if not chosen:
            for a in available:
                an = _normalize_text(a)
                if key_norm and (key_norm in an or an in key_norm):
                    chosen = a
                    break
        # 3) fuzzy match on normalized strings
        if not chosen:
            candidates = list(normalized_available.values())
            match = difflib.get_close_matches(key_norm, candidates, n=1, cutoff=0.75)
            if match:
                # find real corresponding
                for real, norm in normalized_available.items():
                    if norm == match[0]:
                        chosen = real
                        break
        if chosen:
            authoritative[chosen] = display
        else:
            logger.warning(f"Could not reconcile alias key '{raw_key}' -> '{display}' to any model speaker; skipping")

    # Keep any existing real->display mappings if they were already present in RAW and use them
    # Also ensure we don't lose explicit real-id entries in RAW if user supplied those
    for raw_key, display in RAW_SPEAKER_ALIASES.items():
        if raw_key in available and raw_key not in authoritative:
            authoritative[raw_key] = display

    SPEAKER_ALIASES_REAL.clear()
    SPEAKER_ALIASES_REAL.update(authoritative)
    _build_alias_indexes_from_real_map()

    # Persist the resolved map
    _save_aliases_to_file(RESOLVED_ALIASES_FILE_PATH, SPEAKER_ALIASES_REAL)


# Load raw aliases (file has priority). We expect keys MAY BE human names OR real speaker ids.
RAW_SPEAKER_ALIASES = _load_aliases_from_file(ALIASES_FILE_PATH)
if not RAW_SPEAKER_ALIASES:
    RAW_SPEAKER_ALIASES = _load_aliases_from_env()

# At startup we don't yet have model speakers, aliases will be reconciled on first model load.
logger.info(f"Loaded raw speaker alias entries: {len(RAW_SPEAKER_ALIASES)} (will try to reconcile when model is available)")

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
        removed = 0
        for path, mtime, _ in entries:
            if now - mtime > max_age_seconds:
                try:
                    os.remove(path)
                    removed += 1
                except Exception:
                    pass
        entries = sorted(entries, key=lambda x: x[1])
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
# Speaker resolution (accept alias_key or display_name; hide real ids)
# -------------------

def _normalize(s: str) -> str:
    return s.strip().lower() if isinstance(s, str) else ""

async def resolve_speaker_input(speaker_input: Optional[str], model_name: Optional[str] = None) -> Optional[str]:
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

    # 3) Fuzzy match against display names
    for ak, display in ALIASKEY_TO_DISPLAY.items():
        dnorm = _normalize(display)
        if dnorm.startswith(low) or low in dnorm:
            return ALIASKEY_TO_REAL[ak]

    # 4) If model present, try to match against model's real speaker ids
    available = []
    try:
        tts_obj = await tts_manager.get(model_name) if model_name else None
        available = getattr(tts_obj, 'speakers', []) or getattr(tts_obj, 'voices', []) or []
    except Exception:
        available = []

    if available:
        for r in available:
            if r == speaker_input:
                return r
        for r in available:
            if _normalize(r) == low or low in _normalize(r) or _normalize(r).startswith(low):
                return r

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
    try:
        tts = await tts_manager.get(model)
        available = getattr(tts, 'speakers', []) or getattr(tts, 'voices', []) or []
    except Exception:
        available = []

    result = []
    # authoritative aliases intersection
    for ak, real in ALIASKEY_TO_REAL.items():
        if not available or real in available:
            result.append({"id": ak, "display_name": ALIASKEY_TO_DISPLAY.get(ak, real)})

    # include available speakers not aliased (ad-hoc temporary alias)
    if available:
        missing = [r for r in available if r not in REAL_TO_ALIASKEY]
        existing_keys = set(ALIASKEY_TO_REAL.keys())
        for r in missing:
            display = r
            tmp_key = _make_alias_key(display, existing_keys)
            existing_keys.add(tmp_key)
            result.append({"id": tmp_key, "display_name": display})

    return {"model": model or DEFAULT_MODEL, "voices": result, "count": len(result)}

@app.get("/languages", dependencies=[Depends(verify_api_key)])
async def list_languages(model: Optional[str] = None):
    try:
        tts = await tts_manager.get(model)
        langs = getattr(tts, 'languages', None) or getattr(tts, 'supported_languages', None)
    except Exception:
        langs = None
    if not langs:
        langs = ["fr", "en", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "ar", "hi"]
    return {"model": model or DEFAULT_MODEL, "languages": langs}

@app.post("/voices/aliases", dependencies=[Depends(verify_api_key)])
async def update_aliases(body: Dict[str, str]):
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body expected as mapping real_speaker_id->display_name or human_name->display_name")
    for k, v in body.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise HTTPException(status_code=400, detail="All keys and values must be strings")
    # Merge into RAW mapping (we persist raw as provided)
    RAW_SPEAKER_ALIASES.update(body)
    try:
        _save_aliases_to_file(ALIASES_FILE_PATH, RAW_SPEAKER_ALIASES)
    except Exception:
        logger.warning("Could not persist aliases to file; changes remain in-memory only")
    # Attempt to reconcile now if model available
    try:
        # get any loaded model to supply available list
        some_model = next(iter(tts_manager._models.values()), None)
        if some_model:
            available = getattr(some_model, 'speakers', []) or getattr(some_model, 'voices', []) or []
            reconcile_aliases_with_model_available(available)
        else:
            # indexes will be rebuilt when model loads
            _build_alias_indexes_from_real_map()
    except Exception:
        logger.exception("Failed to rebuild alias indexes after update")
    return {"status": "ok", "aliases_count": len(RAW_SPEAKER_ALIASES)}

# Cleanup counter: call cleanup every 100 TTS requests
_cleanup_counter = 0
_CLEANUP_INTERVAL = 100  # Every 100 requests

@app.post("/tts", dependencies=[Depends(verify_api_key)])
async def tts_endpoint(request: Request, background_tasks: BackgroundTasks):
    global _cleanup_counter
    api_key_value = request.headers.get("x-api-key")
    rl_resp = await check_rate_limit(api_key_value)
    if rl_resp:
        return rl_resp

    # Call cleanup every _CLEANUP_INTERVAL requests (async via executor)
    _cleanup_counter += 1
    if _cleanup_counter >= _CLEANUP_INTERVAL:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, cleanup_old_files, OUTPUT_DIR, PERSIST_OUTPUT_MAX_AGE, PERSIST_OUTPUT_MAX_FILES)
        _cleanup_counter = 0

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        text = body.get("text")
        fmt = (body.get("format") or "wav").lower()
        model = body.get("model", None)
        language = body.get("language", DEFAULT_LANGUAGE)
        speaker_input = body.get("speaker", None)
        speaker_wav = body.get("speaker_wav", None)
        options = body.get("options", None) or {}
    else:
        form = await request.form()
        text = form.get("text")
        fmt = (form.get("format") or "wav").lower()
        model = body.get("model", None)
        language = body.get("language", DEFAULT_LANGUAGE)
        speaker_input = body.get("speaker", None)
        speaker_wav = body.get("speaker_wav", None)
        options = {}

    await tts_manager.get(model)  # Force model load and alias reconciliation before resolving speaker

    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    if len(text) > MAX_TEXT_LENGTH:
        log_event("warning", {"event": "text_too_long", "length": len(text), "limit": MAX_TEXT_LENGTH})
        return JSONResponse(status_code=400, content={"error": f"Le texte fourni dépasse la limite de {MAX_TEXT_LENGTH} caractères.", "actual_length": len(text)})

    if fmt not in ("wav", "mp3"):
        raise HTTPException(status_code=400, detail="format must be 'wav' or 'mp3'")

    resolved_speaker_real: Optional[str] = None
    if speaker_input:
        resolved_speaker_real = await resolve_speaker_input(speaker_input, model)
        logger.info(f"Resolved speaker input '{speaker_input}' to '{resolved_speaker_real}'")  # Added for debug
        if not resolved_speaker_real:
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

    METRICS["requests_total"] += 1
    start_ts = time.perf_counter()

    await _inference_semaphore.acquire()
    try:
        try:
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

        loop = asyncio.get_event_loop()
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
