
tee worker_synthesize.py <<'EOF'
#!/usr/bin/env python3
# worker_synthesize.py - File-queue worker pour VicReel TTS
import os
import time
import json
import uuid
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
import unicodedata
import difflib
from TTS.api import TTS
from pydub import AudioSegment

# Config
OUTPUT_DIR = os.getenv("VICREEL_OUTPUT_DIR", "outputs")
JOBS_DIR = os.getenv("VICREEL_JOBS_DIR", "jobs")
RAW_ALIASES_FILE = os.getenv("VICREEL_SPEAKER_ALIASES_FILE", "config/speaker_aliases.json")
RESOLVED_ALIASES_FILE = os.getenv("VICREEL_RESOLVED_ALIASES", "config/resolved_aliases.json")
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
POLL_INTERVAL = float(os.getenv("VICREEL_JOBS_POLL_SEC", "1.0"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)
os.makedirs(Path(RAW_ALIASES_FILE).parent, exist_ok=True)
os.makedirs(Path(RESOLVED_ALIASES_FILE).parent, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vicreel-worker")

def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("Failed to load JSON %s", path)
        return {}

def atomic_write_json(path: str, obj: Dict[str, Any]):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("-", " ").replace("_", " ").replace("'", " ").replace('"', " ")
    import re
    s = re.sub(r"[^a-z0-9\s]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_match_key_to_available(raw_key: str, available: List[str]) -> Optional[str]:
    if not raw_key:
        return None
    if raw_key in available:
        return raw_key
    low = raw_key.strip().lower()
    for a in available:
        if a.lower() == low:
            return a
    norm_raw = _normalize_text(raw_key)
    # ---- CORRECTION ICI : création correcte du mapping normalisé -> réel ----
    norm_map = {_normalize_text(a): a for a in available}
    if norm_raw in norm_map:
        return norm_map[norm_raw]
    for a in available:
        an = _normalize_text(a)
        if norm_raw and (norm_raw in an or an in norm_raw):
            return a
    candidates = list(norm_map.keys())
    if candidates:
        match = difflib.get_close_matches(norm_raw, candidates, n=1, cutoff=0.75)
        if match:
            return norm_map[match[0]]
    return None

def synth_sync(tts_obj, text: str, wav_path: str, model_name: Optional[str], language: str,
               speaker_wav: Optional[str], speaker_real: Optional[str], options: Optional[Dict[str, Any]] = None):
    kwargs = {"text": text, "file_path": wav_path}
    effective_model = (model_name or DEFAULT_MODEL).lower()
    if "xtts" in effective_model:
        kwargs["language"] = language
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        elif speaker_real:
            kwargs["speaker"] = speaker_real
        else:
            speakers = getattr(tts_obj, "speakers", []) or getattr(tts_obj, "voices", []) or []
            if speakers:
                kwargs["speaker"] = speakers[0]
    else:
        if speaker_real:
            kwargs["speaker"] = speaker_real

    if options:
        try:
            import inspect
            func = getattr(tts_obj, "tts_to_file", None)
            if func:
                sig = inspect.signature(func)
                allowed = set(sig.parameters.keys())
                kwargs.update({k: v for k, v in options.items() if k in allowed})
        except Exception:
            logger.exception("options filtering failed")

    func = getattr(tts_obj, "tts_to_file", None)
    if func is None:
        raise RuntimeError("Model object missing tts_to_file")
    func(**kwargs)

def process_job_file(path: str, tts_obj):
    try:
        with open(path, "r", encoding="utf-8") as f:
            job = json.load(f)
    except Exception:
        logger.exception("Failed to read job file %s", path)
        return

    job_id = job.get("id") or str(uuid.uuid4())
    job["id"] = job_id
    status_path = os.path.join(JOBS_DIR, f"{job_id}.status.json")
    wav_out = os.path.join(OUTPUT_DIR, f"{job_id}.wav")
    mp3_out = os.path.join(OUTPUT_DIR, f"{job_id}.mp3")

    atomic_write_json(status_path, {"job_id": job_id, "state": "started", "ts": time.time()})
    try:
        model_name = job.get("model") or DEFAULT_MODEL
        language = job.get("language") or "fr"
        speaker_input = job.get("speaker")
        speaker_real = job.get("speaker_real", None)
        available = getattr(tts_obj, "speakers", []) or getattr(tts_obj, "voices", []) or []

        raw_aliases = load_json(RAW_ALIASES_FILE)
        resolved_map = load_json(RESOLVED_ALIASES_FILE)

        if speaker_real is None and speaker_input:
            if speaker_input in available:
                speaker_real = speaker_input
            else:
                for r, display in resolved_map.items():
                    if _normalize_text(display) == _normalize_text(speaker_input):
                        speaker_real = r
                        break
            if not speaker_real:
                for raw_key, display in raw_aliases.items():
                    if _normalize_text(raw_key) == _normalize_text(speaker_input) or _normalize_text(display) == _normalize_text(speaker_input):
                        matched = fuzzy_match_key_to_available(raw_key, available)
                        if matched:
                            speaker_real = matched
                            break
            if not speaker_real:
                speaker_real = fuzzy_match_key_to_available(speaker_input, available)

        if raw_aliases and available:
            discovered = {}
            for raw_key, display in raw_aliases.items():
                matched = fuzzy_match_key_to_available(raw_key, available)
                if matched and matched not in resolved_map:
                    discovered[matched] = display
            if discovered:
                resolved_map.update(discovered)
                atomic_write_json(RESOLVED_ALIASES_FILE, resolved_map)
                logger.info("Added %d new resolved aliases to %s", len(discovered), RESOLVED_ALIASES_FILE)

        synth_sync(tts_obj, job.get("text", ""), wav_out, model_name, language, job.get("speaker_wav"), speaker_real, job.get("options"))

        fmt = job.get("format", "wav").lower()
        out_path = wav_out
        if fmt != "wav":
            try:
                AudioSegment.from_wav(wav_out).export(mp3_out, format="mp3")
                out_path = mp3_out
                os.remove(wav_out)
            except Exception:
                logger.exception("mp3 conversion failed")
                atomic_write_json(status_path, {"job_id": job_id, "state": "error", "ts": time.time(), "message": "mp3 conversion failed"})
                return

        atomic_write_json(status_path, {"job_id": job_id, "state": "done", "ts": time.time(), "output": out_path, "speaker_real": speaker_real})
        logger.info("Job %s done -> %s (speaker_real=%s)", job_id, out_path, speaker_real)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        atomic_write_json(status_path, {"job_id": job_id, "state": "error", "ts": time.time(), "message": str(e), "trace": traceback.format_exc()})
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def main():
    logger.info("Starting worker — jobs dir: %s, outputs: %s, model: %s", JOBS_DIR, OUTPUT_DIR, DEFAULT_MODEL)
    tts_obj = TTS(DEFAULT_MODEL)
    logger.info("Model loaded; speakers: %d", len(getattr(tts_obj, "speakers", []) or []))
    while True:
        try:
            candidates = sorted(Path(JOBS_DIR).glob("*.json"), key=lambda p: p.stat().st_mtime)
            for p in candidates:
                inprog = p.with_suffix(".inprogress")
                try:
                    os.replace(p, inprog)
                    process_job_file(str(inprog), tts_obj)
                except FileNotFoundError:
                    continue
                except Exception:
                    logger.exception("Failed processing %s", p)
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Exiting worker")
            break
        except Exception:
            logger.exception("Main loop error")
            time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
EOF
