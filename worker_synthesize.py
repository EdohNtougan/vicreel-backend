#!/usr/bin/env python3
# worker_synthesize.py - Version avec logique d'alias simplifiée
import os
import time
import json
import uuid
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from TTS.api import TTS
from pydub import AudioSegment
from threading import Lock
from pydantic import BaseModel, Field

# --- Configuration ---
OUTPUT_DIR = Path(os.getenv("VICREEL_OUTPUT_DIR", "outputs"))
JOBS_DIR = Path(os.getenv("VICREEL_JOBS_DIR", "jobs"))
CONFIG_DIR = Path("config")
# MODIFIÉ: On utilise notre nouvelle carte directe
SPEAKER_MAP_FILE = CONFIG_DIR / "speaker_map.json" 
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
POLL_INTERVAL = float(os.getenv("VICREEL_JOBS_POLL_SEC", "1.0"))

# ... (le reste de l'initialisation et les fonctions utilitaires restent les mêmes) ...
OUTPUT_DIR.mkdir(exist_ok=True); JOBS_DIR.mkdir(exist_ok=True); CONFIG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("vicreel-worker")
job_lock = Lock()

class TTSJob(BaseModel):
    id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    text: str; model: str = DEFAULT_MODEL; language: str = "fr"
    speaker: Optional[str] = None # L'utilisateur fournit l'alias propre ici (ex: "bernice_female")
    speaker_wav: Optional[str] = None; speaker_real: Optional[str] = None
    format: str = "wav"; options: Optional[Dict[str, Any]] = None

def atomic_write_json(path: Path, data: Dict[str, Any]):
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception: logger.exception("Failed atomic write to %s", path)

# ... (gardez la fonction synth_audio telle quelle, elle est parfaite) ...
def synth_audio(tts_obj: TTS, job: TTSJob, wav_path: Path):
    kwargs = { "text": job.text, "file_path": str(wav_path), "language": job.language }
    if "xtts" in job.model:
        if job.speaker_wav: kwargs["speaker_wav"] = job.speaker_wav
        elif job.speaker_real: kwargs["speaker"] = job.speaker_real
        else:
            available_speakers = getattr(tts_obj, "speakers", [])
            if available_speakers:
                kwargs["speaker"] = available_speakers[0]
                logger.warning("No speaker resolved, falling back to first available: '%s'", kwargs["speaker"])
    else:
        if job.speaker_real: kwargs["speaker"] = job.speaker_real
    logger.info("Calling TTS.tts_to_file with: %s", kwargs)
    tts_obj.tts_to_file(**kwargs)
    logger.info("TTS synthesis completed successfully.")


def process_job_file(job_path: Path, tts_obj: TTS, speaker_map: Dict[str, str]):
    job_id_from_filename = job_path.stem.replace(".inprogress", "")
    status_path = JOBS_DIR / f"{job_id_from_filename}.status.json"
    
    try:
        job = TTSJob(**json.loads(job_path.read_text(encoding="utf-8")))
        job.id = job_id_from_filename
    except Exception as e:
        logger.error("Job file %s is invalid: %s", job_path, e)
        atomic_write_json(status_path, {"job_id": job_id_from_filename, "state": "error", "message": f"Invalid job JSON: {e}"})
        return

    logger.info("Processing job ID: %s", job.id)
    atomic_write_json(status_path, {"job_id": job.id, "state": "started", "ts": time.time()})

    wav_out = OUTPUT_DIR / f"{job.id}.wav"
    final_out_path = OUTPUT_DIR / f"{job.id}.{job.format}"

    try:
        # --- LOGIQUE D'ALIAS SIMPLIFIÉE ---
        logger.info("Resolving speaker alias: '%s'", job.speaker)
        if job.speaker:
            # Simple, rapide, efficace.
            job.speaker_real = speaker_map.get(job.speaker)
        
        if job.speaker and not job.speaker_real:
            logger.warning("Alias '%s' not found in speaker map. Will use model default.", job.speaker)
        logger.info("Resolved speaker to real ID: '%s'", job.speaker_real)
        # --- FIN DE LA LOGIQUE D'ALIAS ---

        logger.info("Starting audio synthesis for job %s...", job.id)
        synth_audio(tts_obj, job, wav_out)
        
        if not wav_out.exists() or wav_out.stat().st_size == 0:
            raise RuntimeError("Synthesis finished but output file is missing or empty.")

        if job.format != "wav":
            logger.info("Converting %s to %s...", wav_out, job.format)
            AudioSegment.from_wav(wav_out).export(final_out_path, format=job.format)
            wav_out.unlink()
        
        logger.info("✅ Job %s completed. Output: %s", job.id, final_out_path)
        atomic_write_json(status_path, {
            "job_id": job.id, "state": "done", "ts": time.time(),
            "output": str(final_out_path), "speaker_real": job.speaker_real
        })
        job_path.unlink()

    except Exception as e:
        logger.exception("🔴 Job %s failed.", job.id)
        atomic_write_json(status_path, {
            "job_id": job.id, "state": "error", "ts": time.time(),
            "message": str(e), "trace": traceback.format_exc()
        })


def main():
    logger.info("Starting worker — Jobs: %s, Outputs: %s, Model: %s", JOBS_DIR, OUTPUT_DIR, DEFAULT_MODEL)
    
    # Charger la carte d'alias une seule fois au démarrage
    if not SPEAKER_MAP_FILE.exists():
        logger.error("FATAL: Speaker map file not found at %s. Run sync_aliases.py first.", SPEAKER_MAP_FILE)
        return
    speaker_map = json.loads(SPEAKER_MAP_FILE.read_text(encoding="utf-8"))
    logger.info("Loaded %d speaker aliases from %s", len(speaker_map), SPEAKER_MAP_FILE)
    
    try:
        tts_obj = TTS(DEFAULT_MODEL)
        logger.info("Model loaded; %d speakers available.", len(getattr(tts_obj, "speakers", [])))
    except Exception:
        logger.exception("FATAL: Could not load TTS model. Exiting.")
        return

    while True:
        try:
            # ... (la boucle de recherche de jobs reste identique) ...
            candidates = [p for p in JOBS_DIR.glob("*.json") if not p.name.startswith('.') and not p.name.endswith(('.status.json', '.inprogress'))]
            if not candidates:
                time.sleep(POLL_INTERVAL); continue
            
            job_file = sorted(candidates, key=lambda p: p.stat().st_mtime)[0]
            
            with job_lock:
                if not job_file.exists(): continue
                in_progress_path = job_file.with_suffix(".inprogress")
                job_file.rename(in_progress_path)
                # On passe la carte d'alias à la fonction de traitement
                process_job_file(in_progress_path, tts_obj, speaker_map)

        except KeyboardInterrupt:
            logger.info("Worker shutting down."); break
        except Exception:
            logger.exception("An unexpected error occurred in the main loop."); time.sleep(POLL_INTERVAL * 5)

if __name__ == "__main__":
    main()
