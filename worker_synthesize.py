#!/usr/bin/env python3
# worker_synthesize.py - File-queue worker pour VicReel TTS (Version Robuste)
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
from threading import Lock
from pydantic import BaseModel, Field # NOUVEAU: Validation robuste

# --- Configuration ---
OUTPUT_DIR = Path(os.getenv("VICREEL_OUTPUT_DIR", "outputs"))
JOBS_DIR = Path(os.getenv("VICREEL_JOBS_DIR", "jobs"))
CONFIG_DIR = Path("config")
RAW_ALIASES_FILE = CONFIG_DIR / "speaker_aliases.json"
RESOLVED_ALIASES_FILE = CONFIG_DIR / "resolved_aliases.json"
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
POLL_INTERVAL = float(os.getenv("VICREEL_JOBS_POLL_SEC", "1.0"))

# --- Initialisation ---
OUTPUT_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("vicreel-worker")

job_lock = Lock()

# NOUVEAU: Modèle de validation pour les jobs
class TTSJob(BaseModel):
    id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    text: str
    model: str = DEFAULT_MODEL
    language: str = "fr"
    speaker: Optional[str] = None
    speaker_wav: Optional[str] = None
    speaker_real: Optional[str] = None # Interne, pas dans le job initial
    format: str = "wav"
    options: Optional[Dict[str, Any]] = None

# --- Fonctions Utilitaires ---

def atomic_write_json(path: Path, data: Dict[str, Any]):
    """Écrit un JSON de manière atomique pour éviter les lectures partielles."""
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        logger.debug("Wrote JSON to %s", path)
    except Exception:
        logger.exception("Failed atomic write to %s", path)

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.exception("Failed to load JSON %s", path)
        return {}

# ... (gardez vos fonctions _normalize_text et fuzzy_match_key_to_available ici) ...
def _normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s); s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("-", " ").replace("_", " "); import re; s = re.sub(r"[^a-z0-9\s]+", "", s); s = re.sub(r"\s+", " ", s).strip()
    return s

def fuzzy_match_key_to_available(raw_key: str, available: List[str]) -> Optional[str]:
    if not raw_key or raw_key in available: return raw_key
    low = raw_key.strip().lower()
    for a in available:
        if a.lower() == low: return a
    norm_raw = _normalize_text(raw_key)
    norm_map = {_normalize_text(a): a for a in available}
    if norm_raw in norm_map: return norm_map[norm_raw]
    for a in available:
        if norm_raw in _normalize_text(a) or _normalize_text(a) in norm_raw: return a
    matches = difflib.get_close_matches(norm_raw, list(norm_map.keys()), n=1, cutoff=0.75)
    return norm_map[matches[0]] if matches else None


def synth_audio(tts_obj: TTS, job: TTSJob, wav_path: Path):
    """Wrapper sécurisé pour l'appel de synthèse TTS."""
    kwargs = {
        "text": job.text,
        "file_path": str(wav_path),
        "language": job.language,
    }
    if "xtts" in job.model:
        if job.speaker_wav:
            kwargs["speaker_wav"] = job.speaker_wav
        elif job.speaker_real:
            kwargs["speaker"] = job.speaker_real
        else:
            # Fallback au premier speaker disponible si aucun n'est spécifié
            available_speakers = getattr(tts_obj, "speakers", [])
            if available_speakers:
                kwargs["speaker"] = available_speakers[0]
                logger.warning("No speaker provided for XTTS, falling back to '%s'", kwargs["speaker"])
    else:
        if job.speaker_real:
            kwargs["speaker"] = job.speaker_real
    
    logger.info("Calling TTS.tts_to_file with: %s", kwargs)
    try:
        # 🐞 C'est ici que le crash silencieux se produit le plus souvent
        tts_obj.tts_to_file(**kwargs)
        logger.info("TTS synthesis completed successfully.")
    except Exception as e:
        # Cette exception ne sera probablement jamais attrapée en cas de segfault,
        # mais elle est essentielle pour les erreurs Python au sein de la lib.
        logger.exception("TTS synthesis failed with a Python exception.")
        raise e


def process_job_file(job_path: Path, tts_obj: TTS):
    """Traite un seul fichier de job de manière atomique et sécurisée."""
    job_id_from_filename = job_path.stem.replace(".inprogress", "")
    status_path = JOBS_DIR / f"{job_id_from_filename}.status.json"
    
    # Étape 1: Valider et charger le job
    try:
        job_data = json.loads(job_path.read_text(encoding="utf-8"))
        job = TTSJob(**job_data)
        # S'assurer que l'ID du job correspond au nom du fichier pour la cohérence
        job.id = job_id_from_filename
    except Exception as e:
        logger.error("Job file %s is invalid: %s", job_path, e)
        atomic_write_json(status_path, {
            "job_id": job_id_from_filename, "state": "error", "message": f"Invalid job JSON: {e}",
            "ts": time.time(), "trace": traceback.format_exc()
        })
        # On ne supprime pas le fichier invalide pour inspection
        return

    logger.info("Processing job ID: %s", job.id)
    atomic_write_json(status_path, {"job_id": job.id, "state": "started", "ts": time.time()})

    # Fichiers de sortie
    wav_out = OUTPUT_DIR / f"{job.id}.wav"
    final_out_path = OUTPUT_DIR / f"{job.id}.{job.format}"

    try:
        # Étape 2: Résoudre l'alias du speaker (logique métier)
        logger.info("Resolving speaker: '%s'", job.speaker)
        available_speakers = getattr(tts_obj, "speakers", [])
        if job.speaker:
             job.speaker_real = fuzzy_match_key_to_available(job.speaker, available_speakers)
        logger.info("Resolved speaker to: '%s'", job.speaker_real)
        
        # Étape 3: Synthèse audio (l'opération la plus risquée)
        logger.info("Starting audio synthesis for job %s...", job.id)
        synth_audio(tts_obj, job, wav_out)
        
        if not wav_out.exists() or wav_out.stat().st_size == 0:
            raise RuntimeError("Synthesis finished but output file is missing or empty. Likely a silent crash.")

        # Étape 4: Conversion de format (si nécessaire)
        if job.format != "wav":
            logger.info("Converting %s to %s...", wav_out, job.format)
            audio = AudioSegment.from_wav(wav_out)
            audio.export(final_out_path, format=job.format)
            wav_out.unlink() # Supprimer le WAV intermédiaire
            logger.info("Conversion successful.")
        
        # Étape 5: Succès ! Mettre à jour le statut et nettoyer.
        logger.info("✅ Job %s completed successfully. Output: %s", job.id, final_out_path)
        atomic_write_json(status_path, {
            "job_id": job.id, "state": "done", "ts": time.time(),
            "output": str(final_out_path), "speaker_real": job.speaker_real
        })
        job_path.unlink() # Nettoyer le fichier .inprogress

    except Exception as e:
        logger.exception("🔴 Job %s failed during processing.", job.id)
        atomic_write_json(status_path, {
            "job_id": job.id, "state": "error", "ts": time.time(),
            "message": str(e), "trace": traceback.format_exc()
        })
        # IMPORTANT: On ne supprime PAS le .inprogress en cas d'erreur pour débogage.


def main():
    logger.info("Starting worker — Jobs: %s, Outputs: %s, Model: %s", JOBS_DIR, OUTPUT_DIR, DEFAULT_MODEL)
    try:
        tts_obj = TTS(DEFAULT_MODEL)
        speakers = getattr(tts_obj, "speakers", []) or []
        logger.info("Model loaded; %d speakers available.", len(speakers))
    except Exception:
        logger.exception("FATAL: Could not load TTS model. Exiting.")
        return

    while True:
        try:
            # MODIFIÉ: Filtre les fichiers temporaires et les statuts
            candidates = [
                p for p in JOBS_DIR.glob("*.json")
                if not p.name.startswith('.') and not p.name.endswith(('.status.json', '.inprogress'))
            ]
            
            if not candidates:
                time.sleep(POLL_INTERVAL)
                continue

            # Trier par date de modification pour traiter les plus anciens d'abord
            job_file = sorted(candidates, key=lambda p: p.stat().st_mtime)[0]
            
            with job_lock:
                if not job_file.exists(): continue # Un autre worker l'a peut-être pris

                in_progress_path = job_file.with_suffix(".inprogress")
                logger.info("Claiming job %s -> %s", job_file.name, in_progress_path.name)
                job_file.rename(in_progress_path)
                
                process_job_file(in_progress_path, tts_obj)

        except KeyboardInterrupt:
            logger.info("Worker shutting down.")
            break
        except Exception:
            logger.exception("An unexpected error occurred in the main loop.")
            time.sleep(POLL_INTERVAL * 5) # Attendre plus longtemps en cas d'erreur de boucle

if __name__ == "__main__":
    main()
