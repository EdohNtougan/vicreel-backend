# app.py — VicReel (Version Hybride Finale avec gestion des textes longs et route /languages)
import os
import uuid
import json
import time
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Iterable

# --- Import de NLTK ---
import nltk
from nltk.tokenize import sent_tokenize

from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment

# --- Configuration ---
API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025")
CONFIG_DIR = Path("config")
OUTPUT_DIR = Path(os.getenv("VICREEL_OUTPUT_DIR", "outputs"))
JOBS_DIR = Path(os.getenv("VICREEL_JOBS_DIR", "jobs"))
SPEAKER_MAP_FILE = CONFIG_DIR / "speaker_map.json"
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

MAX_TEXT_LENGTH = int(os.getenv("VICREEL_MAX_TEXT_LENGTH", "5000"))
PERSIST_OUTPUT_MAX_AGE = int(os.getenv("VICREEL_OUTPUT_MAX_AGE", str(60 * 10))) # 10 minutes

# --- NOUVEAU : Liste des langues supportées ---
SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]

# --- Initialisation ---
OUTPUT_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("vicreel-hybrid")

app = FastAPI(title="VicReel - Hybrid TTS API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- Modèles Pydantic ---
class TTSRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    speaker: str
    format: str = Field("mp3", pattern="^(wav|mp3)$")
    language: Optional[str] = "fr"
    split_long_text: bool = Field(True, description="Découper le texte en phrases pour les longues synthèses.")

# --- Fonctions et Classes Conservées (Sécurité, Métriques, Rate Limit, etc.) ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if not API_KEY or api_key == API_KEY: return True
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé API invalide ou manquante")

METRICS = {"requests_total": 0, "success_total": 0, "error_total": 0}
_rate_limit_store: Dict[str, deque] = {}
_rate_limit_lock = asyncio.Lock()
async def check_rate_limit(api_key_value: str):
    key = api_key_value or "ANON"
    now = time.time()
    RATE_LIMIT_WINDOW = 60
    RATE_LIMIT_MAX = 30
    async with _rate_limit_lock:
        dq = _rate_limit_store.setdefault(key, deque())
        while dq and dq[0] <= now - RATE_LIMIT_WINDOW: dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        dq.append(now)

SPEAKER_MAP: Dict[str, str] = json.loads(SPEAKER_MAP_FILE.read_text(encoding="utf-8")) if SPEAKER_MAP_FILE.exists() else {}
if not SPEAKER_MAP: logger.warning("Fichier speaker_map.json non trouvé ou vide.")

# --- Helper pour découper le texte ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Téléchargement du tokenizer NLTK 'punkt'...")
    nltk.download('punkt')

def split_text_into_chunks(text: str, max_chars: int = 250) -> List[str]:
    sentences = sent_tokenize(text, language='french' if 'fr' in text.lower() else 'english')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

# --- Tâche de Synthèse en Arrière-Plan ---
def run_synthesis_task(job_id: str, job_data: dict):
    status_path = JOBS_DIR / f"{job_id}.status.json"
    
    def write_status(state: str, message: Optional[str] = None, output_path: Optional[str] = None):
        status_content = {"job_id": job_id, "state": state, "ts": time.time()}
        if message: status_content["message"] = message
        if output_path: status_content["output"] = str(output_path)
        status_path.write_text(json.dumps(status_content, indent=2), encoding="utf-8")

    try:
        write_status("started", "Chargement du modèle TTS...")
        from TTS.api import TTS
        
        tts = TTS(DEFAULT_MODEL)
        real_speaker_name = SPEAKER_MAP.get(job_data["speaker"])
        if not real_speaker_name: raise ValueError(f"Alias '{job_data['speaker']}' non trouvé.")

        text_to_synth, chunk_files = job_data["text"], []
        final_output_path = None
        
        if job_data.get("split_long_text", True) and len(text_to_synth) > 400:
            chunks = split_text_into_chunks(text_to_synth)
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                chunk_wav_path = OUTPUT_DIR / f"{job_id}_chunk_{i}.wav"
                chunk_files.append(chunk_wav_path)
                status_message = f"Synthèse du segment {i+1}/{total_chunks}..."
                write_status("running", status_message)
                logger.info("Job %s: %s", job_id, status_message)
                tts.tts_to_file(text=chunk, file_path=str(chunk_wav_path), speaker=real_speaker_name, language=job_data["language"])
            
            write_status("running", "Assemblage des segments audio...")
            combined_audio = AudioSegment.empty()
            silence = AudioSegment.silent(duration=200)
            for chunk_file in chunk_files:
                combined_audio += AudioSegment.from_wav(chunk_file) + silence
            final_wav_path = OUTPUT_DIR / f"{job_id}.wav"
            combined_audio.export(final_wav_path, format="wav")
            final_output_path = final_wav_path
            for chunk_file in chunk_files: chunk_file.unlink()
        else:
            final_wav_path = OUTPUT_DIR / f"{job_id}.wav"
            write_status("running", f"Synthèse en cours avec la voix '{real_speaker_name}'...")
            tts.tts_to_file(text=text_to_synth, file_path=str(final_wav_path), speaker=real_speaker_name, language=job_data["language"])
            final_output_path = final_wav_path
            
        if job_data["format"] == "mp3":
            mp3_path = final_output_path.with_suffix(".mp3")
            AudioSegment.from_wav(final_output_path).export(mp3_path, format="mp3")
            final_output_path.unlink()
            final_output_path = mp3_path
        
        write_status("done", "Synthèse terminée.", output_path=final_output_path)
        logger.info("✅ Job %s terminé avec succès.", job_id)

    except Exception as e:
        logger.exception("🔴 Le job %s a échoué.", job_id)
        write_status("error", str(e))

# --- Nettoyage des anciens fichiers ---
def cleanup_old_files():
    try:
        now = time.time()
        for file in [p for p in OUTPUT_DIR.iterdir() if p.is_file()]:
            if now - file.stat().st_mtime > PERSIST_OUTPUT_MAX_AGE: file.unlink()
    except Exception: logger.exception("Le nettoyage des anciens fichiers a échoué.")

# --- Routes de l'API ---
@app.get("/")
def root(): return {"message": "API VicReel TTS - Hybride"}

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/metrics", dependencies=[Depends(verify_api_key)])
def get_metrics(): return METRICS

@app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices():
    return [{"id": alias, "name": alias.replace("_", " ").title()} for alias in sorted(SPEAKER_MAP.keys())]

# --- NOUVEAU : Route pour lister les langues ---
@app.get("/languages", dependencies=[Depends(verify_api_key)])
async def list_languages():
    """Retourne la liste des codes de langue supportés par le modèle XTTSv2."""
    return {"languages": SUPPORTED_LANGUAGES}

@app.get("/jobs/{job_id}/status", dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str, request: Request):
    status_file = JOBS_DIR / f"{job_id}.status.json"
    if not re.match(r"^[a-zA-Z0-9_-]+$", job_id):
        raise HTTPException(status_code=400, detail="Job ID invalide.")
    if not status_file.exists():
        return {"job_id": job_id, "state": "pending"}
    try:
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        if status_data.get("state") == "done":
            output_filename = Path(status_data["output"]).name
            status_data["output_url"] = str(request.url_for('outputs', path=output_filename))
        return status_data
    except Exception:
        raise HTTPException(status_code=500, detail="Impossible de lire le fichier de statut.")

_cleanup_counter = 0
@app.post("/tts", status_code=202, dependencies=[Depends(verify_api_key)])
async def submit_tts_job(request: Request, tts_request: TTSRequest, background_tasks: BackgroundTasks):
    global _cleanup_counter
    api_key = request.headers.get("x-api-key")
    await check_rate_limit(api_key)
    METRICS["requests_total"] += 1

    if tts_request.speaker not in SPEAKER_MAP:
        raise HTTPException(status_code=400, detail=f"Speaker '{tts_request.speaker}' inconnu.")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_data = tts_request.dict()

    background_tasks.add_task(run_synthesis_task, job_id, job_data)

    _cleanup_counter += 1
    if _cleanup_counter >= 20:
        background_tasks.add_task(cleanup_old_files)
        _cleanup_counter = 0

    status_url = str(request.url_for('get_job_status', job_id=job_id))
    logger.info("Job %s soumis. Statut disponible à: %s", job_id, status_url)
    
    return {
        "job_id": job_id,
        "status_url": status_url,
        "message": "Job soumis. Vérifiez l'URL de statut pour suivre la progression."
    }
