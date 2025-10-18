# app.py — VicReel (Version Hybride Finale avec gestion des textes longs)
import os
import uuid
import json
import time
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Iterable

# --- NOUVEAU : Import de NLTK ---
import nltk
from nltk.tokenize import sent_tokenize

from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydub import AudioSegment

# --- Configuration (inchangée) ---
API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025")
CONFIG_DIR = Path("config")
OUTPUT_DIR = Path(os.getenv("VICREEL_OUTPUT_DIR", "outputs"))
JOBS_DIR = Path(os.getenv("VICREEL_JOBS_DIR", "jobs"))
SPEAKER_MAP_FILE = CONFIG_DIR / "speaker_map.json"
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

# --- MODIFIÉ : Limite de texte augmentée ---
MAX_TEXT_LENGTH = int(os.getenv("VICREEL_MAX_TEXT_LENGTH", "5000"))
PERSIST_OUTPUT_MAX_AGE = int(os.getenv("VICREEL_OUTPUT_MAX_AGE", str(60 * 10)))

# ... (Initialisation, Logging, FastAPI app - inchangés) ...
OUTPUT_DIR.mkdir(exist_ok=True); JOBS_DIR.mkdir(exist_ok=True); CONFIG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("vicreel-hybrid")
app = FastAPI(title="VicReel - Hybrid TTS API"); app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]); app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- Modèles Pydantic (modifiés) ---
class TTSRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    speaker: str
    format: str = Field("mp3", pattern="^(wav|mp3)$")
    language: Optional[str] = "fr"
    # --- NOUVEAU : Option pour activer/désactiver le découpage ---
    split_long_text: bool = Field(True, description="Découper le texte en phrases pour les longues synthèses.")

# ... (Sécurité, Métriques, Rate Limiting, Speaker Map - inchangés) ...
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False); def verify_api_key(api_key: Optional[str] = Depends(api_key_header)): ...
METRICS = {"requests_total": 0, "success_total": 0, "error_total": 0}; _rate_limit_store: Dict[str, deque] = {}; _rate_limit_lock = asyncio.Lock(); async def check_rate_limit(api_key_value: str): ...
SPEAKER_MAP: Dict[str, str] = json.loads(SPEAKER_MAP_FILE.read_text(encoding="utf-8")) if SPEAKER_MAP_FILE.exists() else {}

# --- NOUVEAU : Helper pour découper le texte ---
# S'assure que NLTK a les données nécessaires au premier appel
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Téléchargement du tokenizer NLTK 'punkt'...")
    nltk.download('punkt')

def split_text_into_chunks(text: str, max_chars: int = 250) -> List[str]:
    """Découpe le texte en segments basés sur les phrases, sans dépasser max_chars."""
    sentences = sent_tokenize(text, language='french' if 'fr' in text.lower() else 'english')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# --- Tâche de Synthèse en Arrière-Plan (mise à jour) ---
def run_synthesis_task(job_id: str, job_data: dict):
    status_path = JOBS_DIR / f"{job_id}.status.json"
    
    def write_status(state: str, message: Optional[str] = None, output_path: Optional[str] = None): ...

    try:
        write_status("started", "Chargement du modèle TTS...")
        from TTS.api import TTS
        
        tts = TTS(DEFAULT_MODEL)
        real_speaker_name = SPEAKER_MAP.get(job_data["speaker"])
        if not real_speaker_name:
            raise ValueError(f"Alias de speaker '{job_data['speaker']}' non trouvé.")

        text_to_synth = job_data["text"]
        chunk_files = []
        final_output_path = None
        
        # --- LOGIQUE DE DÉCOUPAGE ---
        # On découpe si l'option est activée ET si le texte est assez long
        # (seuil de 400 pour éviter de découper des textes moyens inutilement)
        if job_data.get("split_long_text", True) and len(text_to_synth) > 400:
            logger.info("Texte long détecté (%d caractères). Découpage en segments...", len(text_to_synth))
            chunks = split_text_into_chunks(text_to_synth)
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                chunk_wav_path = OUTPUT_DIR / f"{job_id}_chunk_{i}.wav"
                chunk_files.append(chunk_wav_path)
                
                status_message = f"Synthèse du segment {i+1}/{total_chunks}..."
                write_status("running", status_message)
                logger.info("Job %s: %s", job_id, status_message)
                
                tts.tts_to_file(
                    text=chunk, file_path=str(chunk_wav_path),
                    speaker=real_speaker_name, language=job_data["language"]
                )
            
            # Concaténation des segments
            write_status("running", "Assemblage des segments audio...")
            combined_audio = AudioSegment.empty()
            silence = AudioSegment.silent(duration=200) # 0.2s de silence entre les phrases
            for chunk_file in chunk_files:
                segment = AudioSegment.from_wav(chunk_file)
                combined_audio += segment + silence
            
            final_wav_path = OUTPUT_DIR / f"{job_id}.wav"
            combined_audio.export(final_wav_path, format="wav")
            final_output_path = final_wav_path

            # Nettoyage des fichiers de segment
            for chunk_file in chunk_files:
                chunk_file.unlink()

        else: # Logique pour les textes courts (inchangée)
            final_wav_path = OUTPUT_DIR / f"{job_id}.wav"
            write_status("running", f"Synthèse en cours avec la voix '{real_speaker_name}'...")
            tts.tts_to_file(
                text=text_to_synth, file_path=str(final_wav_path),
                speaker=real_speaker_name, language=job_data["language"]
            )
            final_output_path = final_wav_path
            
        # Conversion finale en MP3 si demandé
        if job_data["format"] == "mp3":
            mp3_path = final_output_path.with_suffix(".mp3")
            audio = AudioSegment.from_wav(final_output_path)
            audio.export(mp3_path, format="mp3")
            final_output_path.unlink() # Supprimer le WAV final
            final_output_path = mp3_path
        
        write_status("done", "Synthèse terminée.", output_path=final_output_path)
        logger.info("✅ Job %s terminé avec succès.", job_id)

    except Exception as e:
        logger.exception("🔴 Le job %s a échoué.", job_id)
        write_status("error", str(e))

# ... (Nettoyage, Routes /health, /metrics, /voices, /jobs/{job_id}/status - inchangés) ...
def cleanup_old_files(): ...
@app.get("/")
def root(): ...
@app.get("/health")
def health(): ...
@app.get("/metrics", dependencies=[Depends(verify_api_key)])
def get_metrics(): ...
@app.get("/voices", dependencies=[Depends(verify_api_key)])
async def list_voices(): ...
@app.get("/jobs/{job_id}/status", dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str, request: Request): ...


# Route /tts (inchangée, le modèle Pydantic gère la nouvelle option)
_cleanup_counter = 0
@app.post("/tts", status_code=202, dependencies=[Depends(verify_api_key)])
async def submit_tts_job(request: Request, tts_request: TTSRequest, background_tasks: BackgroundTasks): ...

