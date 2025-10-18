# app.py — VicReel (Version Finale, Complète, Performante et Sécurisée)
import os
import uuid
import json
import time
import re
import logging
from pathlib import Path
from typing import Optional, Dict, List, Iterable
from collections import deque
from threading import Lock # NOUVEAU : Import du Lock pour le singleton TTS

import nltk
from nltk.tokenize import sent_tokenize
from fastapi import FastAPI, Request, BackgroundTasks, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from pydub import AudioSegment
from google.cloud import storage

# --- Configuration ---
API_KEY = os.getenv("VICREEL_API_KEY", "vicreel_secret_20002025")
CONFIG_DIR = Path("config")
OUTPUT_DIR = Path(os.getenv("VICREEL_OUTPUT_DIR", "outputs"))
JOBS_DIR = Path(os.getenv("VICREEL_JOBS_DIR", "jobs"))
SPEAKER_MAP_FILE = CONFIG_DIR / "speaker_map.json"
DEFAULT_MODEL = os.getenv("VICREEL_DEFAULT_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")

MAX_TEXT_LENGTH = int(os.getenv("VICREEL_MAX_TEXT_LENGTH", "5000"))
PERSIST_OUTPUT_MAX_AGE = int(os.getenv("VICREEL_OUTPUT_MAX_AGE", str(60 * 10))) # 10 minutes
# NOUVEAU : Limite de taille pour les uploads de voix (10 Mo)
MAX_CLONE_FILE_SIZE = 10 * 1024 * 1024 

SUPPORTED_LANGUAGES = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"]

# --- NOUVEAU : Singleton pour le modèle TTS ---
# On initialise le modèle à None. Il sera chargé au premier besoin.
tts_model = None
tts_lock = Lock()

def get_tts_instance():
    """Charge le modèle TTS une seule fois et le réutilise de manière sécurisée."""
    global tts_model
    # Le 'lock' empêche plusieurs requêtes de charger le modèle en même temps
    with tts_lock:
        if tts_model is None:
            logger.info("Chargement du singleton du modèle TTS (la première fois uniquement)...")
            from TTS.api import TTS
            tts_model = TTS(DEFAULT_MODEL)
            logger.info("Modèle TTS chargé et prêt.")
    return tts_model

# --- Configuration Google Cloud Storage ---
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
storage_client = None
bucket = None
if GCS_BUCKET_NAME:
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        logger.info(f"Connecté au bucket Google Cloud Storage : {GCS_BUCKET_NAME}")
    except Exception as e:
        logger.exception("Échec de la connexion à GCS. Le clonage de voix sera désactivé.")
        bucket = None
else:
    logger.warning("GCS_BUCKET_NAME n'est pas configuré. Le clonage de voix sera désactivé.")

# --- Initialisation ---
OUTPUT_DIR.mkdir(exist_ok=True); JOBS_DIR.mkdir(exist_ok=True); CONFIG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("vicreel-hybrid")

app = FastAPI(title="VicReel - Hybrid TTS API")

# --- Configuration de sécurité CORS ---
ALLOWED_ORIGINS_STR = os.getenv("VICREEL_ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS_STR.split(",") if origin.strip()]
if not ALLOWED_ORIGINS:
    ALLOWED_ORIGINS = ["http://localhost", "http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:8080"]
    logger.warning(f"VICREEL_ALLOWED_ORIGINS non définie. Utilisation des valeurs par défaut pour le développement : {ALLOWED_ORIGINS}")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# --- MODIFIÉ : Modèles Pydantic avec validateur et chunk_size ---
class TTSRequest(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    speaker: Optional[str] = None
    speaker_wav_id: Optional[str] = None
    format: str = Field("mp3", pattern="^(wav|mp3)$")
    language: Optional[str] = "fr"
    split_long_text: bool = Field(True, description="Découper le texte en phrases pour les longues synthèses.")
    # NOUVEAU : Taille de segment dynamique
    chunk_size: int = Field(350, gt=100, le=500, description="Taille maximale des segments de texte (entre 100 et 500 caractères).")

    @validator('speaker_wav_id', always=True)
    def check_speaker_exclusive(cls, v, values):
        speaker_provided = 'speaker' in values and values['speaker'] is not None
        speaker_wav_provided = v is not None
        if speaker_provided and speaker_wav_provided:
            raise ValueError("Fournir soit 'speaker', soit 'speaker_wav_id', mais pas les deux.")
        if not speaker_provided and not speaker_wav_provided:
            raise ValueError("Un 'speaker' (prédéfini) ou un 'speaker_wav_id' (cloné) est requis.")
        return v

# --- Fonctions et Classes Conservées (Sécurité, Métriques, Rate Limit, etc.) ---
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if not API_KEY or api_key == API_KEY: return True
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clé API invalide ou manquante")

METRICS = {"requests_total": 0, "success_total": 0, "error_total": 0}
_rate_limit_store: Dict[str, deque] = {}; _rate_limit_lock = asyncio.Lock()
async def check_rate_limit(api_key_value: str):
    key = api_key_value or "ANON"; now = time.time(); RATE_LIMIT_WINDOW = 60; RATE_LIMIT_MAX = 30
    async with _rate_limit_lock:
        dq = _rate_limit_store.setdefault(key, deque())
        while dq and dq[0] <= now - RATE_LIMIT_WINDOW: dq.popleft()
        if len(dq) >= RATE_LIMIT_MAX: raise HTTPException(status_code=429, detail="Rate limit exceeded")
        dq.append(now)

SPEAKER_MAP: Dict[str, str] = json.loads(SPEAKER_MAP_FILE.read_text(encoding="utf-8")) if SPEAKER_MAP_FILE.exists() else {}
if not SPEAKER_MAP: logger.warning("Fichier speaker_map.json non trouvé ou vide.")

# --- Helper pour découper le texte ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("Téléchargement du tokenizer NLTK 'punkt'..."); nltk.download('punkt')

# --- MODIFIÉ : Le helper accepte maintenant une taille de segment dynamique ---
def split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    sentences = sent_tokenize(text, language='french' if 'fr' in text.lower() else 'english')
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars: current_chunk += " " + sentence
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks

# --- Tâche de Synthèse en Arrière-Plan (mise à jour) ---
def run_synthesis_task(job_id: str, job_data: dict):
    status_path = JOBS_DIR / f"{job_id}.status.json"
    def write_status(state: str, message: Optional[str] = None, output_path: Optional[str] = None):
        status_content = {"job_id": job_id, "state": state, "ts": time.time()};
        if message: status_content["message"] = message
        if output_path: status_content["output"] = str(output_path)
        status_path.write_text(json.dumps(status_content, indent=2), encoding="utf-8")

    speaker_ref_path_temp = None
    try:
        write_status("started", "En attente d'une ressource de calcul...")
        # --- MODIFIÉ : Utilisation du singleton TTS ---
        tts = get_tts_instance()
        
        speaker_arg, speaker_wav_arg = None, None; cloned_voice_id = job_data.get("speaker_wav_id")
        if cloned_voice_id:
            if not bucket: raise ConnectionError("Le service de clonage (GCS) n'est pas configuré.")
            blob = bucket.blob(cloned_voice_id)
            if not blob.exists(): raise FileNotFoundError(f"L'ID de voix clonée '{cloned_voice_id}' n'existe pas.")
            speaker_ref_path_temp = OUTPUT_DIR / f"temp_{job_id}_{cloned_voice_id}"
            blob.download_to_filename(speaker_ref_path_temp); speaker_wav_arg = str(speaker_ref_path_temp)
        else:
            predefined_speaker_alias = job_data["speaker"]; speaker_arg = SPEAKER_MAP.get(predefined_speaker_alias)
            if not speaker_arg: raise ValueError(f"L'alias de speaker '{predefined_speaker_alias}' est inconnu.")

        text_to_synth, chunk_files = job_data["text"], []
        final_output_path = None
        
        if job_data.get("split_long_text", True) and len(text_to_synth) > 400:
            # --- MODIFIÉ : Utilisation de la taille de segment dynamique ---
            chunks = split_text_into_chunks(text_to_synth, max_chars=job_data["chunk_size"])
            total_chunks = len(chunks)
            for i, chunk in enumerate(chunks):
                chunk_wav_path = OUTPUT_DIR / f"{job_id}_chunk_{i}.wav"; chunk_files.append(chunk_wav_path)
                status_message = f"Synthèse du segment {i+1}/{total_chunks}..."; write_status("running", status_message)
                logger.info("Job %s: %s", job_id, status_message)
                tts.tts_to_file(text=chunk, file_path=str(chunk_wav_path), speaker=speaker_arg, speaker_wav=speaker_wav_arg, language=job_data["language"])
            
            write_status("running", "Assemblage des segments audio...")
            combined_audio = AudioSegment.empty(); silence = AudioSegment.silent(duration=200); FADE_DURATION = 50
            for chunk_file in chunk_files:
                # --- MODIFIÉ : Ajout du fondu enchaîné (fade-in/out) ---
                segment = AudioSegment.from_wav(chunk_file)
                combined_audio += segment.fade_in(FADE_DURATION).fade_out(FADE_DURATION) + silence
            
            final_wav_path = OUTPUT_DIR / f"{job_id}.wav"; combined_audio.export(final_wav_path, format="wav")
            final_output_path = final_wav_path
            for chunk_file in chunk_files: chunk_file.unlink()
        else:
            final_wav_path = OUTPUT_DIR / f"{job_id}.wav"; write_status("running", "Synthèse en cours...")
            tts.tts_to_file(text=text_to_synth, file_path=str(final_wav_path), speaker=speaker_arg, speaker_wav=speaker_wav_arg, language=job_data["language"])
            final_output_path = final_wav_path
            
        if job_data["format"] == "mp3":
            mp3_path = final_output_path.with_suffix(".mp3")
            AudioSegment.from_wav(final_output_path).export(mp3_path, format="mp3")
            final_output_path.unlink(); final_output_path = mp3_path
        
        write_status("done", "Synthèse terminée.", output_path=final_output_path)
        logger.info("✅ Job %s terminé avec succès.", job_id)

    except Exception as e:
        logger.exception("🔴 Le job %s a échoué.", job_id); write_status("error", str(e))
    finally:
        if speaker_ref_path_temp and speaker_ref_path_temp.exists(): speaker_ref_path_temp.unlink()

# --- Nettoyage des anciens fichiers ---
def cleanup_old_files():
    try:
        now = time.time()
        for file in [p for p in OUTPUT_DIR.iterdir() if p.is_file()]:
            if now - file.stat().st_mtime > PERSIST_OUTPUT_MAX_AGE: file.unlink()
    except Exception: logger.exception("Le nettoyage des anciens fichiers a échoué.")

# --- Routes de l'API ---
@app.get("/"); def root(): return {"message": "API VicReel TTS - Hybride"}
@app.get("/health"); def health(): return {"status": "ok"}
@app.get("/metrics", dependencies=[Depends(verify_api_key)]); def get_metrics(): return METRICS
@app.get("/voices", dependencies=[Depends(verify_api_key)]); async def list_voices(): return [{"id": alias, "name": alias.replace("_", " ").title()} for alias in sorted(SPEAKER_MAP.keys())]
@app.get("/languages", dependencies=[Depends(verify_api_key)]); async def list_languages(): return {"languages": SUPPORTED_LANGUAGES}

# --- MODIFIÉ : Route de clonage avec validation de la taille ---
@app.post("/voices/clone", dependencies=[Depends(verify_api_key)])
async def clone_voice(user_id: str = Form(...), file: UploadFile = File(...)):
    if not bucket: raise HTTPException(status_code=503, detail="Le service de clonage est désactivé.")
    if not file.content_type or not file.content_type.startswith("audio/"): raise HTTPException(status_code=400, detail="Seuls les fichiers audio sont acceptés.")

    # --- NOUVEAU : Validation de la taille du fichier avant de continuer ---
    file_content = await file.read()
    if len(file_content) > MAX_CLONE_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"Le fichier est trop volumineux. La taille maximale est de {MAX_CLONE_FILE_SIZE / 1024 / 1024:.1f} Mo.")
    
    cloned_voice_id = f"user_{user_id}_{uuid.uuid4().hex[:12]}.wav"; blob = bucket.blob(cloned_voice_id)
    try:
        blob.upload_from_string(file_content, content_type=file.content_type)
        logger.info(f"Voix clonée pour l'utilisateur '{user_id}' sauvegardée sur GCS avec l'ID : {cloned_voice_id}")
    except Exception:
        logger.exception("Erreur lors de l'upload vers GCS."); raise HTTPException(status_code=500, detail="Impossible de sauvegarder le fichier de voix.")
    return {"speaker_wav_id": cloned_voice_id, "message": "Fichier reçu. Utilisez cet ID pour la synthèse."}

@app.delete("/voices/clone/{speaker_wav_id}", status_code=204, dependencies=[Depends(verify_api_key)])
async def delete_cloned_voice(speaker_wav_id: str, request: Request):
    user_id = request.headers.get("x-user-id")
    if not user_id: raise HTTPException(status_code=400, detail="L'en-tête 'x-user-id' est requis pour l'autorisation.")
    if not bucket: raise HTTPException(status_code=503, detail="Le service de clonage est désactivé.")
    if not speaker_wav_id.startswith(f"user_{user_id}_"):
        logger.warning(f"Tentative de suppression non autorisée par l'utilisateur '{user_id}' sur la voix '{speaker_wav_id}'.")
        raise HTTPException(status_code=403, detail="Vous n'êtes pas autorisé à supprimer cette ressource.")
    try:
        blob = bucket.blob(speaker_wav_id)
        if not blob.exists(): raise HTTPException(status_code=404, detail="La voix clonée spécifiée n'existe pas.")
        blob.delete()
        logger.info(f"Voix clonée '{speaker_wav_id}' supprimée avec succès par l'utilisateur '{user_id}'.")
    except HTTPException: raise
    except Exception as e:
        logger.exception(f"Erreur lors de la suppression de la voix '{speaker_wav_id}' sur GCS."); raise HTTPException(status_code=500, detail="Erreur interne lors de la suppression de la voix.")

@app.get("/jobs/{job_id}/status", dependencies=[Depends(verify_api_key)])
async def get_job_status(job_id: str, request: Request):
    status_file = JOBS_DIR / f"{job_id}.status.json"
    if not re.match(r"^[a-zA-Z0-9_.-]+$", job_id): raise HTTPException(status_code=400, detail="Job ID invalide.")
    if not status_file.exists(): return {"job_id": job_id, "state": "pending"}
    try:
        status_data = json.loads(status_file.read_text(encoding="utf-8"))
        if status_data.get("state") == "done":
            output_filename = Path(status_data["output"]).name
            status_data["output_url"] = str(request.url_for('outputs', path=output_filename))
        return status_data
    except Exception: raise HTTPException(status_code=500, detail="Impossible de lire le fichier de statut.")

_cleanup_counter = 0
@app.post("/tts", status_code=202, dependencies=[Depends(verify_api_key)])
async def submit_tts_job(request: Request, tts_request: TTSRequest, background_tasks: BackgroundTasks):
    global _cleanup_counter
    api_key = request.headers.get("x-api-key"); await check_rate_limit(api_key); METRICS["requests_total"] += 1
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    job_data = tts_request.dict()
    background_tasks.add_task(run_synthesis_task, job_id, job_data)
    _cleanup_counter += 1
    if _cleanup_counter >= 20:
        background_tasks.add_task(cleanup_old_files); _cleanup_counter = 0
    status_url = str(request.url_for('get_job_status', job_id=job_id))
    logger.info("Job %s soumis. Statut disponible à: %s", job_id, status_url)
    return {"job_id": job_id, "status_url": status_url, "message": "Job soumis. Vérifiez l'URL de statut pour suivre la progression."}
