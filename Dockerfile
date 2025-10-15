# =========================
# 1) STAGE DE BUILD
# =========================
FROM python:3.12-slim AS builder

WORKDIR /app

# Dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Copie requirements
COPY requirements.txt .

# Upgrade pip et installation torch + dépendances
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Téléchargement automatique du modèle XTTS-v2 pendant le build (optionnel)
ENV COQUI_TOS_AGREED=1
RUN python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)"

# Nettoyage du cache pip pour réduire la taille de l'image
RUN pip cache purge || true

# =========================
# 2) STAGE FINAL
# =========================
FROM python:3.12-slim

WORKDIR /app

# Métadonnées utiles pour déploiement (GCP, CI/CD, etc.)
LABEL maintainer="TonNom <tonemail@example.com>"
LABEL description="VicReel Coqui TTS API - Service de synthèse vocale"
LABEL version="1.0"

# Installer ffmpeg et dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les paquets Python depuis le builder
COPY --from=builder /usr/local /usr/local

# Copier le modèle XTTS téléchargé
COPY --from=builder /root/.local/share/tts /root/.local/share/tts

# =========================
# CONFIGURATION ALIASES
# =========================

# Crée le dossier config s’il n’existe pas
RUN mkdir -p /app/config

# Copier le fichier d'aliases s'il existe dans le repo
COPY config/speaker_aliases.json /app/config/speaker_aliases.json

# =========================
# CODE SOURCE
# =========================
COPY . .

# =========================
# VARIABLES D'ENVIRONNEMENT
# =========================

ENV COQUI_TOS_AGREED=1
ENV VICREEL_DEFAULT_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
ENV VICREEL_DEFAULT_LANGUAGE=fr
ENV VICREEL_MAX_CONCURRENCY=1
ENV PYTHONUNBUFFERED=1

# Timeout et purge auto (configurable via .env ou Cloud Run)
ENV VICREEL_TTS_TIMEOUT=60
ENV VICREEL_OUTPUT_MAX_FILES=200
ENV VICREEL_OUTPUT_PURGE_INTERVAL=600

# Fichier d'alias par défaut
ENV VICREEL_SPEAKER_ALIASES_FILE=/app/config/speaker_aliases.json

# Logs structurés
ENV VICREEL_LOG_FORMAT=json

# Nombre de workers uvicorn
ENV UVICORN_NUM_WORKERS=1

# =========================
# PORT ET LANCEMENT
# =========================
EXPOSE 8080

# Lancer uvicorn avec variables dynamiques
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${UVICORN_NUM_WORKERS}
