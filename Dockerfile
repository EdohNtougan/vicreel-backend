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

# =========================
# 2) STAGE FINAL
# =========================
FROM python:3.12-slim

WORKDIR /app

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

# Crée le dossier config s’il n’existe pas (sinon COPY échoue)
RUN mkdir -p /app/config

# Copier le fichier d'aliases s'il existe dans le repo
# (Docker ignore l’erreur si le fichier est absent grâce à l’option --chown ou un .dockerignore propre)
COPY config/speaker_aliases.json /app/config/speaker_aliases.json

# =========================
# CODE SOURCE
# =========================
COPY . .

# Variables d'environnement par défaut
ENV COQUI_TOS_AGREED=1
ENV VICREEL_DEFAULT_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
ENV VICREEL_DEFAULT_LANGUAGE=fr
ENV VICREEL_MAX_CONCURRENCY=1
ENV PYTHONUNBUFFERED=1

# Par défaut, on pointe VICREEL_SPEAKER_ALIASES_FILE vers le fichier dans l'image
ENV VICREEL_SPEAKER_ALIASES_FILE=/app/config/speaker_aliases.json

# Exposer le port
EXPOSE 8080

# Lancer uvicorn
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
