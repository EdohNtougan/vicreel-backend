# =========================
# 1) STAGE DE BUILD
# =========================
FROM python:3.12-slim AS builder

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de configuration et requirements
COPY requirements.txt .
COPY config/speaker_aliases.json ./config/
COPY sync_aliases.py .

# Installation des dépendances Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- ÉTAPE CRUCIALE : Génération de la carte des voix ---
RUN python3 sync_aliases.py

# Téléchargement du modèle TTS
ENV COQUI_TOS_AGREED=1
RUN python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# =========================
# 2) STAGE FINAL
# =========================
FROM python:3.12-slim

WORKDIR /app

# Installer les dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les paquets Python depuis le builder
COPY --from=builder /usr/local /usr/local

# Copier le modèle TTS téléchargé
COPY --from=builder /root/.local/share/tts /root/.local/share/tts

# Copier le code source de l'application
COPY app.py .
COPY config/ ./config/

# --- IMPORTANT : Copier la carte des voix générée ---
COPY --from=builder /app/config/speaker_map.json ./config/speaker_map.json

# Variables d'environnement pour l'exécution
ENV COQUI_TOS_AGREED=1
ENV PYTHONUNBUFFERED=1
# Les autres variables (API_KEY, etc.) seront injectées par Cloud Run ou docker-compose

# Port
EXPOSE 8080

# Commande de lancement
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
