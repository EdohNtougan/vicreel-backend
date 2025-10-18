# =========================
# 1) STAGE DE BUILD : Prépare tout ce qui est lourd
# =========================
FROM python:3.12-slim AS builder

WORKDIR /app

# Dépendances système nécessaires pour l'installation et le build
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers nécessaires pour le build des dépendances et de la config
COPY requirements.txt .
COPY config/speaker_aliases.json ./config/
COPY sync_aliases.py .

# Installation des dépendances Python
# Cette étape peut être longue, c'est pourquoi elle est dans le stage de build
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# --- ÉTAPES D'INITIALISATION AUTOMATISÉES ---

# 1. Téléchargement des données pour NLTK (découpage de phrases)
RUN python3 -m nltk.downloader punkt

# 2. Génération de la carte des voix `speaker_map.json`
RUN python3 sync_aliases.py

# 3. Téléchargement du modèle TTS pour qu'il soit inclus dans l'image
ENV COQUI_TOS_AGREED=1
RUN python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"


# =========================
# 2) STAGE FINAL : L'image légère qui sera déployée
# =========================
FROM python:3.12-slim

WORKDIR /app

# Installer uniquement les dépendances système nécessaires à l'exécution
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copier l'environnement Python pré-installé depuis le stage de build
COPY --from=builder /usr/local /usr/local

# Copier le modèle TTS pré-téléchargé
COPY --from=builder /root/.local/share/tts /root/.local/share/tts

# Copier le code source de l'application
COPY app.py .
COPY config/ ./config/

# --- IMPORTANT : Copier la carte des voix qui a été générée ---
COPY --from=builder /app/config/speaker_map.json ./config/speaker_map.json

# Variables d'environnement pour l'exécution
ENV COQUI_TOS_AGREED=1
ENV PYTHONUNBUFFERED=1
# Les autres variables (API_KEY, etc.) seront injectées par votre plateforme (Cloud Run, .env, etc.)

# Port sur lequel l'application écoute
EXPOSE 8080

# Commande de lancement de l'API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
