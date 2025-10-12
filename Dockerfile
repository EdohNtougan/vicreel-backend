# Dockerfile - pour vikedoh / vicreel-api

FROM python:3.12-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Créer la structure attendue pour Coqui / TTS et copier le modèle depuis le contexte
RUN mkdir -p /root/.local/share/tts/tts_models/multilingual/multi-dataset
COPY models/xtts/tts_models--multilingual--multi-dataset--xtts_v2 \
    /root/.local/share/tts/tts_models/multilingual/multi-dataset/xtts_v2

# Copier le code de l'application
COPY . .

# Variables d'environnement
ENV COQUI_TOS_AGREED=1
ENV VICREEL_DEFAULT_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
ENV VICREEL_DEFAULT_LANGUAGE=fr
ENV VICREEL_MAX_CONCURRENCY=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
