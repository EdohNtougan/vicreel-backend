# --- Dockerfile ---
FROM python:3.12-slim

# --- Configuration du répertoire de travail ---
WORKDIR /app

# --- Installation des dépendances système minimales ---
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# --- Copie des fichiers nécessaires ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# --- Copie du reste du projet ---
COPY . .

# --- Variables d’environnement (optimisations pour Cloud Run) ---
ENV HOSTNAME=0.0.0.0
ENV UVICORN_WORKERS=1
ENV LOG_LEVEL=info
ENV TRANSFORMS_CACHE=/app/.cache/huggingface/transformers
ENV VICREEL_API_KEY=${VICREEL_API_KEY}
ENV VICREEL_DEFAULT_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
ENV VICREEL_DEFAULT_LANGUAGE=fr
ENV VICREEL_MAX_CONCURRENCY=1
ENV PYTHONUNBUFFERED=1

# --- Expose le port (défaut pour Cloud Run) ---
EXPOSE 8080

# --- Commande de lancement (utilise $PORT pour Cloud Run) ---
CMD ["uvicorn", "app:app", "--host", "${HOSTNAME}", "--port", "$PORT", "--workers", "$UVICORN_WORKERS", "--log-level", "$LOG_LEVEL"]