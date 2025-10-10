# Dockerfile
# --- Étape 1 : Image de base légère avec Python 3.12 ---
FROM python:3.12-slim

# --- Configuration du répertoire de travail ---
WORKDIR /app

# --- Installation des dépendances système minimales ---
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# --- fichiers nécessaires ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# --- le reste du projet ---
COPY . .

# --- Variables d’environnement ---
ENV VICREEL_API_KEY=${VICREEL_API_KEY}
ENV VICREEL_DEFAULT_MODEL=tts_models/multilingual/multi-dataset/xtts_v2
ENV VICREEL_DEFAULT_LANGUAGE=fr
ENV VICREEL_MAX_CONCURRENCY=1
ENV PYTHONUNBUFFERED=1

# --- le port ---
EXPOSE 8080

# --- Commande de lancement ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
