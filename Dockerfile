# --- Étape 1 : Image de base légère avec Python 3.12 ---
FROM python:3.12-slim

# --- Étape 2 : Configuration du répertoire de travail ---
WORKDIR /app

# --- Étape 3 : Installation des dépendances système minimales ---
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# --- Étape 4 : Copie des fichiers nécessaires ---
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# --- Étape 5 : Copier le reste du projet ---
COPY . .

# --- Étape 6 : Variables d’environnement ---
ENV VICREEL_API_KEY=${VICREEL_API_KEY}
ENV VICREEL_MAX_CONCURRENCY=1
ENV PYTHONUNBUFFERED=1

# --- Étape 7 : Exposer le port ---
EXPOSE 8000

# --- Étape 8 : Commande de lancement ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
