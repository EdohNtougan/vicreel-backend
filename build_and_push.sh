#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="vikedoh/vicreel-api:latest"

echo "1) Vérifier modèle dans ./models/xtts..."
if [ ! -d "./models/xtts/tts_models--multilingual--multi-dataset--xtts_v2" ]; then
  echo "ERREUR: modèle introuvable dans ./models/xtts. Copier depuis /home/codespace/... puis retry."
  exit 1
fi

echo "2) Nettoyage Docker pour libérer de l'espace (silencieux)..."
docker system prune -a -f || true

echo "3) Build de l'image Docker..."
docker build -t ${IMAGE_NAME} .

echo "4) Vérifier image..."
docker images | grep "$(echo ${IMAGE_NAME} | cut -d: -f1)" || true

echo "5) Push sur Docker Hub (assurez-vous d'être connecté avec 'docker login')..."
docker push ${IMAGE_NAME}

echo "6) Nettoyage post-push..."
docker system prune -a -f || true

echo "Terminé. Image créée et poussée : ${IMAGE_NAME}"
