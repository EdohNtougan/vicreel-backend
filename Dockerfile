FROM python:3.11-slim

# utils + ffmpeg + libs
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    gcc \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8000
ENV VICREEL_OUTPUT_DIR=/app/outputs
ENV VICREEL_MODELS_DIR=/app/models

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
