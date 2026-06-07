# Allergo probe analyzer API
FROM python:3.12-slim

# Pillow runtime needs libjpeg/zlib (wheels bundle most, but be safe for image I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code (samples are handy for a quick smoke test inside the container)
COPY allergo_core.py api.py ./
COPY samples/ ./samples/
# finetune package is imported by allergo_core at runtime (features extractor)
COPY finetune/ ./finetune/
# trained model + threshold fallback (classifier.joblib, thresholds.json)
COPY models/ ./models/

# run as non-root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# basic liveness probe against the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
