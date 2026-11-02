# DeepShield: production image for Hugging Face Spaces (Docker SDK) or any container host.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

USER user
COPY --chown=user . .

# Bake the model weights into the image so cold starts don't download anything.
RUN python -m app.download_models

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
