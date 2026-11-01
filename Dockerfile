# Web app image (CPU inference). Runs on Render (render.yaml), Hugging Face Spaces (Docker SDK, port 7860,
# non-root user with uid 1000) or any Docker host. The server listens on $PORT when the host sets it.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Free hosting tiers give a fraction of one CPU; one math thread avoids oversubscription.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only PyTorch (a fraction of the size of the CUDA build), then the remaining run-time packages.
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu torch
COPY --chown=user requirements-deploy.txt .
RUN pip install --user -r requirements-deploy.txt

COPY --chown=user app.py LICENSE ./
COPY --chown=user src ./src
COPY --chown=user frontend ./frontend
COPY --chown=user models ./models

EXPOSE 7860
# One worker process keeps a single copy of both models in memory; threads serve concurrent requests
# (the predictor is thread-safe). The long timeout covers model loading on very small CPU shares.
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 300 'app:create_app()'"]
