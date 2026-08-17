# All-in-one image: every Glimms service plus a single-port gateway.
#
# `docker build .` at the repository root produces this image.  It runs all
# eight services inside one container and exposes them through the gateway on
# port 8080 ($PORT): route /<service-name>/... to reach each service.
#
# The image is intentionally lightweight — torch, transformers, ultralytics,
# opencv, onnxruntime, rembg and pinecone-client are not installed, so the
# services run their deterministic offline fallbacks (see README).  For the
# full production images, build the per-service Dockerfiles with
# `docker compose up --build`.
FROM python:3.11-slim

WORKDIR /app

RUN addgroup --system app && adduser --system --group app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY services/ ./services/
COPY gateway/ ./gateway/

USER app
ENV PYTHONUNBUFFERED=1

# Single public port; the gateway proxies each /<service-name>/ prefix.
EXPOSE 8080

CMD ["python", "-m", "gateway.serve_all"]
