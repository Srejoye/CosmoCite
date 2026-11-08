# One image runs the whole app: FastAPI serves the API and the built frontend.
FROM node:22-slim AS web
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
ENV HF_HOME=/app/.cache PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY config.yaml ./
COPY backend/ backend/
COPY reports/ reports/
COPY data/processed/ data/processed/
COPY --from=web /app/frontend/dist frontend/dist

# Models and embeddings are baked in, so the container starts without network access.
RUN python backend/features.py embed \
 && python -c "import sys; sys.path.insert(0, 'backend'); from retrieval import _cross_encoder; _cross_encoder()"

RUN useradd --create-home app && chown -R app /app
USER app
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import json, os, urllib.request as u; exit(0 if json.load(u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health'))['ready'] else 1)"
CMD ["sh", "-c", "exec uvicorn api:app --app-dir backend --host 0.0.0.0 --port ${PORT}"]
