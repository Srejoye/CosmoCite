# Deploying CosmoCite

CosmoCite uses a single process to serve the FastAPI backend and the built frontend on one port. The default engine requires no external LLM API key or GPU.

## Run Without Docker

```bash
pip install -r requirements.txt
python backend/features.py embed  # Once: builds data/processed/chunk_emb.npy
cd frontend && npm ci && npm run build && cd ..
uvicorn api:app --app-dir backend --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

The `/api/health` endpoint returns `{"ready": true}` once the application is ready.

## Run With Docker

```bash
docker compose up --build
```

Models and embeddings are prepared during the image build, allowing the container to start without downloading them at runtime.

To also run Ollama for the optional local model engine:

```bash
docker compose --profile local-model up --build
```

Then pull the required model inside the Ollama container. The image includes only the default engine; the FLAN-T5 option is unavailable in the Docker deployment.

The Docker image is tested through the `image` job in `.github/workflows/ci.yml`, which builds the image, starts the container and sends a real question. Check that this job passes before relying on the deployment.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `8000` | Port used by the application |
| `COSMOCITE_ORIGINS` | Local development addresses | Comma-separated list of origins allowed to access the API |
| `OLLAMA_HOST_URL` | `http://localhost:11434` | Ollama address for the optional local model engine |

The `config.yaml` file controls API concurrency and request queuing:

- `api.max_concurrent`: Maximum number of heavy requests running simultaneously.
- `api.queue_seconds`: Maximum waiting time for queued requests.
- `api.max_waiting`: Maximum number of requests allowed in the queue.

Requests exceeding these limits receive a `429` response.

On a CPU, keep `max_concurrent` at 1. In one comparison, increasing it to 2 reduced throughput from 0.40 to 0.17 answers per second.

## What to Expect

Measurements from the development laptop (Intel Core Ultra 7, CPU only), using the default engine:

| Measure | Result |
|---|---|
| Ready after startup | 10–25 seconds |
| New answer | 1–5 seconds |
| Cached repeat answer | Approximately 0.01 seconds |
| Server memory | Approximately 1.4 GB |

Run the load test against an active server:

```bash
python backend/loadtest.py
```

Results are written to `reports/load.json`.

Each request uses a new question. One laptop test produced the following results:

| Concurrent users | Served | Turned away | Typical wait |
|---|---:|---:|---:|
| 1 | 5 of 5 | 0 | 1 second |
| 8 | 24 of 24 | 0 | 12 seconds |
| 20 | 9 of 26 | 17 | 8 seconds |

No requests errored, and throughput was approximately 0.7 answers per second.

These results are hardware-dependent and should be treated as a best-case demonstration rather than a production capacity guarantee. CosmoCite is suitable for classroom demonstrations and small teams, rather than large-scale public deployment.

## Hosting

CosmoCite requires approximately 2 GB of RAM. Hosting providers with only 512 MB of available memory may not be suitable.

Check current hosting limits and pricing before choosing a provider.

| Hosting option | Notes |
|---|---|
| VPS or university server | Install Docker and run `docker compose up -d`. |
| Hugging Face Spaces | Check current Docker Space requirements, hardware availability and sleep policies. |
| Laptop with a secure tunnel | Useful for free live demonstrations, but the laptop must remain powered on. |

Configure HTTPS through your hosting provider's reverse proxy.

CosmoCite does not currently implement user accounts or authentication. Do not expose a public deployment without considering appropriate access controls and security measures.
