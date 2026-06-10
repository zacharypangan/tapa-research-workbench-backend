# Tapa Research Workbench Backend

Standalone FastAPI backend for the Tapa Research Workbench modules developed from the SpatioTemporal Linguistics project.

The backend supports:

- material/reference repository API
- file uploads and link records
- PDF, slide, text, RTF, and link text extraction
- discovered-link logging
- full-text search and search context reports
- source-linked observations for terms, motifs, places, materials, and processes
- optional evidence-only AI retrieval using embeddings and cited corpus passages

This repo can be pushed and tracked independently while still integrating with the original backend through the files listed in `INTEGRATION.md`.

Developed with Python and managed using [uv](https://github.com/astral-sh/uv).

## Setup & Running

We highly recommend running the project using Docker Compose, as it seamlessly orchestrates the FastAPI Backend alongside the required Redis instance out of the box.

### Method 1: With Docker Compose (Recommended)

This method ensures you have an isolated environment fully mirroring production setups like Railway.

1. Ensure [Docker](https://docs.docker.com/get-docker/) is installed and running.
2. Create a `.env` file from `.env.example`.
3. Build and spin up the containers in the background:

   ```bash
   docker compose up -d --build
   ```

To view the server logs in real-time:
```bash
docker compose logs -f backend
```

To stop the services:
```bash
docker compose down
```

### Method 2: Without Docker (Local Development)

If you prefer to run the FastAPI app directly on your host machine, you will need to manage the Redis connection manually.

1. Install `uv` if you haven't already.
2. Ensure you have a Redis instance running locally (e.g., via Docker: `docker run -d -p 6379:6379 redis:alpine`).
3. Create a `.env` file from `.env.example`. Make sure your `REDIS_URL` points to your local instance (e.g. `redis://localhost:6379/0`).

To start the FastAPI server with auto-reload:
```bash
uv run python main.py
```

Or using uvicorn directly:
```bash
uv run uvicorn main:app --reload
```

## Adding Dependencies

To add a new package:

```bash
uv add <package-name>
```

## Repository Storage

Repository uploads, extracted text, observations, and local SQLite data are stored under:

```text
storage/repository
```

This folder is ignored by Git so the standalone repository stays lightweight.

For Railway deployment, attach a persistent volume at `/app/storage` and set:

```env
REPOSITORY_STORAGE_ROOT=/app/storage/repository
```

This keeps `repository.sqlite`, uploaded files, and extracted images across restarts and deploys.

## Optional AI Retrieval

Phase 4 AI features are optional. Exact search, extraction, and observations work without Ollama.

To enable semantic search and Ask Corpus, run Ollama and set:

```env
REPOSITORY_OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_RETRIEVAL_MODEL=llama3.1
OLLAMA_VISION_MODEL=llava
```

Recommended local models:

```bash
ollama pull nomic-embed-text
ollama pull llama3.1
ollama pull llava
```

For the first shared-team Railway deployment, leave `REPOSITORY_OLLAMA_BASE_URL` and `OLLAMA_BASE_URL` empty unless the backend can reach a private Ollama host. The repository AI status endpoint will report AI as unavailable while exact search, uploads, extraction, observations, and downloadable reports continue to work.

The AI layer is evidence-only:

- semantic search returns retrieved passages with source/page/segment citations
- multimodal search keeps document images separate and retrieves them through OCR plus local vision captions
- Ask Corpus answers only from retrieved passages
- responses must say when evidence is weak or absent
- observations remain human-reviewed source records, not AI interpretations

## Standalone Repository Setup

If this folder was originally cloned from another GitHub repo, keep the old remote as `upstream` and push your work to your own repo as `origin`:

```bash
git remote rename origin upstream
git remote add origin https://github.com/YOUR_USERNAME/tapa-research-workbench-backend.git
git push -u origin main
```

## Integration

See `INTEGRATION.md` for the files to port back into the original project.

## Deployment

Use Vercel for the Vite frontend and Railway for this Dockerized backend. See `DEPLOYMENT.md` for the exact settings.

After deploying the backend, run:

```bash
python scripts/smoke_deploy.py https://<railway-backend-domain>
```

The smoke check verifies:

```text
GET /health
GET /api/v1/repository/statuses
GET /api/v1/repository/materials
```

## Managing Python Version

The Python version is specified in `.python-version`. To use a different version:

```bash
uv python pin 3.12
```
