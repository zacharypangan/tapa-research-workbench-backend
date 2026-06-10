# Deployment: Vercel Frontend + Railway Backend

## Recommended Layout

- Frontend: deploy `stling_frontend/` to Vercel as a Vite app.
- Backend: deploy `stling_backend/` to Railway as a Docker service.
- Redis: use a Railway Redis service and connect it to the backend.
- Repository storage: attach a Railway volume at `/app/storage`.
- AI/Ollama: leave disabled in the first shared-team deployment unless you have a private reachable Ollama host.

## Railway Backend

Create a Railway service from this backend directory and let Railway build with the existing `Dockerfile`.

Required settings:

```env
REDIS_URL=<railway redis private url>
ALLOWED_ORIGINS=https://<vercel-domain>,https://<custom-domain-if-any>
REPOSITORY_STORAGE_ROOT=/app/storage/repository
REPOSITORY_OLLAMA_BASE_URL=
OLLAMA_BASE_URL=
```

Railway provides `PORT`; the Docker command already uses `${PORT:-8000}`.

Attach a persistent volume:

```text
Mount path: /app/storage
```

This preserves:

```text
/app/storage/repository/repository.sqlite
/app/storage/repository/files
/app/storage/repository/images
```

## Vercel Frontend

Create a Vercel project rooted at `stling_frontend/`.

Build settings:

```text
Framework preset: Vite
Build command: npm run build
Output directory: dist
```

Environment variable:

```env
VITE_API_BASE_URL=https://<railway-backend-domain>/api/v1
```

Enable Vercel deployment protection or team-only access for the shared-team deployment.

## Smoke Checks

After Railway deploys, run:

```bash
python scripts/smoke_deploy.py https://<railway-backend-domain>
```

Expected checks:

```text
GET /health
GET /api/v1/repository/statuses
GET /api/v1/repository/materials
```

After Vercel deploys:

1. Open the protected frontend URL.
2. Confirm the repository workbench loads.
3. Create a test material.
4. Upload a small file.
5. Restart/redeploy the Railway backend.
6. Confirm the material and file still exist.

## AI Behavior

Exact search, repository CRUD, uploads, extraction, observations, and downloadable reports do not require Ollama.

For the first deployment, keep these empty:

```env
REPOSITORY_OLLAMA_BASE_URL=
OLLAMA_BASE_URL=
```

The AI status endpoint should report Ollama unavailable instead of breaking the app. To enable hosted/private AI later, set `REPOSITORY_OLLAMA_BASE_URL` to the reachable Ollama base URL and configure the model env vars.

## Access Control

For v1, use platform protection:

- Vercel deployment protection or team-only access for the frontend.
- CORS restricted with `ALLOWED_ORIGINS`.
- Railway project access limited to trusted teammates.

Do not treat an obscure Railway URL as security.
