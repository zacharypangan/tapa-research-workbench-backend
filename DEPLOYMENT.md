# Deployment: Vercel Frontend + Railway Backend

## Shared Workbench release branch

The shared frontend URL is `https://tapa-research-workbench-frontend.vercel.app`.
Use `codex/mvp-document-graphs` as the Vercel production branch and as the branch
connected to the existing Railway `production` backend service. Keep its
existing `/app/storage` volume and production API hostname:
`https://tapa-research-workbench-backend-production.up.railway.app/api/v1`.
Do not point the shared frontend at the empty staging repository. Check the
existing material IDs before and after rollout, `/health`, and
`/api/v1/research-runs/capabilities`. The graph workflow adds files beneath
the existing repository storage root without a database migration.

## Document graph staging

Use `codex/mvp-document-graphs` in both repositories. It starts from the hosted
`mvp-ollama-cloud` branch and adds only the document graph workflow. Deploy the
backend branch to the Railway `graph-staging` environment in the existing
Workbench project, with its own persistent volume mounted at `/app/storage`.
The API starts and recovers graph jobs itself; no additional worker or database
is needed. `uv.lock` includes the approved static web extraction dependency.

The corresponding Vercel **preview branch override** is:

```text
Branch: codex/mvp-document-graphs
VITE_API_BASE_URL=https://tapa-research-workbench-backend-graph-staging.up.railway.app/api/v1
```

Pushes to this branch update graph staging. The existing production branch and
production storage are independent. Verify `/api/v1/research-runs/capabilities`
and a document-to-graph run before sharing the preview. See
`docs/knowledge-graphs.md` for the method, API and focused checks.

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
