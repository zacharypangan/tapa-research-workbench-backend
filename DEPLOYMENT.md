# Deployment: Vercel Frontend + Railway Backend

## Recommended Layout

- Frontend: deploy `stling_frontend/` to Vercel as a Vite app.
- Backend: deploy `stling_backend/` as one Railway Docker service. The API and sequential ARQ worker run as supervised processes in that service and share its storage volume.
- Redis: use a Railway Redis service and connect it to the backend.
- Repository storage: attach a Railway volume at `/app/storage`.
- AI/Ollama: leave disabled in the first shared-team deployment unless you have a private reachable Ollama host.

## Railway Backend

Create a Railway service from this backend directory and let Railway build with the existing `Dockerfile`.

Required settings:

```env
REDIS_URL=<railway redis private url>
DATABASE_URL=<railway postgres private url>
ALLOWED_ORIGINS=https://<vercel-domain>,https://<custom-domain-if-any>
CLERK_ISSUER=https://<clerk-frontend-api-domain>
CLERK_AUTHORIZED_PARTIES=https://<vercel-domain>,https://<custom-domain-if-any>
AUTH_DISABLED=false
APP_ENV=production
CLERK_ORGANIZATION_ID=<authorized project organization id>
REPOSITORY_STORAGE_ROOT=/app/storage/repository
REPOSITORY_OLLAMA_BASE_URL=
OLLAMA_BASE_URL=
```

Railway provides `PORT`; the process supervisor passes it to Uvicorn, defaulting to 8000 for local use.

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

The Docker startup runs migrations once, then starts the API and worker through `python -m app.service`. It shuts down both if either exits. Do not create a second worker service: separate Railway services cannot share this SQLite/file volume.

Keep this deployment to one replica. Separate services or replicas will require shared database/file storage and a planned migration.

Use `/health` for process liveness and `/ready` to check PostgreSQL tables, repository storage access, and the worker's Redis heartbeat. Set the Railway health-check path to `/ready`.

Link extraction is restricted to administrator-approved public hosts and is disabled when no hosts are configured. To enable it, set `REPOSITORY_ALLOWED_LINK_HOSTS` to exact comma-separated hostnames, including approved redirect targets. Uploaded-file extraction works without that setting. Private addresses, arbitrary ports, oversized responses, and unapproved redirects are rejected.

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
VITE_CLERK_PUBLISHABLE_KEY=<production Clerk publishable key>
```

Enable Vercel deployment protection or team-only access for the shared-team deployment.

## Smoke Checks

After Railway deploys, run:

```bash
SMOKE_AUTH_TOKEN=<clerk-session-token> python scripts/smoke_deploy.py https://<railway-backend-domain>
```

Expected checks:

```text
GET /health
GET /ready
GET /api/v1/repository/statuses
GET /api/v1/repository/materials
GET /api/v1/progress/status
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

Use Clerk application authentication in addition to platform protection:

- Vercel deployment protection or team-only access for the frontend.
- Configure one authorized Clerk Organization through `CLERK_ORGANIZATION_ID`, using viewer, member, reviewer, and admin roles. Users select that organization after signing in. Unknown roles and other organizations are denied.
- FastAPI bearer-token verification for every mounted workbench router. Legacy chat, dataset, and tile routes are not exposed by this workbench app. Repository writes require member permission; review requires reviewer permission; material deletion requires admin permission.
- CORS restricted with `ALLOWED_ORIGINS`.
- Railway project access limited to trusted teammates.

Do not treat an obscure Railway URL as security.

## Database Operations

The backend container runs `alembic upgrade head` before starting Uvicorn. Keep Railway Postgres automated backups enabled and perform a restore drill before the first production import. Repository SQLite and files remain on the mounted Railway volume and require a separate volume snapshot or encrypted copy.

Before a schema deployment:

```bash
pg_dump "$DATABASE_URL" --format=custom --file=progress_predeploy.dump
```

Verify restore into a temporary database with `pg_restore`, then run the authenticated smoke check against the restored application environment.

## Evidence and job recovery

Extraction routes return HTTP 202 and a durable job identifier. The browser polls `/api/v1/progress/jobs/{job_id}`; API and worker use the existing Progress job table. Worker startup reconciles queued/running database jobs with Redis. A replayed extraction job reuses its committed extraction run instead of replacing segment identifiers. Image descriptions finish within the same extraction job; standalone image-index jobs use the same queue and database.

Forced re-extraction stages parsing and image files before committing replacements. Failed or empty replacements retain existing evidence. Materials cited by observations, Progress citations, or source alignments cannot be overwritten or deleted: upload a new material/version instead.

Test before publishing: denied anonymous/viewer/member actions, more than 100 materials, a scanned PDF while browsing, failed re-extraction, API/worker restart during extraction, and backup restoration across PostgreSQL plus the complete repository volume. Runtime production variables and backup schedules must be configured on the hosting platform; repository changes do not configure them automatically.
