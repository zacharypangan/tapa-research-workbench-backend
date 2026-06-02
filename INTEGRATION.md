# Integration Notes

This backend can run as a standalone FastAPI service or be integrated with the original SpatioTemporal Linguistics backend.

## Standalone Mode

Recommended local run:

```bash
cp .env.example .env
docker compose up -d --build
```

API root:

```text
http://127.0.0.1:8000/api/v1
```

Repository API routes are mounted at:

```text
http://127.0.0.1:8000/api/v1/repository
```

Repository files and SQLite data are stored under:

```text
storage/repository
```

This is intentionally ignored by Git.

## Original Project Integration

The repository workbench backend lives primarily in:

```text
app/api/repository.py
```

It is mounted in:

```text
main.py
```

To integrate changes back into the original backend, compare and port:

- `app/api/repository.py`
- `main.py`
- `Dockerfile`
- `docker-compose.yml`
- `pyproject.toml`
- `uv.lock`
- `.gitignore`

The backend startup hook calls `repository.init_repository_db()`, which creates or updates local repository tables.

## Phase Tracking

Recommended branch names:

- `phase-1-repository-api`
- `phase-2-extraction-search`
- `phase-3-observations`
- `deployment-runtime`

