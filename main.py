import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from sqlalchemy import inspect, text
from starlette.concurrency import run_in_threadpool
from app.api import governance, progress, repository
from app.auth import _auth_disabled
from app.progress.database import SessionLocal
from app.progress.models import ProgressJob

@asynccontextmanager
async def lifespan(application: FastAPI):
    await run_in_threadpool(on_startup)
    yield


app = FastAPI(title="Tapa Research Workbench Backend", lifespan=lifespan)

# Base CORS origins
origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://192.168.1.22:5173",
    "https://spatiotemporallinguistics.vercel.app",
]

env_origins = os.getenv("ALLOWED_ORIGINS", "")
if env_origins:
    origins = [o.strip() for o in env_origins.split(",") if o.strip()]

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(repository.router, prefix="/api/v1")
app.include_router(governance.router, prefix="/api/v1")
app.include_router(progress.router, prefix="/api/v1")


def on_startup():
    if not _auth_disabled():
        required = ("CLERK_ISSUER", "CLERK_ORGANIZATION_ID", "CLERK_AUTHORIZED_PARTIES", "ALLOWED_ORIGINS")
        if any(not os.getenv(name, "").strip() for name in required):
            raise RuntimeError("Configure Clerk issuer, project organization, and authorized frontend origins before starting the API.")
    repository.init_repository_db()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    checks = {"database": False, "repository": False, "worker": False}

    def check_storage():
        try:
            with SessionLocal() as session:
                session.execute(text("SELECT 1"))
                checks["database"] = inspect(session.bind).has_table(ProgressJob.__tablename__)
        except Exception:
            pass
        try:
            with repository.get_connection() as con:
                con.execute("SELECT 1 FROM materials LIMIT 1")
            checks["repository"] = all(os.access(path, os.W_OK) for path in (repository.STORAGE_ROOT, repository.FILES_ROOT, repository.IMAGES_ROOT))
        except Exception:
            pass

    await run_in_threadpool(check_storage)
    client = Redis.from_url(os.getenv("REDIS_PRIVATE_URL") or os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_connect_timeout=2, socket_timeout=2)
    try:
        checks["worker"] = bool(await client.get("arq:queue:health-check"))
    except Exception:
        pass
    finally:
        await client.aclose()
    ready = all(checks.values())
    return JSONResponse({"status": "ready" if ready else "unavailable", "checks": checks}, status_code=200 if ready else 503)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
