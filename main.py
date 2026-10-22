import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.db import init_db
from app.api import tiles, chat, data, repository, research_runs, processing, translations, graph_networks
from app.repository.document_graph_runtime import get_scheduler
from app.repository.processing_jobs import get_processing_scheduler
from app.repository.translations import get_translation_scheduler

app = FastAPI(title="Tapa Research Workbench Backend")

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
    origins.extend([o.strip() for o in env_origins.split(",") if o.strip()])

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tiles.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(data.router, prefix="/api/v1")
app.include_router(repository.router, prefix="/api/v1")
app.include_router(research_runs.router, prefix="/api/v1")
app.include_router(processing.router, prefix="/api/v1")
app.include_router(translations.router, prefix="/api/v1")
app.include_router(graph_networks.router, prefix="/api/v1")


@app.on_event("startup")
async def on_startup():
    init_db()
    repository.init_repository_db()
    get_scheduler().start()
    get_processing_scheduler().start()
    get_translation_scheduler().start()


@app.on_event("shutdown")
async def on_shutdown():
    await get_translation_scheduler().stop()
    await get_processing_scheduler().stop()
    await get_scheduler().stop()


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
