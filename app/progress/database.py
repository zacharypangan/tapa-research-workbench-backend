"""Database engine and sessions for portal-owned relational data."""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def database_url() -> str:
    configured = os.getenv("DATABASE_URL", "").strip()
    if configured.startswith("postgres://"):
        return configured.replace("postgres://", "postgresql+psycopg://", 1)
    if configured.startswith("postgresql://"):
        return configured.replace("postgresql://", "postgresql+psycopg://", 1)
    if configured:
        return configured
    local_path = Path(__file__).resolve().parents[2] / "storage" / "progress.sqlite"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{local_path}"


class Base(DeclarativeBase):
    pass


DATABASE_URL = database_url()
ENGINE = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=ENGINE, expire_on_commit=False, class_=Session)


def get_session() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
