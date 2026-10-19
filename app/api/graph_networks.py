"""Source-preserving graph combination and library database snapshots."""
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.repository.graph_networks import shared_network_store

router = APIRouter(prefix="/graph-networks", tags=["graph-networks"])


class NetworkCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    run_ids: list[str] = Field(default_factory=list, max_length=200)
    scope: Literal["selected", "library"] = "selected"
    infer_connections: bool = True


def mapped_error(exc):
    if isinstance(exc, FileNotFoundError):
        return HTTPException(404, "Graph network, research run or source artifact not found.")
    if isinstance(exc, OSError) and not isinstance(exc, FileNotFoundError):
        return HTTPException(503, "Graph network storage is temporarily unavailable. Retry after storage recovers.")
    return HTTPException(422, str(exc))


@router.get("")
def list_networks(limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0, le=100000)):
    try:
        return shared_network_store().list(limit, offset)
    except OSError as exc:
        raise mapped_error(exc) from exc


@router.post("", status_code=201)
async def create_network(body: NetworkCreate):
    try:
        return await run_in_threadpool(shared_network_store().create, body.title, body.run_ids, body.scope, body.infer_connections)
    except (ValueError, OSError) as exc:
        raise mapped_error(exc) from exc


@router.get("/{network_id}")
def get_network(network_id: str):
    try:
        return shared_network_store().get(network_id)["network"]
    except (ValueError, OSError) as exc:
        raise mapped_error(exc) from exc


@router.post("/{network_id}/refresh", status_code=201)
async def refresh_network(network_id: str):
    """Build a new snapshot with current saved candidates and connection methods."""
    try:
        return await run_in_threadpool(shared_network_store().refresh, network_id)
    except (ValueError, OSError) as exc:
        raise mapped_error(exc) from exc


@router.get("/{network_id}/graph")
def network_graph(network_id: str):
    try:
        return shared_network_store().get(network_id)["graph"]
    except (ValueError, OSError) as exc:
        raise mapped_error(exc) from exc


@router.get("/{network_id}/export")
def export_network(network_id: str):
    try:
        store = shared_network_store()
        store.get(network_id)  # Validate cached JSON before offering a download.
        return Response(store.path(network_id).read_bytes(), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="graph-network-{network_id}.json"'})
    except (ValueError, OSError) as exc:
        raise mapped_error(exc) from exc
