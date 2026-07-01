"""Progress portal API entry point."""

from fastapi import APIRouter, Depends

from app.auth import AuthContext, require_permission


router = APIRouter(prefix="/progress", tags=["progress"])


@router.get("/status")
def progress_status(
    context: AuthContext = Depends(require_permission("progress:read")),
):
    return {
        "available": True,
        "user_id": context.user_id,
        "role": context.role,
        "permissions": sorted(context.permissions),
    }
