"""Clerk session authentication and role-aware authorization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

import jwt
from fastapi import Depends, HTTPException, Request, status
from jwt import PyJWKClient
from starlette.concurrency import run_in_threadpool


ROLE_PERMISSIONS = {
    "viewer": {"progress:read"},
    "member": {"progress:read", "progress:write"},
    "reviewer": {"progress:read", "progress:write", "progress:review"},
    "admin": {
        "progress:read",
        "progress:write",
        "progress:review",
        "progress:admin",
    },
}


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    organization_id: str | None
    role: str
    permissions: frozenset[str]
    claims: dict[str, Any]


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _auth_disabled() -> bool:
    disabled = _truthy(os.getenv("AUTH_DISABLED"))
    if disabled and (os.getenv("APP_ENV", "").strip().lower() == "production" or os.getenv("RAILWAY_PROJECT_ID")):
        raise HTTPException(status_code=503, detail="Development authentication is disabled in production.")
    return disabled


@lru_cache(maxsize=4)
def _jwks_client(issuer: str) -> PyJWKClient:
    return PyJWKClient(f"{issuer.rstrip('/')}/.well-known/jwks.json")


def _decode_token(token: str) -> dict[str, Any]:
    issuer = os.getenv("CLERK_ISSUER", "").strip().rstrip("/")
    if not issuer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Clerk authentication is not configured.",
        )

    try:
        signing_key = _jwks_client(issuer).get_signing_key_from_jwt(token)
        options = {"require": ["exp", "iat", "iss", "sub"]}
        audience = os.getenv("CLERK_AUDIENCE", "").strip() or None
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            options={**options, "verify_aud": audience is not None},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    authorized_parties = {
        item.strip()
        for item in os.getenv("CLERK_AUTHORIZED_PARTIES", "").split(",")
        if item.strip()
    }
    if authorized_parties and claims.get("azp") not in authorized_parties:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token is not authorized for this application.",
        )
    return claims


def _context_from_claims(claims: dict[str, Any]) -> AuthContext:
    organization = claims.get("o") if isinstance(claims.get("o"), dict) else {}
    expected_organization = os.getenv("CLERK_ORGANIZATION_ID", "").strip()
    if not expected_organization:
        raise HTTPException(status_code=503, detail="Project organization is not configured.")
    organization_id = organization.get("id") or claims.get("org_id")
    if organization_id != expected_organization:
        raise HTTPException(status_code=403, detail="Select the authorized project organization.")
    role = str(
        organization.get("rol")
        or claims.get("org_role")
        or ""
    ).removeprefix("org:")
    if role not in ROLE_PERMISSIONS:
        raise HTTPException(status_code=403, detail="Project role is not authorized.")
    permissions = ROLE_PERMISSIONS[role]
    return AuthContext(
        user_id=str(claims["sub"]),
        organization_id=organization_id,
        role=role,
        permissions=frozenset(permissions),
        claims=claims,
    )


async def get_auth_context(request: Request) -> AuthContext:
    if _auth_disabled():
        return AuthContext(
            user_id="local-admin",
            organization_id="local-development",
            role="admin",
            permissions=frozenset(ROLE_PERMISSIONS["admin"]),
            claims={"sub": "local-admin", "development": True},
        )

    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = await run_in_threadpool(_decode_token, token)
    return _context_from_claims(claims)


def require_permission(permission: str) -> Callable[..., AuthContext]:
    async def dependency(
        context: AuthContext = Depends(get_auth_context),
    ) -> AuthContext:
        if permission not in context.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission}",
            )
        return context

    return dependency
