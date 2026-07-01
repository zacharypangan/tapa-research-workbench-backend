import os
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import progress
from app.auth import AuthContext, get_auth_context


class ProgressAuthTests(unittest.TestCase):
    def make_client(self) -> TestClient:
        app = FastAPI()
        app.include_router(progress.router, prefix="/api/v1")
        return TestClient(app)

    def test_progress_requires_authentication(self):
        with patch.dict(os.environ, {"AUTH_DISABLED": "false", "CLERK_ISSUER": ""}):
            response = self.make_client().get("/api/v1/progress/status")
        self.assertEqual(response.status_code, 401)

    def test_local_auth_mode_exposes_admin_permissions(self):
        with patch.dict(os.environ, {"AUTH_DISABLED": "true"}):
            response = self.make_client().get("/api/v1/progress/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["role"], "admin")
        self.assertIn("progress:admin", response.json()["permissions"])

    def test_missing_permission_returns_forbidden(self):
        app = FastAPI()
        app.include_router(progress.router, prefix="/api/v1")
        app.dependency_overrides[get_auth_context] = lambda: AuthContext(
            user_id="member-1",
            organization_id="org-1",
            role="member",
            permissions=frozenset(),
            claims={"sub": "member-1"},
        )
        response = TestClient(app).get("/api/v1/progress/status")
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
