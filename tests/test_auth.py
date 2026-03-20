import pytest

from app.auth.jwt_auth import create_token, verify_token

SECRET = "s3cret-key-of-realistic-length-0123456789"
OTHER = "other-key-of-realistic-length-9876543210"


def test_token_roundtrip():
    token = create_token("alice", secret=SECRET, ttl_seconds=60)
    claims = verify_token(token, secret=SECRET)
    assert claims["sub"] == "alice"
    assert claims["exp"] > claims["iat"]


def test_wrong_secret_rejected():
    import jwt as pyjwt

    token = create_token("alice", secret=SECRET, ttl_seconds=60)
    with pytest.raises(pyjwt.PyJWTError):
        verify_token(token, secret=OTHER)


def test_expired_token_rejected():
    import jwt as pyjwt

    token = create_token("alice", secret=SECRET, ttl_seconds=-10)
    with pytest.raises(pyjwt.ExpiredSignatureError):
        verify_token(token, secret=SECRET)


def test_endpoints_enforce_auth_when_enabled():
    from fastapi.testclient import TestClient

    from app.main import app
    from tests.test_api import make_client

    client, _ = make_client()
    assert isinstance(client, TestClient)
    with client:
        app.state.settings.auth_enabled = True
        app.state.settings.jwt_secret = SECRET
        try:
            # No token -> 401
            resp = client.post("/query", json={"question": "hi"})
            assert resp.status_code == 401

            # Garbage token -> 401
            resp = client.post(
                "/query",
                json={"question": "hi"},
                headers={"authorization": "Bearer not-a-jwt"},
            )
            assert resp.status_code == 401

            # Valid token -> 200
            token = create_token("tester", secret=SECRET, ttl_seconds=60)
            resp = client.post(
                "/query",
                json={"question": "hi"},
                headers={"authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 200

            # /healthz stays public
            assert client.get("/healthz").status_code == 200
        finally:
            app.state.settings.auth_enabled = False
    app.dependency_overrides.clear()
