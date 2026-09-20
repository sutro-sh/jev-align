from __future__ import annotations

import json
from pathlib import Path

import pytest
from keyring.errors import NoKeyringError

from jev_align import registry_auth


class MemoryStore:
    def __init__(self) -> None:
        self.saved: registry_auth.RegistryCredential | None = None

    def save(self, credential: registry_auth.RegistryCredential) -> str:
        self.saved = credential
        return "keyring"


def test_github_device_login_exchanges_temporary_token(monkeypatch) -> None:
    responses = {
        "https://registry.example/api/v1/config": {
            "github_client_id": "Iv1.client",
        },
        registry_auth.GITHUB_DEVICE_CODE_URL: {
            "device_code": "device-secret",
            "user_code": "ABCD-EFGH",
            "verification_uri": "https://github.com/login/device",
            "expires_in": 900,
            "interval": 1,
        },
        registry_auth.GITHUB_ACCESS_TOKEN_URL: [
            {"error": "authorization_pending"},
            {"access_token": "github-temporary-token"},
        ],
        "https://registry.example/api/v1/auth/github/exchange": {
            "token": "jeva_registry-token",
            "expires_at": "2027-01-01T00:00:00Z",
            "user": {"login": "octocat"},
        },
    }
    calls: list[tuple[str, dict[str, object]]] = []

    def request(url: str, **kwargs):
        calls.append((url, kwargs))
        response = responses[url]
        if isinstance(response, list):
            return response.pop(0)
        return response

    monkeypatch.setattr(registry_auth, "_request_json", request)
    opened: list[str] = []
    monkeypatch.setattr(registry_auth.webbrowser, "open", opened.append)
    shown: list[tuple[str, str]] = []
    store = MemoryStore()

    credential, storage = registry_auth.login_with_github(
        registry="https://registry.example/",
        on_device_code=lambda uri, code: shown.append((uri, code)),
        store=store,  # type: ignore[arg-type]
        sleeper=lambda _seconds: None,
        monotonic=lambda: 0.0,
    )

    assert credential.login == "octocat"
    assert credential.token == "jeva_registry-token"
    assert storage == "keyring"
    assert store.saved == credential
    assert opened == ["https://github.com/login/device"]
    assert shown == [("https://github.com/login/device", "ABCD-EFGH")]
    exchange = calls[-1][1]["json_body"]
    assert exchange == {"github_access_token": "github-temporary-token"}


def test_credential_store_falls_back_to_restricted_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        registry_auth.keyring,
        "set_password",
        lambda *_args: (_ for _ in ()).throw(NoKeyringError()),
    )
    path = tmp_path / "jeva" / "credentials.json"
    store = registry_auth.CredentialStore(path)
    credential = registry_auth.RegistryCredential(
        registry="https://registry.example",
        login="octocat",
        token="jeva_secret",
        expires_at="2027-01-01T00:00:00Z",
    )

    assert store.save(credential) == "file"
    assert store.load(credential.registry) == credential
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["storage"] == "file"
    assert payload["token"] == "jeva_secret"
    assert path.stat().st_mode & 0o777 == 0o600


def test_credential_store_prefers_environment_token(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("JEVA_TOKEN", "jeva_from_environment")
    credential = registry_auth.CredentialStore(tmp_path / "missing.json").load(
        "https://registry.example"
    )

    assert credential is not None
    assert credential.token == "jeva_from_environment"


def test_whoami_requires_login(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("JEVA_TOKEN", raising=False)
    with pytest.raises(registry_auth.RegistryAuthError, match="Run jeva login"):
        registry_auth.authenticated_user(
            registry="https://registry.example",
            store=registry_auth.CredentialStore(Path(tmp_path) / "missing.json"),
        )
