"""GitHub device authentication for the hosted AI Function registry."""

from __future__ import annotations

import json
import os
import time
import webbrowser
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import keyring
from keyring.errors import KeyringError

DEFAULT_REGISTRY_URL = "https://ai-functions.dev"
GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
GITHUB_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
KEYRING_SERVICE = "jeva"
REQUEST_TIMEOUT_SECONDS = 15


class RegistryAuthError(RuntimeError):
    """A safe, user-facing authentication failure."""


@dataclass(frozen=True)
class RegistryCredential:
    registry: str
    login: str
    token: str
    expires_at: str


def registry_url(value: str | None = None) -> str:
    return (value or os.environ.get("JEVA_REGISTRY_URL") or DEFAULT_REGISTRY_URL).rstrip(
        "/"
    )


def _config_directory() -> Path:
    override = os.environ.get("JEVA_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "jeva"
    if sys_platform() == "darwin":
        return Path.home() / "Library" / "Application Support" / "jeva"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "jeva"


def sys_platform() -> str:
    # Isolated for deterministic tests without mutating sys.platform.
    import sys

    return sys.platform


class CredentialStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (_config_directory() / "credentials.json")

    def save(self, credential: RegistryCredential) -> str:
        storage = "keyring"
        payload: dict[str, object] = {
            "version": 1,
            "registry": credential.registry,
            "login": credential.login,
            "expires_at": credential.expires_at,
        }
        try:
            keyring.set_password(KEYRING_SERVICE, credential.registry, credential.token)
        except KeyringError:
            storage = "file"
            payload["token"] = credential.token
        payload["storage"] = storage
        self._write(payload)
        return storage

    def load(self, registry: str) -> RegistryCredential | None:
        environment_token = os.environ.get("JEVA_TOKEN")
        if environment_token:
            return RegistryCredential(registry, "", environment_token, "")
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version") != 1 or payload.get("registry") != registry:
            return None
        token: str | None
        if payload.get("storage") == "keyring":
            try:
                token = keyring.get_password(KEYRING_SERVICE, registry)
            except KeyringError:
                token = None
        else:
            token = payload.get("token")
        if not isinstance(token, str) or not token:
            return None
        return RegistryCredential(
            registry=registry,
            login=str(payload.get("login") or ""),
            token=token,
            expires_at=str(payload.get("expires_at") or ""),
        )

    def delete(self, registry: str) -> None:
        try:
            keyring.delete_password(KEYRING_SERVICE, registry)
        except KeyringError:
            pass
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if payload.get("registry") == registry:
            self.path.unlink(missing_ok=True)

    def _write(self, payload: Mapping[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.parent.chmod(0o700)
        except OSError:
            pass
        with NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temporary = Path(handle.name)
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, self.path)


def _request_json(
    url: str,
    *,
    method: str = "GET",
    json_body: Mapping[str, object] | None = None,
    form_body: Mapping[str, str] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "jeva (+https://github.com/sutro-sh/jev-align)",
    }
    data: bytes | None = None
    if json_body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(json_body).encode("utf-8")
    elif form_body is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        data = urlencode(form_body).encode("ascii")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = json.load(response)
    except HTTPError as error:
        try:
            payload = json.load(error)
            message = payload.get("error", {}).get("message")
        except (AttributeError, json.JSONDecodeError, TypeError):
            message = None
        raise RegistryAuthError(message or f"Registry returned HTTP {error.code}") from error
    except URLError as error:
        raise RegistryAuthError(f"Could not reach authentication service: {error.reason}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RegistryAuthError("Authentication service returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RegistryAuthError("Authentication service returned an invalid response")
    return payload


def login_with_github(
    *,
    registry: str | None = None,
    open_browser: bool = True,
    on_device_code: Callable[[str, str], None] | None = None,
    store: CredentialStore | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> tuple[RegistryCredential, str]:
    base = registry_url(registry)
    config = _request_json(f"{base}/api/v1/config")
    client_id = config.get("github_client_id")
    if not isinstance(client_id, str) or not client_id:
        raise RegistryAuthError("Registry did not provide a GitHub App client ID")

    device = _request_json(
        GITHUB_DEVICE_CODE_URL,
        method="POST",
        form_body={"client_id": client_id},
    )
    try:
        device_code = str(device["device_code"])
        user_code = str(device["user_code"])
        verification_uri = str(device["verification_uri"])
        expires_in = int(device["expires_in"])
        interval = max(1, int(device.get("interval", 5)))
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryAuthError("GitHub returned an invalid device authorization") from error

    if on_device_code:
        on_device_code(verification_uri, user_code)
    if open_browser:
        webbrowser.open(verification_uri)

    deadline = monotonic() + expires_in
    github_token: str | None = None
    while monotonic() < deadline:
        sleeper(interval)
        status = _request_json(
            GITHUB_ACCESS_TOKEN_URL,
            method="POST",
            form_body={
                "client_id": client_id,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        access_token = status.get("access_token")
        if isinstance(access_token, str) and access_token:
            github_token = access_token
            break
        error = status.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += max(1, int(status.get("interval", 5)))
            continue
        if error == "access_denied":
            raise RegistryAuthError("GitHub authorization was cancelled")
        if error in {"expired_token", "token_expired"}:
            raise RegistryAuthError("GitHub device code expired; run jeva login again")
        raise RegistryAuthError(str(status.get("error_description") or error or "GitHub login failed"))
    if github_token is None:
        raise RegistryAuthError("GitHub device code expired; run jeva login again")

    exchanged = _request_json(
        f"{base}/api/v1/auth/github/exchange",
        method="POST",
        json_body={"github_access_token": github_token},
    )
    token = exchanged.get("token")
    expires_at = exchanged.get("expires_at")
    user = exchanged.get("user")
    if (
        not isinstance(token, str)
        or not isinstance(expires_at, str)
        or not isinstance(user, dict)
        or not isinstance(user.get("login"), str)
    ):
        raise RegistryAuthError("Registry returned an invalid login response")
    credential = RegistryCredential(base, user["login"], token, expires_at)
    storage = (store or CredentialStore()).save(credential)
    return credential, storage


def authenticated_user(
    *, registry: str | None = None, store: CredentialStore | None = None
) -> dict[str, Any]:
    base = registry_url(registry)
    credential = (store or CredentialStore()).load(base)
    if credential is None:
        raise RegistryAuthError("Not signed in. Run jeva login first.")
    payload = _request_json(f"{base}/api/v1/me", token=credential.token)
    user = payload.get("user")
    if not isinstance(user, dict) or not isinstance(user.get("login"), str):
        raise RegistryAuthError("Your login has expired. Run jeva login again.")
    return user


def logout(*, registry: str | None = None, store: CredentialStore | None = None) -> bool:
    base = registry_url(registry)
    credentials = store or CredentialStore()
    credential = credentials.load(base)
    if credential is None:
        return False
    try:
        _request_json(
            f"{base}/api/v1/tokens/current",
            method="DELETE",
            token=credential.token,
        )
    except RegistryAuthError:
        # Local logout must still succeed if the registry is temporarily unavailable.
        pass
    credentials.delete(base)
    return True
