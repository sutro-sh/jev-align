"""Authenticated publishing client for the AI Function registry."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .artifacts import FunctionArtifact
from .registry_auth import CredentialStore, registry_url

REQUEST_TIMEOUT_SECONDS = 30


class RegistryPublishError(RuntimeError):
    """A safe, user-facing publishing failure."""


class RegistryPullError(RuntimeError):
    """A safe, user-facing function download failure."""


class RegistryUnpublishError(RuntimeError):
    """A safe, user-facing function unpublishing failure."""


@dataclass(frozen=True)
class PublishResult:
    reference: str
    version: int
    digest: str
    url: str
    created: bool


@dataclass(frozen=True)
class PullResult:
    reference: str
    version: int
    digest: str
    artifact: FunctionArtifact
    registry: str


@dataclass(frozen=True)
class UnpublishResult:
    reference: str
    changed: bool


def _function_reference(reference: str) -> tuple[str, str]:
    normalized = reference.strip()
    if normalized.startswith("@"):
        normalized = normalized[1:]
    parts = normalized.split("/")
    identifier = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
    if len(parts) != 2 or any(identifier.fullmatch(part) is None for part in parts):
        raise ValueError("Function reference must be namespace/function-name")
    return parts[0], parts[1]


def publish_artifact(
    artifact: bytes,
    *,
    slug: str,
    registry: str | None = None,
    store: CredentialStore | None = None,
) -> PublishResult:
    base = registry_url(registry)
    credential = (store or CredentialStore()).load(base)
    if credential is None:
        raise RegistryPublishError("Not signed in. Run jeva login first.")
    request = Request(
        f"{base}/api/v1/functions/{quote(slug, safe='')}",
        data=artifact,
        method="PUT",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {credential.token}",
            "Content-Type": "application/json",
            "User-Agent": "jeva (+https://github.com/sutro-sh/jev-align)",
        },
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload: Any = json.load(response)
    except HTTPError as error:
        try:
            body = json.load(error)
            message = body.get("error", {}).get("message")
        except (AttributeError, json.JSONDecodeError, TypeError):
            message = None
        raise RegistryPublishError(
            message or f"Registry returned HTTP {error.code}"
        ) from error
    except URLError as error:
        raise RegistryPublishError(
            f"Could not reach the function registry: {error.reason}"
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RegistryPublishError("Registry returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RegistryPublishError("Registry returned an invalid response")
    try:
        return PublishResult(
            reference=str(payload["reference"]),
            version=int(payload["version"]),
            digest=str(payload["digest"]),
            url=str(payload["url"]),
            created=bool(payload["created"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryPublishError("Registry returned an invalid response") from error


def _pull_error(error: HTTPError) -> RegistryPullError:
    try:
        body = json.load(error)
        message = body.get("error", {}).get("message")
    except (AttributeError, json.JSONDecodeError, TypeError):
        message = None
    return RegistryPullError(message or f"Registry returned HTTP {error.code}")


def unpublish_function(
    reference: str,
    *,
    registry: str | None = None,
    store: CredentialStore | None = None,
) -> UnpublishResult:
    """Hide a function from public discovery and pulls without deleting versions."""
    try:
        namespace, slug = _function_reference(reference)
    except ValueError as error:
        raise RegistryUnpublishError(str(error)) from error
    base = registry_url(registry)
    credential = (store or CredentialStore()).load(base)
    if credential is None:
        raise RegistryUnpublishError("Not signed in. Run jeva login first.")
    request = Request(
        f"{base}/api/v1/functions/{quote(namespace, safe='')}/{quote(slug, safe='')}",
        method="DELETE",
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {credential.token}",
            "User-Agent": "jeva (+https://github.com/sutro-sh/jev-align)",
        },
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload: Any = json.load(response)
    except HTTPError as error:
        try:
            body = json.load(error)
            message = body.get("error", {}).get("message")
        except (AttributeError, json.JSONDecodeError, TypeError):
            message = None
        raise RegistryUnpublishError(
            message or f"Registry returned HTTP {error.code}"
        ) from error
    except URLError as error:
        raise RegistryUnpublishError(
            f"Could not reach the function registry: {error.reason}"
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RegistryUnpublishError("Registry returned invalid JSON") from error
    if not isinstance(payload, dict):
        raise RegistryUnpublishError("Registry returned an invalid response")
    try:
        if payload["unpublished"] is not True:
            raise ValueError
        return UnpublishResult(
            reference=str(payload["reference"]),
            changed=bool(payload["changed"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryUnpublishError("Registry returned an invalid response") from error


def pull_artifact(
    reference: str,
    *,
    version: int | None = None,
    registry: str | None = None,
) -> PullResult:
    """Fetch and verify one immutable public AI Function artifact."""
    try:
        namespace, slug = _function_reference(reference)
    except ValueError as error:
        raise RegistryPullError(str(error)) from error
    if version is not None and version < 1:
        raise RegistryPullError("Function version must be positive")
    base = registry_url(registry)
    detail_url = f"{base}/api/v1/functions/{quote(namespace, safe='')}/{quote(slug, safe='')}"
    detail_request = Request(
        detail_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "jeva (+https://github.com/sutro-sh/jev-align)",
        },
    )
    try:
        with urlopen(detail_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            detail: Any = json.load(response)
    except HTTPError as error:
        raise _pull_error(error) from error
    except URLError as error:
        raise RegistryPullError(
            f"Could not reach the function registry: {error.reason}"
        ) from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RegistryPullError("Registry returned invalid function metadata") from error
    if not isinstance(detail, dict):
        raise RegistryPullError("Registry returned invalid function metadata")
    try:
        latest_version = int(detail["version"])
        latest_digest = str(detail["digest"])
    except (KeyError, TypeError, ValueError) as error:
        raise RegistryPullError("Registry returned invalid function metadata") from error

    resolved_version = version or latest_version
    artifact_url = f"{detail_url}/versions/{resolved_version}/artifact"
    artifact_request = Request(
        artifact_url,
        headers={
            "Accept": "application/json",
            "User-Agent": "jeva (+https://github.com/sutro-sh/jev-align)",
        },
    )
    try:
        with urlopen(artifact_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = response.read(50 * 1024 * 1024 + 1)
            response_digest = response.headers.get("X-Jeva-Digest")
    except HTTPError as error:
        raise _pull_error(error) from error
    except URLError as error:
        raise RegistryPullError(
            f"Could not reach the function registry: {error.reason}"
        ) from error
    if len(payload) > 50 * 1024 * 1024:
        raise RegistryPullError("Function artifact exceeds 50 MiB")
    if response_digest is None:
        raise RegistryPullError("Registry omitted the function artifact digest")
    expected_digest = latest_digest if version is None else response_digest
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != expected_digest or response_digest != expected_digest:
        raise RegistryPullError("Function artifact failed its integrity check")
    try:
        artifact = FunctionArtifact.model_validate_json(payload)
    except ValueError as error:
        raise RegistryPullError("Function artifact has an unsupported schema") from error
    if artifact.slug != slug:
        raise RegistryPullError("Function artifact does not match its registry reference")
    return PullResult(
        reference=f"{namespace}/{slug}",
        version=resolved_version,
        digest=expected_digest,
        artifact=artifact,
        registry=base,
    )
