from __future__ import annotations

import json
import hashlib

from jev_align import registry_client
from jev_align.artifacts import FunctionArtifact
from jev_align.registry_auth import RegistryCredential


class MemoryStore:
    def load(self, registry: str) -> RegistryCredential:
        return RegistryCredential(registry, "octocat", "jeva_secret", "later")


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "reference": "octocat/is-aviation",
                "version": 1,
                "digest": "abc123",
                "url": "https://ai-functions.dev/octocat/is-aviation",
                "created": True,
            }
        ).encode()


def _artifact_payload() -> bytes:
    artifact = FunctionArtifact.model_validate(
        {
            "schema_version": 1,
            "name": "Is aviation",
            "slug": "is-aviation",
            "task_type": "binary",
            "inputs": {"columns": ["text"], "mode": "selected"},
            "definition": {
                "instructions": "Is this aviation?",
                "true_criteria": "Aircraft and flight.",
                "false_criteria": "Everything else.",
            },
            "backend": {"provider": "typesafe", "model": "jev-1.13.0"},
            "annotations": [
                {
                    "inputs": {"text": "A new runway"},
                    "label": True,
                    "rationale": "A runway is aviation infrastructure.",
                    "split": "train",
                }
            ],
            "learning": {
                "reflection_model": "openai/gpt-5.6-luna",
                "metric_budget": 300,
                "batch_size": 5,
                "holdout_fraction": 0.0,
                "concurrency": 16,
                "seed": 0,
            },
            "metrics": {"training_score": 1.0},
        }
    )
    return artifact.model_dump_json().encode()


class PullResponse:
    def __init__(self, payload: bytes, *, digest: str | None = None):
        self.payload = payload
        self.headers = {} if digest is None else {"X-Jeva-Digest": digest}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _limit: int | None = None) -> bytes:
        return self.payload


def test_unpublish_sends_authenticated_delete(monkeypatch) -> None:
    received = {}

    def open_request(request, *, timeout):
        received["request"] = request
        return PullResponse(
            json.dumps(
                {
                    "reference": "octocat/is-aviation",
                    "unpublished": True,
                    "changed": True,
                }
            ).encode()
        )

    monkeypatch.setattr(registry_client, "urlopen", open_request)
    result = registry_client.unpublish_function(
        "@octocat/is-aviation",
        registry="https://registry.example/",
        store=MemoryStore(),  # type: ignore[arg-type]
    )

    request = received["request"]
    assert request.full_url == (
        "https://registry.example/api/v1/functions/octocat/is-aviation"
    )
    assert request.method == "DELETE"
    assert request.get_header("Authorization") == "Bearer jeva_secret"
    assert result.reference == "octocat/is-aviation"
    assert result.changed is True


def test_publish_sends_exact_artifact_with_bearer_token(monkeypatch) -> None:
    artifact = b'{"schema_version":1}\n'
    received = {}

    def open_request(request, *, timeout):
        received["request"] = request
        received["timeout"] = timeout
        return Response()

    monkeypatch.setattr(registry_client, "urlopen", open_request)
    result = registry_client.publish_artifact(
        artifact,
        slug="is-aviation",
        registry="https://registry.example/",
        store=MemoryStore(),  # type: ignore[arg-type]
    )

    request = received["request"]
    assert request.full_url == "https://registry.example/api/v1/functions/is-aviation"
    assert request.data == artifact
    assert request.get_header("Authorization") == "Bearer jeva_secret"
    assert result.reference == "octocat/is-aviation"
    assert result.version == 1


def test_pull_verifies_and_parses_latest_artifact(monkeypatch) -> None:
    artifact = _artifact_payload()
    digest = hashlib.sha256(artifact).hexdigest()
    requests = []

    def open_request(request, *, timeout):
        requests.append((request, timeout))
        if request.full_url.endswith("/octocat/is-aviation"):
            return PullResponse(
                json.dumps({"version": 2, "digest": digest}).encode()
            )
        return PullResponse(artifact, digest=digest)

    monkeypatch.setattr(registry_client, "urlopen", open_request)
    result = registry_client.pull_artifact(
        "octocat/is-aviation",
        registry="https://registry.example/",
    )

    assert result.reference == "octocat/is-aviation"
    assert result.version == 2
    assert result.digest == digest
    assert result.artifact.name == "Is aviation"
    assert requests[1][0].full_url.endswith(
        "/octocat/is-aviation/versions/2/artifact"
    )


def test_pull_explicit_old_version_uses_immutable_response_digest(monkeypatch) -> None:
    artifact = _artifact_payload()
    old_digest = hashlib.sha256(artifact).hexdigest()

    def open_request(request, *, timeout):
        if request.full_url.endswith("/octocat/is-aviation"):
            return PullResponse(
                json.dumps({"version": 3, "digest": "latest-digest"}).encode()
            )
        return PullResponse(artifact, digest=old_digest)

    monkeypatch.setattr(registry_client, "urlopen", open_request)
    result = registry_client.pull_artifact(
        "octocat/is-aviation",
        version=1,
        registry="https://registry.example",
    )

    assert result.version == 1
    assert result.digest == old_digest


def test_pull_rejects_artifact_with_wrong_digest(monkeypatch) -> None:
    artifact = _artifact_payload()

    def open_request(request, *, timeout):
        if request.full_url.endswith("/octocat/is-aviation"):
            return PullResponse(
                json.dumps({"version": 1, "digest": "wrong"}).encode()
            )
        return PullResponse(artifact, digest="wrong")

    monkeypatch.setattr(registry_client, "urlopen", open_request)

    try:
        registry_client.pull_artifact(
            "octocat/is-aviation",
            registry="https://registry.example",
        )
    except registry_client.RegistryPullError as error:
        assert "integrity" in str(error)
    else:
        raise AssertionError("corrupt artifact was accepted")
