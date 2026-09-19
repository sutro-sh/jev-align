from __future__ import annotations

import pytest

from jev_align.backends import (
    BackendCapabilities,
    available_backend_providers,
    backend_credential_error,
    create_backend,
    default_backend_model,
    validate_backend_for_task,
)
from jev_align.jev import TypeSafeJevEvaluator
from jev_align.jev_gateways import CloudflareJevEvaluator, VercelJevEvaluator
from jev_align.models import BackendConfig, CandidateSpec


def test_typesafe_backend_factory_exposes_uncertainty_capabilities() -> None:
    backend = create_backend(
        BackendConfig(provider="typesafe", model="jev-test"), concurrency=3
    )

    assert isinstance(backend, TypeSafeJevEvaluator)
    assert backend.model == "jev-test"
    assert backend.concurrency == 3
    assert backend.capabilities.uncertainty["binary"] == "calibrated_probability"
    assert backend.capabilities.uncertainty["multiclass"] == "native_confidence"


def test_unknown_backend_fails_at_the_factory_boundary() -> None:
    with pytest.raises(ValueError, match="unsupported evaluation backend: local"):
        create_backend(BackendConfig(provider="local", model="test"), concurrency=1)


def test_backend_credentials_are_provider_specific() -> None:
    config = BackendConfig(provider="typesafe", model="jev-test")

    assert backend_credential_error(config, {}) is not None
    assert backend_credential_error(config, {"TYPESAFE_API_KEY": "key"}) is None

    cloudflare = BackendConfig(provider="cloudflare", model="typesafe/jev")
    assert "CLOUDFLARE_ACCOUNT_ID" in backend_credential_error(cloudflare, {})
    assert (
        backend_credential_error(
            cloudflare,
            {
                "CLOUDFLARE_ACCOUNT_ID": "account",
                "CLOUDFLARE_API_TOKEN": "token",
            },
        )
        is None
    )

    vercel = BackendConfig(provider="vercel", model="typesafe-ai/jev")
    assert backend_credential_error(vercel, {}) is not None
    assert backend_credential_error(vercel, {"AI_GATEWAY_API_KEY": "key"}) is None


def test_gateway_backend_factories_and_defaults() -> None:
    cloudflare = create_backend(
        BackendConfig(provider="cloudflare", model="typesafe/jev"), concurrency=2
    )
    vercel = create_backend(
        BackendConfig(provider="vercel", model="typesafe-ai/jev"), concurrency=4
    )

    assert isinstance(cloudflare, CloudflareJevEvaluator)
    assert cloudflare.concurrency == 2
    assert isinstance(vercel, VercelJevEvaluator)
    assert vercel.concurrency == 4
    assert default_backend_model("cloudflare") == "typesafe/jev"
    assert default_backend_model("vercel") == "typesafe-ai/jev"


def test_available_backends_are_ordered_and_require_complete_cloudflare_config() -> (
    None
):
    environment = {
        "TYPESAFE_API_KEY": "direct",
        "AI_GATEWAY_API_KEY": "gateway",
        "CLOUDFLARE_ACCOUNT_ID": "account",
        "CLOUDFLARE_API_TOKEN": "token",
    }

    assert [provider for provider, _ in available_backend_providers(environment)] == [
        "typesafe",
        "vercel",
        "cloudflare",
    ]
    assert available_backend_providers({"CLOUDFLARE_API_TOKEN": "token"}) == []


def test_backend_must_supply_uncertainty_for_the_task() -> None:
    backend = create_backend(
        BackendConfig(provider="typesafe", model="jev-test"), concurrency=1
    )
    backend.capabilities = BackendCapabilities(
        task_types=frozenset({"binary"}), uncertainty={}
    )
    candidate = CandidateSpec(
        instructions="Is this relevant?",
        true_criteria="Relevant.",
        false_criteria="Not relevant.",
    )

    with pytest.raises(ValueError, match="does not provide usable uncertainty"):
        validate_backend_for_task(backend, candidate)
