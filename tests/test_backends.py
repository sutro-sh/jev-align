from __future__ import annotations

import pytest

from jev_align.backends import (
    BackendCapabilities,
    backend_credential_error,
    create_backend,
    validate_backend_for_task,
)
from jev_align.jev import TypeSafeJevEvaluator
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
