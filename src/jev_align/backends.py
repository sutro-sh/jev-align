from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from .models import (
    DEFAULT_JEV_MODEL,
    BackendConfig,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    Prediction,
    ScoreCandidateSpec,
    Story,
    TaskSpec,
)

TaskKind = Literal["binary", "multiclass", "multilabel", "score"]
UncertaintyKind = Literal["calibrated_probability", "native_confidence"]

DEFAULT_BACKEND_MODELS = {
    "typesafe": DEFAULT_JEV_MODEL,
    "cloudflare": "typesafe/jev-1.14.0",
    "vercel": "typesafe-ai/jev-1.14.0",
    "perplexity": "pplx-decider-v1.2",
    "openai": "gpt-6-luna",
    "liquid": "d1.1",
    "openrouter": "typesafe/jev-1.14",
}


OPENROUTER_MODEL_CHOICES = [
    ("typesafe/jev-1.14", "Jev 1.14"),
    ("perplexity/pplx-decider-v1.2", "Perplexity Decider v1.2"),
    ("openai/gpt-6-luna-decisions", "GPT-6 Luna Decisions"),
    ("liquid/d1.1", "Liquid D1.1"),
    ("cloudflare/clef", "Clef"),
    ("cloudflare/clef-flash", "Clef Flash"),
]


@dataclass(frozen=True)
class BackendCapabilities:
    """What an evaluation backend can reliably provide to active learning."""

    task_types: frozenset[TaskKind]
    uncertainty: Mapping[TaskKind, UncertaintyKind]


class EvaluationBackend(Protocol):
    backend_id: str
    display_name: str
    model: str
    capabilities: BackendCapabilities

    def evaluate_many(
        self, candidate: TaskSpec, stories: Sequence[Story]
    ) -> list[Prediction]: ...


def create_backend(config: BackendConfig, *, concurrency: int) -> EvaluationBackend:
    """Construct the configured adapter without leaking providers into the core."""
    if config.provider == "typesafe":
        from .jev import TypeSafeJevEvaluator

        return TypeSafeJevEvaluator(model=config.model, concurrency=concurrency)
    if config.provider == "cloudflare":
        from .decision_backends import CLEF_MODELS, CloudflareClefEvaluator
        from .jev_gateways import CloudflareJevEvaluator

        if config.model in CLEF_MODELS:
            return CloudflareClefEvaluator(model=config.model, concurrency=concurrency)
        return CloudflareJevEvaluator(model=config.model, concurrency=concurrency)
    if config.provider == "vercel":
        from .jev_gateways import VercelJevEvaluator

        return VercelJevEvaluator(model=config.model, concurrency=concurrency)
    if config.provider in {"perplexity", "openai", "liquid", "openrouter"}:
        from .decision_backends import (
            LiquidDecisionEvaluator,
            OpenAIDecisionEvaluator,
            OpenRouterDecisionEvaluator,
            PerplexityDeciderEvaluator,
        )

        adapter = {
            "perplexity": PerplexityDeciderEvaluator,
            "openai": OpenAIDecisionEvaluator,
            "liquid": LiquidDecisionEvaluator,
            "openrouter": OpenRouterDecisionEvaluator,
        }[config.provider]
        return adapter(model=config.model, concurrency=concurrency)
    raise ValueError(f"unsupported evaluation backend: {config.provider}")


def backend_credential_error(
    config: BackendConfig, environment: Mapping[str, str]
) -> str | None:
    if config.provider == "typesafe" and not environment.get("TYPESAFE_API_KEY"):
        return "TYPESAFE_API_KEY must be set for the TypeSafe backend"
    if config.provider == "cloudflare":
        missing = [
            name
            for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN")
            if not environment.get(name)
        ]
        if missing:
            return f"{', '.join(missing)} must be set for the Cloudflare backend"
    if config.provider == "vercel" and not environment.get("AI_GATEWAY_API_KEY"):
        return "AI_GATEWAY_API_KEY must be set for the Vercel backend"
    key = {
        "perplexity": "PERPLEXITY_API_KEY",
        "openai": "OPENAI_API_KEY",
        "liquid": "LIQUID_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(config.provider)
    if key and not environment.get(key):
        return f"{key} must be set for the {config.provider} evaluation backend"
    return None


def default_backend_model(provider: str) -> str:
    try:
        return DEFAULT_BACKEND_MODELS[provider]
    except KeyError as error:
        raise ValueError(f"unsupported evaluation backend: {provider}") from error


def available_backend_providers(
    environment: Mapping[str, str],
) -> list[tuple[str, str]]:
    """Return configured evaluation providers in preferred order."""
    available: list[tuple[str, str]] = []
    if environment.get("TYPESAFE_API_KEY"):
        available.append(("typesafe", "TypeSafe AI (direct)"))
    if environment.get("AI_GATEWAY_API_KEY"):
        available.append(("vercel", "Vercel AI Gateway"))
    if environment.get("CLOUDFLARE_ACCOUNT_ID") and environment.get(
        "CLOUDFLARE_API_TOKEN"
    ):
        available.append(("cloudflare", "Cloudflare Workers AI"))
    for provider, key, label in (
        ("perplexity", "PERPLEXITY_API_KEY", "Perplexity Decider"),
        ("openai", "OPENAI_API_KEY", "OpenAI Decisions · GPT-6 Luna"),
        ("liquid", "LIQUID_API_KEY", "Liquid D1"),
        ("openrouter", "OPENROUTER_API_KEY", "OpenRouter Decisions"),
    ):
        if environment.get(key):
            available.append((provider, label))
    return available


def backend_display_name(backend: EvaluationBackend) -> str:
    """Give minimal third-party adapters a useful name without extra boilerplate."""
    return str(getattr(backend, "display_name", backend.__class__.__name__))


def task_kind(candidate: TaskSpec) -> TaskKind:
    if isinstance(candidate, MultilabelCandidateSpec):
        return "multilabel"
    if isinstance(candidate, ScoreCandidateSpec):
        return "score"
    if isinstance(candidate, MulticlassCandidateSpec):
        return "multiclass"
    return "binary"


def validate_backend_for_task(backend: EvaluationBackend, candidate: TaskSpec) -> None:
    """Reject adapters that cannot supply uncertainty for the requested task."""
    kind = task_kind(candidate)
    capabilities = backend.capabilities
    if kind not in capabilities.task_types:
        raise ValueError(
            f"{backend_display_name(backend)} does not support {kind} AI Functions"
        )
    if kind not in capabilities.uncertainty:
        raise ValueError(
            f"{backend_display_name(backend)} does not provide usable uncertainty "
            f"for {kind} AI Functions"
        )


def evaluate_with_progress(
    backend: EvaluationBackend,
    candidate: TaskSpec,
    stories: Sequence[Story],
    description: str,
) -> list[Prediction]:
    """Use progress when supported, while preserving lightweight test adapters."""
    method = getattr(backend, "evaluate_many_with_progress", None)
    if callable(method):
        return method(candidate, stories, description)
    return backend.evaluate_many(candidate, stories)
