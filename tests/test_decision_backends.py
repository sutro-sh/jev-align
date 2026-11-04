from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from jev_align import decision_backends, decision_http
from jev_align.acquisition import prediction_ambiguity
from jev_align.backends import (
    available_backend_providers,
    backend_credential_error,
    create_backend,
    default_backend_model,
    validate_backend_for_task,
)
from jev_align.models import (
    BackendConfig,
    CandidateSpec,
    MulticlassCandidateSpec,
    MultilabelCandidateSpec,
    MultilabelCriteria,
    ScoreCandidateSpec,
    Story,
)

BACKENDS = [
    ("perplexity", "pplx-decider-v1.2", "PERPLEXITY_API_KEY"),
    ("openrouter", "typesafe/jev-1.14", "OPENROUTER_API_KEY"),
    ("liquid", "d1.1", "LIQUID_API_KEY"),
    ("openai", "gpt-6-luna", "OPENAI_API_KEY"),
    ("cloudflare", "@cf/cloudflare/clef", "CLOUDFLARE_API_TOKEN"),
    ("cloudflare", "@cf/cloudflare/clef-flash", "CLOUDFLARE_API_TOKEN"),
]
TASKS = [
    CandidateSpec(
        instructions="Relevant?", true_criteria="Aviation", false_criteria="Other"
    ),
    MulticlassCandidateSpec(
        instructions="Route ticket", criteria={"billing": "Money", "technical": "Bugs"}
    ),
    ScoreCandidateSpec(instructions="Severity?", levels=["Low", "Medium", "High"]),
    MultilabelCandidateSpec(
        instructions="Tag ticket",
        labels={
            "billing": MultilabelCriteria(
                true_criteria="Money", false_criteria="Other"
            ),
            "urgent": MultilabelCriteria(
                true_criteria="Urgent", false_criteria="Can wait"
            ),
        },
    ),
]


def _story(name="row-1"):
    return Story(
        id=name, row_number=1, fields={"title": "Airport", "text": "Runway opens"}
    )


def response_for(task, *, openai=False):
    if isinstance(task, MulticlassCandidateSpec):
        answer = {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.6,
            "probabilities": {"billing": 0.8, "technical": 0.2},
        }
    elif isinstance(task, ScoreCandidateSpec):
        answer = {
            "type": "score",
            "score": 1.25,
            "confidence": 0.7,
            "probabilities": {"0": 0.0, "1": 0.75, "2": 0.25},
        }
    else:
        answer = {"type": "noul", "noul": 0.63}
    answers = {"alignment": answer}
    if isinstance(task, MultilabelCandidateSpec):
        answers = {
            "label_0": {"type": "noul", "noul": 0.9},
            "label_1": {"type": "noul", "noul": 0.4},
        }
    if openai:
        translated = []
        for name, answer in answers.items():
            answer = dict(answer, name=name)
            if answer["type"] == "noul":
                answer["type"] = "predicate"
                answer["probability"] = answer.pop("noul")
            else:
                answer["probabilities"] = [
                    {
                        "value": int(key) if answer["type"] == "score" else key,
                        "probability": value,
                    }
                    for key, value in answer["probabilities"].items()
                ]
            translated.append(answer)
        return {"model": "served-model", "answers": list(reversed(translated))}
    return {"model": "served-model", "answers": answers}


@pytest.mark.parametrize("provider,model,key", BACKENDS)
@pytest.mark.parametrize("task", TASKS, ids=["binary", "choice", "score", "multilabel"])
def test_hosted_contracts(monkeypatch, provider, model, key, task):
    requests = []
    monkeypatch.setenv(key, "test-secret")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")

    def post(url, headers, body):
        requests.append((url, headers, body))
        payload = response_for(task, openai=provider == "openai")
        return (
            {"success": True, "result": payload}
            if provider == "cloudflare"
            else payload
        )

    monkeypatch.setattr(decision_backends, "_post_json", post)
    backend = create_backend(
        BackendConfig(provider=provider, model=model), concurrency=2
    )
    validate_backend_for_task(backend, task)
    predictions = backend.evaluate_many(task, [_story(), _story("row-2")])
    assert [item.story_id for item in predictions] == ["row-1", "row-2"]
    prediction = predictions[0]
    assert prediction.resolved_model == "served-model"
    if isinstance(task, MulticlassCandidateSpec):
        assert prediction.choice == "billing"
        assert prediction.probabilities == {"billing": 0.8, "technical": 0.2}
        assert prediction.confidence == 0.6  # Preserve native confidence, not p_top.
        assert prediction_ambiguity(prediction) == pytest.approx(0.4)
    elif isinstance(task, ScoreCandidateSpec):
        assert prediction.score == 1.25
        assert prediction.score_probabilities == {0: 0.0, 1: 0.75, 2: 0.25}
        assert prediction.confidence == 0.7
    elif isinstance(task, MultilabelCandidateSpec):
        assert prediction.label_probabilities == {"billing": 0.9, "urgent": 0.4}
    else:
        assert prediction.probability == 0.63
    assert len(requests) == 2
    url, headers, body = requests[0]
    assert headers == {"Authorization": "Bearer test-secret"}
    if provider == "openai":
        assert url == "https://api.openai.com/v1/decisions"
        assert body["input"] == [
            {
                "role": "user",
                "content": [{"type": "input_json", "data": _story().state()}],
            }
        ]
        question = body["questions"][0]
        assert question["name"] == (
            "label_0" if isinstance(task, MultilabelCandidateSpec) else "alignment"
        )
        if isinstance(task, MulticlassCandidateSpec):
            assert question["choices"] == [
                {"value": "billing", "description": "Money"},
                {"value": "technical", "description": "Bugs"},
            ]
        elif isinstance(task, ScoreCandidateSpec):
            assert question["levels"] == [
                {"label": "0", "description": "Low"},
                {"label": "1", "description": "Medium"},
                {"label": "2", "description": "High"},
            ]
        else:
            assert question["type"] == "predicate"
            assert "True criteria:" in question["instructions"]
            assert "False criteria:" in question["instructions"]
    else:
        expected_url = {
            "perplexity": "https://api.perplexity.ai/v1/decisions",
            "openrouter": "https://openrouter.ai/api/alpha/decisions",
            "liquid": "https://api.liquid.ai/decisions/v1/systemone",
            "cloudflare": "https://api.cloudflare.com/client/v4/accounts/account/ai/run/"
            + model,
        }[provider]
        assert url == expected_url
        assert body["state"] == _story().state()
        assert body["model"] == (
            model.rsplit("/", 1)[-1] if provider == "cloudflare" else model
        )
        if isinstance(task, MultilabelCandidateSpec):
            assert body["questions"]["label_0"]["criteria"] == {
                "true": "Money",
                "false": "Other",
            }
        elif isinstance(task, CandidateSpec):
            assert body["questions"]["alignment"]["criteria"] == {
                "true": "Aviation",
                "false": "Other",
            }
        else:
            assert body["questions"]["alignment"]["criteria"] == (
                task.levels if isinstance(task, ScoreCandidateSpec) else task.criteria
            )


@pytest.mark.parametrize("provider,model,key", BACKENDS)
def test_credentials_discovery_and_empty_batches(provider, model, key):
    environment = {key: "secret"}
    if provider == "cloudflare":
        environment["CLOUDFLARE_ACCOUNT_ID"] = "account"
    config = BackendConfig(provider=provider, model=model)
    assert key in backend_credential_error(config, {})
    assert backend_credential_error(config, environment) is None
    assert [item[0] for item in available_backend_providers(environment)] == [provider]
    backend = create_backend(config, concurrency=1)
    assert backend.evaluate_many(TASKS[0], []) == []
    assert backend.evaluate_many_with_progress(TASKS[0], [], "empty") == []
    if provider != "cloudflare":
        assert default_backend_model(provider) == model


@pytest.mark.parametrize("model", ["clef", "clef-flash"])
def test_clef_short_model_names(model):
    backend = create_backend(
        BackendConfig(provider="cloudflare", model=model), concurrency=1
    )
    assert isinstance(backend, decision_backends.CloudflareClefEvaluator)
    assert "Clef" in backend.display_name


@pytest.mark.parametrize("provider,model,key", BACKENDS)
@pytest.mark.parametrize(
    "fault",
    ["missing", "wrong_type", "missing_confidence", "nan", "wrong_classes", "bad_sum"],
)
def test_rejects_malformed_predictions(monkeypatch, provider, model, key, fault):
    monkeypatch.setenv(key, "test-secret")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    task = TASKS[1]
    payload = response_for(task, openai=provider == "openai")
    answers = payload["answers"]
    answer = answers[0] if provider == "openai" else answers["alignment"]
    if fault == "missing":
        answers.clear()
    elif fault == "wrong_type":
        answer["type"] = "score"
    elif fault == "missing_confidence":
        del answer["confidence"]
    else:
        distribution = answer["probabilities"]
        if provider == "openai":
            if fault == "wrong_classes":
                distribution[0]["value"] = "unknown"
            else:
                distribution[0]["probability"] = float("nan") if fault == "nan" else 0.2
        elif fault == "wrong_classes":
            distribution["unknown"] = distribution.pop("billing")
        else:
            distribution["billing"] = float("nan") if fault == "nan" else 0.2
    wire = {"result": payload} if provider == "cloudflare" else payload
    monkeypatch.setattr(decision_backends, "_post_json", lambda *_args: wire)
    backend = create_backend(
        BackendConfig(provider=provider, model=model), concurrency=1
    )
    with pytest.raises(RuntimeError, match="row row-1") as error:
        backend.evaluate_many(task, [_story()])
    assert "test-secret" not in str(error.value)


@pytest.mark.parametrize(
    "fault", ["refusal", "duplicate", "missing_name", "duplicate_probability"]
)
def test_openai_special_response_errors(monkeypatch, fault):
    payload = response_for(TASKS[1], openai=True)
    answer = payload["answers"][0]
    if fault == "refusal":
        answer["type"] = "refusal"
    elif fault == "duplicate":
        payload["answers"].append(dict(answer))
    elif fault == "missing_name":
        del answer["name"]
    else:
        answer["probabilities"].append(dict(answer["probabilities"][0]))
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr(decision_backends, "_post_json", lambda *_args: payload)
    with pytest.raises(RuntimeError):
        decision_backends.OpenAIDecisionEvaluator("gpt-6-luna", 1).evaluate_many(
            TASKS[1], [_story()]
        )


@pytest.mark.parametrize(
    "provider,limit", [("perplexity", 128), ("openrouter", 128), ("openai", 400)]
)
def test_multilabel_request_limits(monkeypatch, provider, limit):
    task = MultilabelCandidateSpec(
        instructions="Apply tags",
        labels={
            str(index): MultilabelCriteria(true_criteria="Yes", false_criteria="No")
            for index in range(limit + 1)
        },
    )
    batches = []

    def post(_url, _headers, body):
        questions = body["questions"]
        batches.append(len(questions))
        if provider == "openai":
            return {
                "answers": [
                    {"name": question["name"], "type": "predicate", "probability": 0.7}
                    for question in reversed(questions)
                ]
            }
        return {"answers": {name: {"type": "noul", "noul": 0.7} for name in questions}}

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    monkeypatch.setattr(decision_backends, "_post_json", post)
    backend = create_backend(
        BackendConfig(provider=provider, model=default_backend_model(provider)),
        concurrency=1,
    )
    prediction = backend.evaluate_many(task, [_story()])[0]
    assert batches == [limit, 1]
    assert prediction.label_probabilities == {
        str(index): 0.7 for index in range(limit + 1)
    }


@pytest.mark.parametrize("model", ["@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"])
def test_clef_multilabel_request_limit(monkeypatch, model):
    task = MultilabelCandidateSpec(
        instructions="Apply tags",
        labels={
            str(index): MultilabelCriteria(true_criteria="Yes", false_criteria="No")
            for index in range(65)
        },
    )
    batches = []

    def post(_url, _headers, body):
        questions = body["questions"]
        assert len(questions) <= 64
        batches.append(list(questions))
        return {
            "success": True,
            "result": {
                "model": "served-model",
                "answers": {
                    name: {"type": "noul", "noul": int(name.split("_")[1]) / 100}
                    for name in questions
                },
            },
        }

    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "secret")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    monkeypatch.setattr(decision_backends, "_post_json", post)
    backend = create_backend(
        BackendConfig(provider="cloudflare", model=model), concurrency=1
    )
    prediction = backend.evaluate_many(task, [_story()])[0]
    assert batches == [[f"label_{index}" for index in range(64)], ["label_64"]]
    assert prediction.label_probabilities == {
        str(index): index / 100 for index in range(65)
    }
    assert prediction.resolved_model == "served-model"


@pytest.mark.parametrize("code", [400, 401, 403, 408, 429, 500, 502, 503, 504, 520])
def test_http_errors_do_not_expose_response_secrets(monkeypatch, code):
    monkeypatch.setattr(decision_http.time, "sleep", lambda _delay: None)

    calls = []

    def fail(*_args, **_kwargs):
        calls.append(1)
        raise HTTPError(
            "https://example.com", code, "secret", {}, io.BytesIO(b"secret-body")
        )

    monkeypatch.setattr(decision_http, "urlopen", fail)
    with pytest.raises(RuntimeError, match=f"HTTP {code}") as error:
        decision_http._post_json(
            "https://example.com", {"Authorization": "Bearer secret"}, {}
        )
    assert len(calls) == (4 if code in {408, 429, 500, 502, 503, 504, 520} else 1)
    assert "secret" not in str(error.value)
    if code == 429:
        assert "rate limit" in str(error.value)


@pytest.mark.parametrize(
    "failure", [TimeoutError(), URLError("unreachable"), ConnectionResetError()]
)
def test_http_connection_failures(monkeypatch, failure):
    monkeypatch.setattr(decision_http.time, "sleep", lambda _delay: None)
    calls = []

    def fail(*_args, **_kwargs):
        calls.append(1)
        raise failure

    monkeypatch.setattr(decision_http, "urlopen", fail)
    with pytest.raises(RuntimeError, match="after 4 attempts"):
        decision_http._post_json("https://example.com", {}, {})
    assert len(calls) == 4


@pytest.mark.parametrize("body", [b"broken json", b"[]"])
def test_http_invalid_json(monkeypatch, body):
    monkeypatch.setattr(
        decision_http, "urlopen", lambda *_args, **_kwargs: io.BytesIO(body)
    )
    with pytest.raises(RuntimeError, match="invalid"):
        decision_http._post_json("https://example.com", {}, {})


def test_http_wire_encoding(monkeypatch):
    requests = []

    def send(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(json.dumps({"answers": {}}).encode())

    monkeypatch.setattr(decision_http, "urlopen", send)
    decision_http._post_json(
        "https://example.com", {"Authorization": "Bearer secret"}, {"state": "東京"}
    )
    request, timeout = requests[0]
    assert request.method == "POST"
    assert request.headers["Content-type"] == "application/json"
    assert json.loads(request.data) == {"state": "東京"}
    assert timeout == 60


@pytest.mark.parametrize("provider,model,key", BACKENDS)
@pytest.mark.parametrize("task", TASKS, ids=["binary", "choice", "score", "multilabel"])
def test_saved_runtime_capture_cache_and_artifact_roundtrip(
    monkeypatch, tmp_path, provider, model, key, task
):
    from jev_align import AIFunction, Capture
    from jev_align.artifacts import (
        artifact_bytes,
        build_function_artifact,
        materialize_function_artifact,
    )
    from jev_align.models import CandidateHistory, LabelRecord, RunState
    from jev_align.persistence import RunStore
    from jev_align.session import ClimbSession

    requests = []
    monkeypatch.setenv(key, "runtime-secret")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")

    def post(url, _headers, body):
        requests.append(body)
        response = response_for(task, openai=provider == "openai")
        return {"result": response} if provider == "cloudflare" else response

    monkeypatch.setattr(decision_backends, "_post_json", post)
    config = BackendConfig(provider=provider, model=model)
    state = RunState(
        run_id="portable",
        source_path="missing.csv",
        source_sha256="unused",
        selected_columns=["title", "text"],
        backend=config,
        reflection_model="openai/reflection",
        current_candidate=task,
        history=[CandidateHistory(round_number=0, candidate=task, decision="seed")],
    )
    store = RunStore(tmp_path / "run")
    store.initialize(state)
    assert store.load_state().backend == config
    backend = create_backend(config, concurrency=1)
    session = ClimbSession(store.load_state(), [_story()], store, backend)
    original = session.current_pool_predictions()
    assert len(requests) == 1
    resumed = ClimbSession(store.load_state(), [_story()], store, backend)
    assert resumed.current_pool_predictions() == original
    assert len(requests) == 1
    resumed.state.backend = BackendConfig(provider=provider, model=model + "-different")
    assert resumed._load_pool_cache("current", task) is None
    resumed.state.backend = BackendConfig(provider="different-provider", model=model)
    assert resumed._load_pool_cache("current", task) is None

    with Capture(tmp_path / "captures", ambiguity_threshold=0) as recorder:
        with AIFunction.load(store.directory, capture=recorder) as function:
            prediction = function(title="Airport", text="Runway opens")
            assert prediction.model_dump(exclude={"story_id"}) == original[
                0
            ].model_dump(exclude={"story_id"})
    assert len(requests) == 2  # Capture makes no extra evaluation call.
    record = json.loads(recorder.path.read_text())
    assert record["definition"]["backend"] == config.model_dump()
    assert record["prediction"]["resolved_model"] == "served-model"
    assert record["input"] == _story().state()
    if isinstance(task, MulticlassCandidateSpec):
        label = "billing"
    elif isinstance(task, ScoreCandidateSpec):
        label = 1
    elif isinstance(task, MultilabelCandidateSpec):
        label = ["billing"]
    else:
        label = True
    store.append_label(
        LabelRecord(
            story_id="row-1",
            inputs=_story().state(),
            label=label,
            round_number=1,
            acquired_by="ambiguous",
            acquisition_probability=0.5,
        )
    )
    artifact = build_function_artifact(store.directory, name="Portable decision")
    assert artifact.backend == config
    assert b"runtime-secret" not in artifact_bytes(artifact)
    target = tmp_path / "pulled"
    materialize_function_artifact(
        artifact,
        target,
        reference="tester/portable-decision",
        version=1,
        digest="test-digest",
        registry="https://example.com",
    )
    assert RunStore(target).load_state().backend == config
    with AIFunction.load(target) as pulled:
        assert (
            pulled(title="Airport", text="Runway opens").resolved_model
            == "served-model"
        )
    assert len(requests) == 3


@pytest.mark.parametrize(
    "provider,model,key",
    BACKENDS
    + [
        ("cloudflare", "typesafe/jev", "CLOUDFLARE_API_TOKEN"),
        ("vercel", "typesafe-ai/jev", "AI_GATEWAY_API_KEY"),
    ],
)
@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 520])
def test_all_http_backends_retry_transient_http_then_succeed(
    monkeypatch, provider, model, key, status
):
    monkeypatch.setenv(key, "private-token")
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "account")
    requests = []
    delays = []
    payload = response_for(TASKS[0], openai=provider == "openai")
    if provider == "cloudflare":
        payload = {"result": payload}
    elif provider == "vercel":
        payload["answers"]["alignment"] = {"type": "boolean", "probability": 0.63}

    def send(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise HTTPError(
                request.full_url,
                status,
                "Too many requests",
                {"Retry-After": "2"},
                io.BytesIO(b"private body"),
            )
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(decision_http, "urlopen", send)
    monkeypatch.setattr(decision_http.time, "sleep", delays.append)
    backend = create_backend(
        BackendConfig(provider=provider, model=model), concurrency=1
    )
    prediction = backend.evaluate_many(TASKS[0], [_story()])[0]
    assert prediction.probability == 0.63
    assert len(requests) == 2
    assert delays == [2]
    assert requests[0].data == requests[1].data


@pytest.mark.parametrize("status", [429, 503, 520])
def test_http_transient_backoff_and_exhaustion(monkeypatch, status):
    from jev_align import retries

    requests = []
    delays = []
    failures = []

    def send(request, timeout):
        requests.append(request)
        error = HTTPError(
            request.full_url, status, "private", {}, io.BytesIO(b"private body")
        )
        failures.append(error)
        raise error

    monkeypatch.setattr(decision_http, "urlopen", send)
    monkeypatch.setattr(decision_http.time, "sleep", delays.append)
    monkeypatch.setattr(retries.random, "uniform", lambda *_args: 0)
    with pytest.raises(RuntimeError, match="after 4 attempts") as caught:
        decision_http._post_json("https://example.com", {}, {})
    assert len(requests) == 4
    assert delays == [0.5, 1, 2]
    assert all(error.fp.closed for error in failures)
    assert "private" not in str(caught.value)


def test_http_does_not_retry_early_for_long_retry_after(monkeypatch):
    requests = []

    def send(request, timeout):
        requests.append(request)
        raise HTTPError(
            request.full_url, 429, "rate limited", {"Retry-After": "3600"}, io.BytesIO()
        )

    monkeypatch.setattr(decision_http, "urlopen", send)
    with pytest.raises(RuntimeError, match="rate limit"):
        decision_http._post_json("https://example.com", {}, {})
    assert len(requests) == 1


def test_multilabel_retries_only_the_rate_limited_chunk(monkeypatch):
    monkeypatch.setenv("PERPLEXITY_API_KEY", "private-token")
    task = MultilabelCandidateSpec(
        instructions="Apply tags",
        labels={
            str(index): MultilabelCriteria(true_criteria="Yes", false_criteria="No")
            for index in range(129)
        },
    )
    batches = []

    def send(request, timeout):
        questions = json.loads(request.data)["questions"]
        batches.append(list(questions))
        if len(batches) == 2:
            raise HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "0"},
                io.BytesIO(),
            )
        return io.BytesIO(
            json.dumps(
                {"answers": {name: {"type": "noul", "noul": 0.7} for name in questions}}
            ).encode()
        )

    monkeypatch.setattr(decision_http, "urlopen", send)
    monkeypatch.setattr(decision_http.time, "sleep", lambda _delay: None)
    prediction = decision_backends.PerplexityDeciderEvaluator(
        "pplx-decider-v1.2", 1
    ).evaluate_many(task, [_story()])[0]
    assert [len(batch) for batch in batches] == [128, 1, 1]
    assert batches[1] == batches[2] == ["label_128"]
    assert len(prediction.label_probabilities) == 129


@pytest.mark.parametrize("recover", [True, False])
def test_gepa_openai_evaluation_handles_503_without_fabricating_scores(
    monkeypatch, recover
):
    from jev_align.optimizer import F1BatchEvaluator

    monkeypatch.setenv("OPENAI_API_KEY", "private-token")
    monkeypatch.setattr(decision_http.time, "sleep", lambda _delay: None)
    requests = []

    def send(request, timeout):
        requests.append(request)
        if not recover or len(requests) == 1:
            raise HTTPError(
                request.full_url, 503, "Service unavailable", {}, io.BytesIO()
            )
        return io.BytesIO(json.dumps(response_for(TASKS[0], openai=True)).encode())

    monkeypatch.setattr(decision_http, "urlopen", send)
    story = _story("row-000896")
    backend = create_backend(
        BackendConfig(provider="openai", model="gpt-6-luna"), concurrency=1
    )
    evaluate = F1BatchEvaluator(backend, {story.id: story}, TASKS[0])
    pairs = [(TASKS[0].as_gepa(), {"story_id": story.id, "label": True})]
    if recover:
        result = evaluate(pairs)
        assert result[0][0] == 1.0
        assert result[0][1]["binary_probability"] == 0.63
        assert len(requests) == 2
    else:
        with pytest.raises(
            RuntimeError, match="row row-000896:.*HTTP 503.*after 4 attempts"
        ):
            evaluate(pairs)
        assert len(requests) == 4


@pytest.mark.parametrize(
    "failure", [TimeoutError(), URLError("unreachable"), ConnectionResetError()]
)
def test_http_connection_failure_then_success(monkeypatch, failure):
    requests = []

    def send(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            raise failure
        return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(decision_http, "urlopen", send)
    monkeypatch.setattr(decision_http.time, "sleep", lambda _delay: None)
    assert decision_http._post_json("https://example.com", {}, {}) == {"ok": True}
    assert len(requests) == 2


@pytest.mark.parametrize(
    "model",
    [
        "typesafe/jev-1.14",
        "typesafe/jev-1.13",
        "perplexity/pplx-decider-v1.2",
        "openai/gpt-6-luna-decisions",
        "liquid/d1.1",
        "cloudflare/clef",
        "cloudflare/clef-flash",
        "liquid/d1-ultra",
    ],
)
def test_openrouter_sends_selected_model_without_direct_provider_keys(
    monkeypatch, model
):
    for key in (
        "OPENAI_API_KEY",
        "PERPLEXITY_API_KEY",
        "LIQUID_API_KEY",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_API_TOKEN",
        "TYPESAFE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-secret")
    requests = []

    def post(url, headers, body):
        requests.append((url, headers, body))
        return response_for(TASKS[0])

    monkeypatch.setattr(decision_backends, "_post_json", post)
    backend = create_backend(
        BackendConfig(provider="openrouter", model=model), concurrency=1
    )
    assert backend.evaluate_many(TASKS[0], [_story()])[0].probability == 0.63
    assert requests[0][0] == "https://openrouter.ai/api/alpha/decisions"
    assert requests[0][1] == {"Authorization": "Bearer router-secret"}
    assert requests[0][2]["model"] == model
    assert requests[0][2]["questions"]["alignment"]["type"] == "noul"


def test_openrouter_requires_its_own_credentials():
    error = backend_credential_error(
        BackendConfig(provider="openrouter", model="cloudflare/clef"),
        {"OPENAI_API_KEY": "unrelated"},
    )
    assert "OPENROUTER_API_KEY" in error
