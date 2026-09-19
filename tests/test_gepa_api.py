from __future__ import annotations

from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig
from gepa.strategies.proposal_sampling import PxNSampling
from gepa.strategies.proposal_selection import AllImprovements
from gepa.utils import ScoreThresholdStopper

from jev_align.optimizer import ClassAwareBatchSampler


def test_installed_gepa_supports_planned_configuration() -> None:
    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=300,
            sampling_strategy=PxNSampling(p=2, n=2),
            selection_strategy=AllImprovements(),
            frontier_type="cartesian",
            cache_evaluation=False,
        ),
        reflection=ReflectionConfig(
            reflection_lm="openai/test",
            batch_sampler=ClassAwareBatchSampler(),
            module_selector="all",
            skip_perfect_score=True,
            perfect_score=1.0,
        ),
        stop_callbacks=ScoreThresholdStopper(1.0),
    )
    assert config.engine.max_metric_calls == 300
    assert isinstance(config.engine.sampling_strategy, PxNSampling)
    assert config.engine.frontier_type == "cartesian"
    assert config.engine.cache_evaluation is False
    assert isinstance(config.reflection.batch_sampler, ClassAwareBatchSampler)
    assert config.reflection.skip_perfect_score is True
    assert config.reflection.perfect_score == 1.0
    assert isinstance(config.stop_callbacks, ScoreThresholdStopper)
