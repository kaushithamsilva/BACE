"""Public population profile factories."""

from __future__ import annotations
from typing import Any

from coevolution.core.interfaces import BayesianConfig, PublicTestProfile
from coevolution.populations.registries import profile_registry
from coevolution.core.interfaces.language import ILanguage
from coevolution.strategies.llm_base import ILanguageModel


@profile_registry.public_factory("public")
def create_public_test_profile(
    llm_client: ILanguageModel,
    language_adapter: ILanguage,
    alpha: float = 0.001,
    beta: float = 0.1,
    gamma: float = 0.1,
    learning_rate: float = 0.05,
    **factory_config: Any,
) -> PublicTestProfile:
    """Create a public/ground-truth test profile (fixed tests, no evolution)."""
    
    return PublicTestProfile(
        bayesian_config=BayesianConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            learning_rate=learning_rate,
        ),
    )


__all__ = ["create_public_test_profile"]
