"""Public population profile factories."""

from __future__ import annotations
from typing import Any

from coevolution.core.interfaces import BayesianConfig, PublicTestProfile
from coevolution.populations.registries import profile_registry, operator_registry
from coevolution.core.interfaces.language import ILanguage
from coevolution.strategies.llm_base import ILanguageModel
from coevolution.core.interfaces.operators import RegisteredOperator
from coevolution.strategies.selection.failing_test_selection import FailingTestSelector

from .operators.code_repair import PublicCodeRepairOperator


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
    
    # Instantiate the specialized repair operator
    # Note: public profile doesn't have its own breeder/selector, so we use common ones
    public_code_repair = RegisteredOperator(
        weight=0.0,
        operator=operator_registry._instantiate(
            PublicCodeRepairOperator,
            {
                "llm": llm_client,
                "parser": language_adapter.parser,
                "language_name": language_adapter.language,
                "failing_test_selector": FailingTestSelector,
                **factory_config,
            },
        ),
    )

    return PublicTestProfile(
        bayesian_config=BayesianConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            learning_rate=learning_rate,
        ),
        repair_operators=(public_code_repair,),
    )


__all__ = ["create_public_test_profile"]
