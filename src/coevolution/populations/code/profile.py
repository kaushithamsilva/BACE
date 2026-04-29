"""Code population profile factory."""

from __future__ import annotations

from typing import Any

from coevolution.core.individual import CodeIndividual
from coevolution.core.interfaces import (
    CodeProfile,
    IEliteSelectionStrategy,
    IPopulationInitializer,
    PopulationConfig,
)
from coevolution.core.interfaces.language import ILanguage
from coevolution.populations.registries import initializer_registry, operator_registry
from coevolution.strategies.breeding.breeder import Breeder
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.elite import (
    CodeDiversityEliteSelector,
    TopKEliteSelector,
)
from coevolution.strategies.selection.failing_test_selection import FailingTestSelector
from coevolution.strategies.selection.parent_selection import (
    RouletteWheelParentSelection,
)
from infrastructure.llm_client import LLMClient

from ..registries import profile_registry
# Ensure decorators are run by importing the packages
from . import initializers, operators  # noqa: F401


@profile_registry.code_factory("default")
def create_default_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    initial_prior: float = 0.2,
    initial_population_size: int = 10,
    max_population_size: int = 15,
    offspring_rate: float = 0.8,
    elitism_rate: float = 0.2,
    diversity_enabled: bool = True,
    prob_assigner_strategy: str = "min",
    injected_operators: dict[str, Any] | None = None,
    **factory_config: Any,
) -> CodeProfile:
    """Create a standard code population profile."""
    # Configuration delegated to registries

    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        offspring_rate=offspring_rate,
        elitism_rate=elitism_rate,
        diversity_selection=diversity_enabled,
    )

    parent_selector: RouletteWheelParentSelection[CodeIndividual] = (
        RouletteWheelParentSelection()
    )
    prob_assigner = ProbabilityAssigner(
        strategy=prob_assigner_strategy, initial_prior=initial_prior
    )

    breeder: Breeder[CodeIndividual] = operator_registry.build_weighted_breeder(
        population="code",
        config=factory_config,
        injected_operators=injected_operators,
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
        failing_test_selector=FailingTestSelector,
    )

    initializer: IPopulationInitializer[CodeIndividual] = (
        initializer_registry.build_weighted_initializer(
            population="code",
            config=factory_config,
            llm=llm_client,
            parser=language_adapter.parser,
            language_name=language_adapter.language,
            pop_config=population_config,
        )
    )

    elite_selector: IEliteSelectionStrategy[CodeIndividual] = (
        CodeDiversityEliteSelector() if diversity_enabled else TopKEliteSelector()
    )

    return CodeProfile(
        population_config=population_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
    )


__all__ = ["create_default_code_profile"]
