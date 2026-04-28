"""Unittest population profile factories."""

from __future__ import annotations
from typing import Any

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    BayesianConfig,
    IPopulationInitializer,
    PopulationConfig,
    PublicTestProfile,
    TestProfile,
)
from coevolution.populations.registries import (
    initializer_registry,
    operator_registry,
)
from coevolution.core.interfaces.language import ILanguage
from coevolution.strategies.breeding.breeder import Breeder
from coevolution.strategies.probability.assigner import ProbabilityAssigner
from coevolution.strategies.selection.elite import TestDiversityEliteSelector
from coevolution.strategies.selection.parent_selection import (
    RouletteWheelParentSelection,
)
from infrastructure.llm_client import LLMClient

from ..registries import profile_registry
# Operators/Initializers imported here to ensure decorators are run
from .operators.crossover import UnittestCrossoverOperator  # noqa: F401
from .operators.repair import UnittestRepairOperator  # noqa: F401
from .initializers import UnittestInitializer  # noqa: F401
from .operators.mutation import UnittestMutationOperator  # noqa: F401


@profile_registry.test_factory("unittest")
def create_unittest_test_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    # ... (rest of parameters)
    initial_prior: float = 0.2,
    initial_population_size: int = 20,
    max_population_size: int = 20,
    elitism_rate: float = 0.4,
    offspring_rate: float = 0.8,
    alpha: float = 0.01,
    beta: float = 0.3,
    gamma: float = 0.3,
    learning_rate: float = 0.05,
    prob_assigner_strategy: str = "min",
    diversity_enabled: bool = True,
    **factory_config: Any,
) -> TestProfile:
    """Create a unittest test population profile."""
    # ... (function body)
    # Configuration delegated to registries

    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        elitism_rate=elitism_rate,
        offspring_rate=offspring_rate,
        diversity_selection=diversity_enabled,
    )

    parent_selector: RouletteWheelParentSelection[TestIndividual] = (
        RouletteWheelParentSelection()
    )
    prob_assigner = ProbabilityAssigner(
        strategy=prob_assigner_strategy, initial_prior=initial_prior
    )

    breeder: Breeder[TestIndividual] = operator_registry.build_weighted_breeder(
        population="unittest",
        config=factory_config,
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
    )

    initializer: IPopulationInitializer[TestIndividual] = (
        initializer_registry.build_weighted_initializer(
            population="unittest",
            config=factory_config,
            llm=llm_client,
            parser=language_adapter.parser,
            language_name=language_adapter.language,
            pop_config=population_config,
        )
    )

    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="unittest")
    )

    bayesian_config = BayesianConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        learning_rate=learning_rate,
    )

    return TestProfile(
        population_config=population_config,
        breeder=breeder,
        initializer=initializer,
        elite_selector=elite_selector,
        bayesian_config=bayesian_config,
    )


@profile_registry.public_factory("public")
def create_public_test_profile(
    alpha: float = 0.001,
    beta: float = 0.1,
    gamma: float = 0.1,
    learning_rate: float = 0.05,
) -> PublicTestProfile:
    """Create a public/ground-truth test profile (fixed tests, no evolution)."""
    return PublicTestProfile(
        bayesian_config=BayesianConfig(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            learning_rate=learning_rate,
        )
    )


__all__ = ["create_unittest_test_profile", "create_public_test_profile"]
