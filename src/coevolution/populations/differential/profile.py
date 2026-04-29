"""Differential population profile factory."""

from __future__ import annotations
from typing import Any

from coevolution.core.individual import TestIndividual
from coevolution.core.interfaces import (
    BayesianConfig,
    IPopulationInitializer,
    PopulationConfig,
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
from infrastructure.sandbox.types import SandboxConfig

from ..registries import profile_registry
# Ensure decorators are run by importing the packages
from . import initializers, operators  # noqa: F401
from .helpers.finder import DifferentialFinder
from .helpers.llm_service import DifferentialLLMService
from .helpers.selector import FunctionallyEqSelector


@profile_registry.test_factory("differential")
def create_differential_test_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    sandbox_config: SandboxConfig,
    # ... (parameters)
    initial_prior: float = 0.5,
    initial_population_size: int = 0,
    max_population_size: int = 20,
    offspring_rate: float = 1.0,
    elitism_rate: float = 0.3,
    discovery_rate: float = 1.0,
    alpha: float = 0.05,
    beta: float = 0.3,
    gamma: float = 0.3,
    learning_rate: float = 0.05,
    cpu_workers: int = 4,
    prob_assigner_strategy: str = "min",
    diversity_enabled: bool = True,
    max_pairs_per_group: int = 5,
    num_passing_tests_to_sample: int = 5,
    **factory_config: Any,
) -> TestProfile:
    """Create a differential test population profile."""
    # Split cpu_workers budget across two parallelism levels so total OS
    # processes never exceed the user-configured COEVOLUTION_WORKERS value:
    #   Level-1  pair_workers:     how many pairs run find_differential() concurrently
    #   Level-2  workers_per_pair: processes in each pair's inner multiprocessing.Pool
    # With cpu_workers=16  → pair_workers=4,  workers_per_pair=4,  total=16
    # With cpu_workers=8   → pair_workers=2,  workers_per_pair=4,  total=8
    # With cpu_workers=4   → pair_workers=1,  workers_per_pair=4,  total=4
    pair_workers = max(1, cpu_workers // 4)
    workers_per_pair = max(1, cpu_workers // pair_workers)

    population_config = PopulationConfig(
        initial_prior=initial_prior,
        initial_population_size=initial_population_size,
        max_population_size=max_population_size,
        offspring_rate=offspring_rate,
        elitism_rate=elitism_rate,
        diversity_selection=diversity_enabled,
    )

    prob_assigner = ProbabilityAssigner(
        strategy=prob_assigner_strategy, initial_prior=initial_prior
    )
    parent_selector: RouletteWheelParentSelection[TestIndividual] = (
        RouletteWheelParentSelection()
    )

    llm_service = DifferentialLLMService(
        llm=llm_client,
        parser=language_adapter.parser,
        composer=language_adapter.composer,
        language_name=language_adapter.language,
    )

    differential_finder = DifferentialFinder(
        parser=language_adapter.parser,
        composer=language_adapter.composer,
        runtime=language_adapter.runtime,
        sandbox_config=sandbox_config,
        enable_multiprocessing=workers_per_pair > 1,
        cpu_workers=workers_per_pair,
    )

    breeder: Breeder[TestIndividual] = operator_registry.build_weighted_breeder(
        population="differential",
        config=factory_config,
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
        llm_service=llm_service,
        differential_finder=differential_finder,
        func_eq_selector=FunctionallyEqSelector(),
        max_pairs_per_group=max_pairs_per_group,
        num_passing_tests_to_sample=num_passing_tests_to_sample,
        llm_workers=1,  # Sequential execution: DiscoveryOperator handles internal parallelism
        pair_workers=pair_workers,
        # Default rates if not in factory_config
        discovery_rate=discovery_rate,
    )

    initializer: IPopulationInitializer[TestIndividual] = (
        initializer_registry.build_weighted_initializer(
            population="differential",
            config=factory_config,
            llm=llm_client,
            parser=language_adapter.parser,
            language_name=language_adapter.language,
            pop_config=population_config,
        )
    )

    elite_selector: TestDiversityEliteSelector[TestIndividual] = (
        TestDiversityEliteSelector(test_population_key="differential")
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


__all__ = ["create_differential_test_profile"]
