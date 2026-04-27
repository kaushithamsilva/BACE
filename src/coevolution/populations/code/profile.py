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
from coevolution.populations.registries import (
    initializer_registry,
    operator_registry,
)
from coevolution.core.interfaces.language import ILanguage
from infrastructure.llm_client import LLMClient

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

# Operators/Initializers imported here to ensure decorators are run
from .operators.mutation import CodeMutationOperator  # noqa: F401
from .operators.crossover import CodeCrossoverOperator  # noqa: F401
from .operators.edit import CodeGenericEditOperator  # noqa: F401
from .operators.initializer import (
    PlanningCodeInitializer,  # noqa: F401
    StandardCodeInitializer,  # noqa: F401
)


from ..registries import registry


@registry.code_factory("default")
def create_default_code_profile(
    llm_client: LLMClient,
    language_adapter: ILanguage,
    # ... (rest of parameters)
    initial_prior: float = 0.2,
    initial_population_size: int = 10,
    max_population_size: int = 15,
    offspring_rate: float = 0.8,
    elitism_rate: float = 0.2,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.2,
    generic_edit_rate: float = 0.6,
    init_pop_batch_size: int = 2,
    diversity_enabled: bool = True,
    prob_assigner_strategy: str = "min",
    k_failing_tests: int = 10,
    **factory_config: Any,
) -> CodeProfile:
    """Create a standard code population profile."""
    # ... (function body)
    # Component initialization delegated to registry

    total_rate = mutation_rate + crossover_rate + generic_edit_rate
    if not (0.99 <= total_rate <= 1.01):
        raise ValueError(
            f"Operation rates must sum to 1.0, got {total_rate:.4f} "
            f"(mutation={mutation_rate}, crossover={crossover_rate}, generic_edit={generic_edit_rate})"
        )

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
        llm=llm_client,
        parser=language_adapter.parser,
        language_name=language_adapter.language,
        parent_selector=parent_selector,
        prob_assigner=prob_assigner,
        failing_test_selector=FailingTestSelector,
        k_failing_tests=k_failing_tests,
        # Pass explicit rates if they are not in factory_config
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        generic_edit_rate=generic_edit_rate,
    )

    initializer: IPopulationInitializer[CodeIndividual] = (
        initializer_registry.build_weighted_initializer(
            population="code",
            config=factory_config,
            llm=llm_client,
            parser=language_adapter.parser,
            language_name=language_adapter.language,
            pop_config=population_config,
            init_pop_batch_size=init_pop_batch_size,
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
