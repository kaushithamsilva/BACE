# coevolution/core/interfaces/initializer.py
"""
Population initializer protocol and weighted composite — separated from breeding.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from loguru import logger

from .base import BaseIndividual
from .data import Problem

if TYPE_CHECKING:
    from .config import PopulationConfig


class IPopulationInitializer[T: BaseIndividual](Protocol):
    """
    Protocol for creating Generation 0 individuals.

    Separated from IBreedingStrategy so that initialization logic
    (batching, planning mode, retries) can evolve independently from
    the per-generation breeding loop.
    """

    @property
    def yields_per_call(self) -> int:
        """
        The number of individuals typically produced in a single logical request.
        For LLM-based code generators, this is usually 1.
        For unittests or brainstormers, it might be 20 or 50.
        """
        return 1

    def initialize(self, problem: Problem, size: int | None = None) -> list[T]:
        """
        Create the initial individuals for a given problem.

        Args:
            problem: Problem context (description, starter code, test cases).
            size: The number of individuals to produce in this call.
                  The worker should aim to fill this quota efficiently
                  (e.g. using one big LLM request if possible).

        Returns:
            List of initial individuals (Generation 0).
        """
        ...


@dataclass
class RegisteredInitializer[T: BaseIndividual]:
    """
    Pairs a population initializer with its sampling weight.

    The weight is a non-negative float expressing the *relative* share of the
    total initial population that this sub-initializer is responsible for.
    Weights do not need to sum to 1.0; ``WeightedPopulationInitializer``
    normalises them internally.

        Example::

        RegisteredInitializer(weight=1.0, initializer=DirectCodeInitializer(...))
    """

    weight: float
    initializer: IPopulationInitializer[T]


class WeightedPopulationInitializer[T: BaseIndividual]:
    """
    Composite initializer that distributes a population-size budget across
    multiple sub-initializers according to their weights.

    This mirrors the ``Breeder`` / ``RegisteredOperator`` pattern used for
    breeding operators, giving profile factories a uniform, extensible way to
    compose initialisation strategies.

    Concurrency Orchestration
    -------------------------
    This orchestrator centralizes all initialization concurrency. It calculates 
    the necessary number of parallel workers for each strategy based on its
    ``yields_per_call`` hint, then schedules them in a single thread pool.
    This ensures that workers remain simple and stateless, while the 
    framework handles optimal parallelization of LLM calls.
    """

    def __init__(
        self,
        registered_initializers: list[RegisteredInitializer[T]],
        pop_config: "PopulationConfig",
    ) -> None:
        if not registered_initializers:
            raise ValueError(
                "WeightedPopulationInitializer requires at least one RegisteredInitializer."
            )
        for ri in registered_initializers:
            if ri.weight < 0:
                raise ValueError(
                    f"Initializer weight must be non-negative, got {ri.weight} "
                    f"for {ri.initializer.__class__.__name__}."
                )

        self._registered = registered_initializers
        self._pop_config = pop_config

    @property
    def yields_per_call(self) -> int:
        """The composite orchestrator fills the entire requested size in one go."""
        return self._pop_config.initial_population_size

    # ------------------------------------------------------------------
    # IPopulationInitializer
    # ------------------------------------------------------------------

    def initialize(self, problem: Problem, size: int | None = None) -> list[T]:
        """
        Distribute the size budget across sub-initializers and concatenate results.
        """
        if size is None:
            size = self._pop_config.initial_population_size

        if size <= 0:
            return []

        slots = self._compute_slots(size)

        logger.debug(
            f"WeightedPopulationInitializer: size={size}, "
            f"allocation="
            + ", ".join(
                f"{ri.initializer.__class__.__name__}={s}"
                for ri, s in zip(self._registered, slots)
            )
        )

        individuals: list[T] = []

        # Generate a list of "Parallel Call Tasks"
        # We determine how many parallel LLM calls are needed to fulfill each slot quota.
        tasks: list[tuple[IPopulationInitializer[T], int]] = []
        for ri, slot in zip(self._registered, slots):
            if slot <= 0:
                continue

            yields = ri.initializer.yields_per_call
            num_calls = math.ceil(slot / yields)
            
            # Each call will be responsible for a portion of the slot.
            per_call = math.ceil(slot / num_calls)
            
            for i in range(num_calls):
                # Ensure the last call doesn't overshoot the slot (aesthetic only,
                # as the worker often returns a fixed batch size anyway)
                current_quota = min(per_call, slot - (i * per_call))
                if current_quota > 0:
                    tasks.append((ri.initializer, current_quota))

        if not tasks:
            return []

        # Centralized task execution
        with ThreadPoolExecutor(max_workers=min(len(tasks), 32)) as executor:
            future_to_task = {
                executor.submit(init.initialize, problem, size=quota): (init, quota)
                for init, quota in tasks
            }

            for future in as_completed(future_to_task):
                init, quota = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        # Enforce initial prior and other orchestrator-level defaults
                        for ind in result:
                            if ind.probability == 0: # Only set if worker didn't set it
                                ind.probability = self._pop_config.initial_prior
                        individuals.extend(result)
                except Exception as e:
                    logger.error(
                        f"WeightedPopulationInitializer: {init.__class__.__name__} task failed: {e}"
                    )

        # Truncate if over-generated (common with batch-yield workers)
        if len(individuals) > size:
            individuals = individuals[:size]

        return individuals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_slots(self, total: int) -> list[int]:
        """Proportionally allocate *total* across registered initializers."""
        weights = [ri.weight for ri in self._registered]
        weight_sum = sum(weights)

        if weight_sum == 0:
            raise ValueError(
                "WeightedPopulationInitializer: all weights are zero — "
                "cannot allocate any individuals."
            )

        # Initial proportional allocation (may not sum to `total` due to rounding)
        raw_slots = [total * w / weight_sum for w in weights]
        slots = [math.floor(s) for s in raw_slots]

        # Distribute the rounding remainder to the largest fractional parts
        remainder = total - sum(slots)
        if remainder > 0:
            # Sort indices by descending fractional part to distribute remainder fairly
            fractional_parts = [(raw_slots[i] - slots[i], i) for i in range(len(slots))]
            fractional_parts.sort(reverse=True)
            for _, idx in fractional_parts[:remainder]:
                slots[idx] += 1

        return slots


__all__ = [
    "IPopulationInitializer",
    "RegisteredInitializer",
    "WeightedPopulationInitializer",
]
