# coevolution/core/interfaces/initializer.py
"""
Population initializer protocol and weighted composite — separated from breeding.
"""

from __future__ import annotations

import math
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

    def initialize(self, problem: Problem, size: int | None = None) -> list[T]:
        """
        Create the initial population for a given problem.

        Args:
            problem: Problem context (description, starter code, test cases).
            size: Optional override for how many individuals to produce.
                  When None, each concrete initializer uses its own
                  ``pop_config.initial_population_size``.
                  WeightedPopulationInitializer always passes an explicit
                  integer to its children.

        Returns:
            List of initial individuals (Generation 0).
            May return [] for populations that start empty (e.g. differential).
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

    Slot allocation
    ---------------
    Given a total budget *N* and *k* sub-initializers with weights
    ``w_0 … w_{k-1}``, each sub-initializer receives::

        slot_i = round(N * w_i / sum(weights))

    Integer rounding means the slots might not sum to exactly *N*.  The
    last slot is adjusted to absorb the rounding remainder so the total is
    always precise.

    Usage::

        WeightedPopulationInitializer(
            registered_initializers=[
                RegisteredInitializer(weight=1.0, initializer=DirectCodeInitializer(...)),
            ],
            pop_config=population_config,
        )
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

    # ------------------------------------------------------------------
    # IPopulationInitializer
    # ------------------------------------------------------------------

    def initialize(self, problem: Problem, size: int | None = None) -> list[T]:
        """
        Distribute the size budget across sub-initializers and concatenate results.

        Args:
            problem: The problem context forwarded to each sub-initializer.
            size: Total number of individuals to produce.  When ``None``,
                  falls back to ``pop_config.initial_population_size``.

        Returns:
            Concatenated list of individuals in registration order.
        """
        total = size if size is not None else self._pop_config.initial_population_size

        slots = self._compute_slots(total)

        logger.debug(
            f"WeightedPopulationInitializer: total={total}, "
            f"slot allocation="
            + ", ".join(
                f"{ri.initializer.__class__.__name__}={s}"
                for ri, s in zip(self._registered, slots)
            )
        )

        individuals: list[T] = []
        for ri, slot in zip(self._registered, slots):
            if slot <= 0:
                logger.debug(
                    f"WeightedPopulationInitializer: skipping "
                    f"{ri.initializer.__class__.__name__} (slot=0)"
                )
                continue
            result = ri.initializer.initialize(problem, size=slot)
            individuals.extend(result)

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
