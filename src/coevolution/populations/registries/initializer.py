"""Registry for population initializers.

Allows dynamic discovery and weighted instantiation based on YAML configuration.
"""

from __future__ import annotations
from typing import Any, Dict, List

from coevolution.core.interfaces.initializer import (
    IPopulationInitializer,
    RegisteredInitializer,
    WeightedPopulationInitializer,
)
from .base import DIRegistry


class InitializerRegistry(DIRegistry[IPopulationInitializer[Any]]):
    """Registry for population initializers with dependency injection."""

    def build_weighted_initializer(
        self,
        population: str,
        config: Dict[str, Any],
        **dependencies: Any,
    ) -> IPopulationInitializer[Any]:
        """Build a WeightedPopulationInitializer from YAML configuration.

        Args:
            population: The population type (e.g. 'code').
            config: Dict of YAML configuration keys (e.g. {standard_init_rate: 0.5}).
            dependencies: Common dependencies (llm, parser, pop_config, etc.)

        Returns:
            A WeightedPopulationInitializer instance.
        """
        registered_classes = self.get_all(population)
        if not registered_classes:
             # Backward compatibility: if nothing registered, this population
             # might not support initializers yet.
             return None # type: ignore

        # 1. Map registered names to their configured weights
        weights: Dict[str, float] = {}
        for name in registered_classes:
            # Convension: '{name}_init_rate'
            weight = config.get(f"{name}_init_rate", 0.0)
            if weight > 0:
                weights[name] = weight

        if not weights:
            # Backward compatibility fallback: use first registered initializer
            # if no weights are specified in the config.
            first_name = list(registered_classes.keys())[0]
            weights[first_name] = 1.0

        # 2. Instantiate weighted initializers
        context = {**dependencies, **config}
        registered_inits: List[RegisteredInitializer[Any]] = []

        for name, weight in weights.items():
            cls = registered_classes[name]
            instance = self._instantiate(cls, context)
            registered_inits.append(RegisteredInitializer(weight, instance))

        # Check for pop_config in dependencies
        pop_config = dependencies.get("pop_config")
        if not pop_config:
            raise ValueError("InitializerRegistry requires 'pop_config' in dependencies.")

        return WeightedPopulationInitializer(registered_inits, pop_config)


# Global registry instance
initializer_registry = InitializerRegistry()
