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

    def _extract_rates(
        self, population: str, config: Dict[str, Any], section_name: str = "init_rates"
    ) -> Dict[str, float]:
        """Extract and validate initialization rates from a named config section."""
        rates = config.get(section_name)
        if rates is None:
            # Fallback for populations without initializers or old configs
            # If nothing is in the section, we return an empty dict
            # The caller will handle empty weights.
            return {}

        registered_names = self.list_names(population)
        
        # 1. Validate registration
        for name in rates.keys():
            if name not in registered_names:
                raise ValueError(
                    f"Initializer '{name}' listed in '{section_name}' is not registered for "
                    f"population '{population}'. Registered: {registered_names}"
                )

        # 2. Extract weights
        weights = {name: float(weight) for name, weight in rates.items() if weight > 0}
        
        if not weights:
            return {}

        # 3. Strict Sum-to-1.0 Validation
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Initializer weights in '{section_name}' for '{population}' must sum to 1.0, "
                f"got {total:.4f} (from {weights})"
            )

        return weights

    def build_weighted_initializer(
        self,
        population: str,
        config: Dict[str, Any],
        **dependencies: Any,
    ) -> IPopulationInitializer[Any]:
        """Build a WeightedPopulationInitializer from YAML configuration.

        Args:
            population: The population type (e.g. 'code').
            config: Dict of YAML configuration keys (must contain 'init_rates').
            dependencies: Common dependencies (llm, parser, pop_config, etc.)

        Returns:
            A WeightedPopulationInitializer instance.
        """
        # 1. Extract and validate rates
        weights = self._extract_rates(population, config)

        registered_classes = self.get_all(population)
        if not registered_classes:
             return None # type: ignore

        if not weights:
             # If no weights, but classes exist, and it's not generation 0,
             # this might be okay. But usually, we want an initializer for Gen 0.
             return None # type: ignore

        # 2. Combine config and explicit dependencies/overrides
        full_config = {**dependencies, **config}

        # 3. Instantiate weighted initializers
        registered_inits: List[RegisteredInitializer[Any]] = []

        for name, weight in weights.items():
            cls = registered_classes[name]
            instance = self._instantiate(cls, full_config)
            registered_inits.append(RegisteredInitializer(weight, instance))

        # Check for pop_config in dependencies
        pop_config = full_config.get("pop_config")
        if not pop_config:
            raise ValueError("InitializerRegistry requires 'pop_config' in dependencies.")

        return WeightedPopulationInitializer(registered_inits, pop_config)


# Global registry instance
initializer_registry = InitializerRegistry()
