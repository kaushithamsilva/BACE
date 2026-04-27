"""Base classes for dependency-injection enabled registries."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Type, TypeVar

from loguru import logger

T = TypeVar("T")


class DIRegistry[T]:
    """Base registry with dependency injection capabilities."""

    def __init__(self) -> None:
        # Structure: {population_type: {name: class}}
        self._registry: Dict[str, Dict[str, Type[T]]] = {}

    def register(self, name: str, population: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class.

        Args:
            name: Unique name for this component within the population.
            population: Population type (e.g. 'code', 'unittest').
        """

        def wrapper(cls: Type[T]) -> Type[T]:
            if population not in self._registry:
                self._registry[population] = {}
            self._registry[population][name] = cls
            logger.trace(f"Registered {cls.__name__} as '{name}' for population '{population}'")
            return cls

        return wrapper

    def get_all(self, population: str) -> Dict[str, Type[T]]:
        """Get all registered components for a population."""
        return self._registry.get(population, {}).copy()

    def list_names(self, population: str) -> List[str]:
        """List registered names for a population."""
        return list(self._registry.get(population, {}).keys())

    def _instantiate(self, cls: Type[T], context: Dict[str, Any]) -> T:
        """Instantiate a class by injecting matching parameters from the context."""
        # Handle classes without an explicit __init__ (like protocols or simple classes)
        if not hasattr(cls, "__init__") or cls.__init__ is object.__init__:
             return cls()
             
        sig = inspect.signature(cls.__init__)

        # Inject context values that match parameter names
        kwargs = {k: v for k, v in context.items() if k in sig.parameters}

        # Check for missing required parameters (those without a default value)
        for name, param in sig.parameters.items():
            if name in {"self", "args", "kwargs"}:
                continue
            if name not in kwargs and param.default is inspect.Parameter.empty:
                logger.warning(
                    f"Required parameter '{name}' for {cls.__name__} not found in creation context."
                )

        return cls(**kwargs)
