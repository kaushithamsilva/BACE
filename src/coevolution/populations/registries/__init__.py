"""Centralized registries for population profiles, initializers, and operators."""

from .profile import profile_registry, registry  # 'registry' kept for backward compat
from .initializer import initializer_registry
from .operator import operator_registry

__all__ = [
    "profile_registry",
    "registry",
    "initializer_registry",
    "operator_registry",
]
