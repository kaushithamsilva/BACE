"""Centralized registries for population profiles, initializers, and operators."""

from .profile import profile_registry
from .initializer import initializer_registry
from .operator import operator_registry

__all__ = [
    "profile_registry",
    "initializer_registry",
    "operator_registry",
]
