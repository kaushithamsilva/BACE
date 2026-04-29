"""Code initializers package."""

import pkgutil
import importlib

# Automatically discover and import all modules in this package.
# This triggers the @initializer_registry decorators for code initializers.
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f".{module_name}", __package__)
