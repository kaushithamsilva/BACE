"""Code population — operators package.

Core operators for the code population. Specialized test-driven repair
operators are provided by their respective test population packages.
"""

import pkgutil
import importlib

# Automatically discover and import all modules in this package.
# This triggers the @operator_registry decorators for code operators.
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    importlib.import_module(f".{module_name}", __package__)
