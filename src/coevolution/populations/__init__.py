import pkgutil
import importlib
from .registries import profile_registry

# Automatically discover and import all population packages in this directory.
# This triggers the @register decorators in each population's profile.py.
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if is_pkg and module_name != "registries":
        importlib.import_module(f".{module_name}", __package__)

__all__ = ["profile_registry"]

