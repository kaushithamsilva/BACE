"""Population discovery service for dynamic profile construction."""

import inspect

from typing import Any, Dict

from loguru import logger

from ..core.interfaces import IExecutionSystem
from ..core.interfaces.language import ILanguage
from ..populations.registries import profile_registry
from ..populations.registries.profile import ProfileRegistry
from infrastructure.llm_client import LLMClient
from infrastructure.sandbox.types import SandboxConfig


class PopulationDiscoveryService:
    """Service to dynamically discover and construct population profiles."""

    def __init__(
        self,
        llm_client: LLMClient,
        language_adapter: ILanguage,
        execution_system: IExecutionSystem,
        sandbox_config: SandboxConfig,
        cpu_workers: int,
    ) -> None:
        self.llm_client = llm_client
        self.language_adapter = language_adapter
        self.execution_system = execution_system
        self.sandbox_config = sandbox_config
        self.cpu_workers = cpu_workers
        self.registry: ProfileRegistry = profile_registry

    def construct_all(self, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Discover and construct all profiles defined in the config.

        Returns:
            Dict containing:
                - code_profile: CodeProfile
                - evolved_test_profiles: Dict[str, TestProfile]
                - public_test_profile: Optional[PublicTestProfile]
        """
        results: Dict[str, Any] = {
            "code_profile": None,
            "evolved_test_profiles": {},
            "public_test_profile": None,
        }

        # 1. Public Profile
        public_cfg = experiment_config.get("public_profile")
        if public_cfg:
            results["public_test_profile"] = self._construct_profile(
                self.registry.get_public_factory("public"), public_cfg
            )
            logger.info("Constructed public test profile")

        # 2. Evolved Test Populations
        # We iterate through all registered test populations and see if they are in the config
        injected_repair_ops: dict[str, Any] = {}

        for test_type in self.registry.registered_test_populations:
            profile_key = f"{test_type}_profile"
            test_cfg = experiment_config.get(profile_key)
            if test_cfg:
                factory = self.registry.get_test_factory(test_type)
                profile = self._construct_profile(factory, test_cfg)
                results["evolved_test_profiles"][test_type] = profile
                
                # Collect repair operators "brought in" by this test population
                for reg_op in profile.repair_operators:
                    name = reg_op.operator.operation_name()
                    injected_repair_ops[name] = reg_op.operator
                    logger.debug(f"Collected injected repair operator '{name}' from '{test_type}' profile")
                
                logger.info(f"Constructed evolved test profile for '{test_type}'")

        # 3. Code Profile
        code_cfg = experiment_config.get("code_profile")
        if code_cfg:
            # Determine profile type (e.g., "default", "agent_coder")
            code_type = experiment_config.get("code_profile_type", "default")
            
            # Pass injected operators to the code factory
            results["code_profile"] = self._construct_profile(
                self.registry.get_code_factory(code_type), 
                code_cfg,
                injected_operators=injected_repair_ops
            )
            logger.info(f"Constructed code profile of type '{code_type}'")

        return results

    def _construct_profile(
        self, factory: Any, config: Dict[str, Any], **extra_deps: Any
    ) -> Any:
        """Call a factory function with arguments from config + common dependencies."""
        sig = inspect.signature(factory)
        kwargs = {}

        # Common dependencies that factories might need
        common_deps = {
            "llm_client": self.llm_client,
            "language_adapter": self.language_adapter,
            "execution_system": self.execution_system,
            "sandbox_config": self.sandbox_config,
            "cpu_workers": self.cpu_workers,
            **extra_deps,
        }

        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        for param_name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue # Handled by has_kwargs logic
            if param_name in config:
                kwargs[param_name] = config[param_name]
            elif param_name in common_deps:
                kwargs[param_name] = common_deps[param_name]
            elif param.default is not inspect.Parameter.empty:
                # Use default value if it exists
                continue
            else:
                # Parameter is required but not in config or common deps
                logger.warning(
                    f"Parameter '{param_name}' required by {factory.__name__} "
                    f"not found in config or common dependencies."
                )

        if has_kwargs:
            # Pass all keys from config that weren't explicitly handled
            # (although passing all of them is usually fine as long as names don't clash)
            for k, v in config.items():
                if k not in kwargs:
                    kwargs[k] = v

        return factory(**kwargs)
