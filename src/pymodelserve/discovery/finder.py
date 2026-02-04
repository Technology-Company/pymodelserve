"""Model discovery and registry for managing multiple models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

from pymodelserve.config.loader import find_config, load_config
from pymodelserve.config.schema import ModelConfig
from pymodelserve.core.manager import ModelManager

logger = logging.getLogger(__name__)


class DiscoveryError(Exception):
    """Error during model discovery."""


def discover_models(
    base_dir: str | Path,
    recursive: bool = True,
    max_depth: int = 3,
) -> dict[str, ModelConfig]:
    """Discover models in a directory.

    Scans for model.yaml/model.yml/model.toml files.

    Args:
        base_dir: Base directory to search.
        recursive: Search subdirectories beyond immediate children.
        max_depth: Maximum recursion depth.

    Returns:
        Dictionary mapping model names to their configurations.
    """
    base_dir = Path(base_dir)

    if not base_dir.is_dir():
        raise DiscoveryError(f"Not a directory: {base_dir}")

    models: dict[str, ModelConfig] = {}

    def scan_dir(dir_path: Path, depth: int = 0) -> None:
        if depth > max_depth:
            return

        # Check for config file in this directory
        config_path = find_config(dir_path)
        if config_path is not None:
            try:
                config = load_config(config_path, model_dir=dir_path)
                if config.name in models:
                    logger.warning(
                        f"Duplicate model name '{config.name}' found at {dir_path}"
                    )
                else:
                    models[config.name] = config
                    logger.debug(f"Discovered model '{config.name}' at {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to load config at {config_path}: {e}")
            return  # Don't recurse into model directories

        # Always scan immediate subdirectories (depth 0 -> 1)
        # Only recurse deeper if recursive=True
        if depth == 0 or recursive:
            for subdir in dir_path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    scan_dir(subdir, depth + 1)

    scan_dir(base_dir)
    return models


class ModelRegistry:
    """Registry for managing multiple models.

    Example:
        registry = ModelRegistry()
        registry.register("fruit", "./models/fruit_classifier/")
        registry.register("sentiment", "./models/sentiment/")

        registry.start_all()

        result = registry.get("fruit").request("classify", {"image_path": "..."})

        registry.stop_all()
    """

    def __init__(self) -> None:
        self._managers: dict[str, ModelManager] = {}
        self._configs: dict[str, ModelConfig] = {}

    def register(
        self,
        name: str | None = None,
        model_dir: str | Path | None = None,
        config: ModelConfig | None = None,
        **manager_kwargs: Any,
    ) -> ModelManager:
        """Register a model with the registry.

        Args:
            name: Model name (uses config name if not provided).
            model_dir: Path to model directory.
            config: Pre-loaded ModelConfig (alternative to model_dir).
            **manager_kwargs: Additional arguments for ModelManager.

        Returns:
            The registered ModelManager instance.
        """
        if config is None:
            if model_dir is None:
                raise ValueError("Must provide either model_dir or config")
            from pymodelserve.config.loader import load_config_from_dir

            config = load_config_from_dir(Path(model_dir))

        model_name = name or config.name

        if model_name in self._managers:
            raise ValueError(f"Model '{model_name}' already registered")

        manager = ModelManager(config, **manager_kwargs)
        self._managers[model_name] = manager
        self._configs[model_name] = config

        logger.info(f"Registered model '{model_name}'")
        return manager

    def register_from_dir(
        self,
        base_dir: str | Path,
        recursive: bool = True,
        **manager_kwargs: Any,
    ) -> list[str]:
        """Register all models discovered in a directory.

        Args:
            base_dir: Directory to scan for models.
            recursive: Search subdirectories.
            **manager_kwargs: Arguments for ModelManager instances.

        Returns:
            List of registered model names.
        """
        configs = discover_models(base_dir, recursive=recursive)
        registered = []

        for name, config in configs.items():
            try:
                self.register(config=config, **manager_kwargs)
                registered.append(name)
            except ValueError as e:
                logger.warning(f"Skipping model '{name}': {e}")

        return registered

    def unregister(self, name: str) -> None:
        """Unregister a model.

        Args:
            name: Model name to unregister.
        """
        if name not in self._managers:
            raise KeyError(f"Model '{name}' not registered")

        manager = self._managers[name]
        if manager.is_running:
            manager.stop()

        del self._managers[name]
        del self._configs[name]
        logger.info(f"Unregistered model '{name}'")

    def get(self, name: str) -> ModelManager:
        """Get a model manager by name.

        Args:
            name: Model name.

        Returns:
            ModelManager instance.
        """
        if name not in self._managers:
            raise KeyError(f"Model '{name}' not registered")
        return self._managers[name]

    def get_config(self, name: str) -> ModelConfig:
        """Get a model's configuration.

        Args:
            name: Model name.

        Returns:
            ModelConfig instance.
        """
        if name not in self._configs:
            raise KeyError(f"Model '{name}' not registered")
        return self._configs[name]

    def __getitem__(self, name: str) -> ModelManager:
        """Get model by name using bracket notation."""
        return self.get(name)

    def __contains__(self, name: str) -> bool:
        """Check if model is registered."""
        return name in self._managers

    def __iter__(self) -> Iterator[str]:
        """Iterate over registered model names."""
        return iter(self._managers)

    def __len__(self) -> int:
        """Number of registered models."""
        return len(self._managers)

    @property
    def names(self) -> list[str]:
        """List of registered model names."""
        return list(self._managers.keys())

    def start(self, name: str, **kwargs: Any) -> None:
        """Start a specific model.

        Args:
            name: Model name.
            **kwargs: Arguments for ModelManager.start().
        """
        self.get(name).start(**kwargs)

    def stop(self, name: str, **kwargs: Any) -> None:
        """Stop a specific model.

        Args:
            name: Model name.
            **kwargs: Arguments for ModelManager.stop().
        """
        self.get(name).stop(**kwargs)

    def start_all(self, **kwargs: Any) -> dict[str, Exception | None]:
        """Start all registered models.

        Args:
            **kwargs: Arguments for ModelManager.start().

        Returns:
            Dictionary mapping model names to exceptions (None if successful).
        """
        results: dict[str, Exception | None] = {}

        for name, manager in self._managers.items():
            try:
                if not manager.is_running:
                    manager.start(**kwargs)
                results[name] = None
            except Exception as e:
                logger.error(f"Failed to start model '{name}': {e}")
                results[name] = e

        return results

    def stop_all(self, **kwargs: Any) -> None:
        """Stop all running models.

        Args:
            **kwargs: Arguments for ModelManager.stop().
        """
        for name, manager in self._managers.items():
            try:
                if manager.is_running:
                    manager.stop(**kwargs)
            except Exception as e:
                logger.error(f"Error stopping model '{name}': {e}")

    def status(self) -> dict[str, dict[str, Any]]:
        """Get status of all models.

        Returns:
            Dictionary mapping model names to status info.
        """
        return {
            name: {
                "running": manager.is_running,
                "version": self._configs[name].version,
                "handlers": self._configs[name].get_handler_names(),
            }
            for name, manager in self._managers.items()
        }

    def __enter__(self) -> ModelRegistry:
        """Context manager entry - starts all models."""
        self.start_all()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit - stops all models."""
        self.stop_all()
