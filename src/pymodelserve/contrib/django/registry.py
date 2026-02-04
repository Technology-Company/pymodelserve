"""Global registry for Django integration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pymodelserve.discovery.finder import ModelRegistry
    from pymodelserve.core.manager import ModelManager

# Global registry instance
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    """Get the global model registry.

    Creates the registry on first access.

    Returns:
        The global ModelRegistry instance.
    """
    global _registry

    if _registry is None:
        from pymodelserve.discovery.finder import ModelRegistry

        _registry = ModelRegistry()

    return _registry


def get_model(name: str) -> ModelManager:
    """Get a model by name from the global registry.

    Convenience function for Django views and code.

    Args:
        name: Model name.

    Returns:
        ModelManager instance.

    Raises:
        KeyError: If model not found.
    """
    return get_registry().get(name)


def shutdown_models() -> None:
    """Shutdown all models in the global registry.

    Call this when Django shuts down (e.g., in AppConfig.ready or signal handler).
    """
    global _registry

    if _registry is not None:
        _registry.stop_all()
        _registry = None
