"""Django app configuration for pymodelserve."""

from django.apps import AppConfig


class PyModelServeConfig(AppConfig):
    """Django app config for pymodelserve."""

    name = "pymodelserve.contrib.django"
    verbose_name = "PyModelServe"
    default_auto_field = "django.db.models.BigAutoField"

    def ready(self) -> None:
        """Initialize models when Django starts."""
        from pymodelserve.contrib.django.settings import get_settings

        settings = get_settings()

        if settings.get("auto_start", False):
            from pymodelserve.contrib.django.registry import get_registry

            registry = get_registry()

            # Register and start models
            models_dir = settings.get("models_dir")
            if models_dir:
                from pathlib import Path

                registry.register_from_dir(Path(models_dir))
                registry.start_all()
