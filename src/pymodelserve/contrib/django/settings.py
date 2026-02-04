"""Django settings integration for pymodelserve."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def get_settings() -> dict[str, Any]:
    """Get pymodelserve settings from Django settings.

    Reads from MLSERVE setting in Django settings.py.

    Example Django settings:
        MLSERVE = {
            "models_dir": BASE_DIR / "ml_models",
            "auto_start": True,
            "health_check_interval": 30,
        }
    """
    try:
        from django.conf import settings

        return getattr(settings, "MLSERVE", {})
    except Exception:
        return {}


def get_models_dir() -> Path | None:
    """Get the configured models directory."""
    settings = get_settings()
    models_dir = settings.get("models_dir")

    if models_dir is None:
        return None

    return Path(models_dir)


def get_health_check_interval() -> int:
    """Get health check interval in seconds."""
    settings = get_settings()
    return settings.get("health_check_interval", 30)


def is_auto_start_enabled() -> bool:
    """Check if auto-start is enabled."""
    settings = get_settings()
    return settings.get("auto_start", False)
