"""Configuration system for pymodelserve."""

from pymodelserve.config.loader import load_config
from pymodelserve.config.schema import ModelConfig

__all__ = ["ModelConfig", "load_config"]
