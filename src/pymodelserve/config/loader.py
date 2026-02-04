"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pymodelserve.config.schema import ModelConfig


class ConfigError(Exception):
    """Configuration loading or validation error."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents as a dictionary.

    Raises:
        ConfigError: If the file cannot be loaded or parsed.
    """
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"Cannot read config file {path}: {e}") from e


def load_config(
    source: Path | str | dict[str, Any],
    model_dir: Path | None = None,
) -> ModelConfig:
    """Load and validate model configuration.

    Args:
        source: Path to config file (YAML), or config dict.
        model_dir: Model directory. If not provided, inferred from config path.

    Returns:
        Validated ModelConfig instance.

    Raises:
        ConfigError: If configuration is invalid.
    """
    if isinstance(source, dict):
        config_dict = source
        if model_dir is None:
            raise ConfigError("model_dir required when loading from dict")
    else:
        source = Path(source)

        if not source.exists():
            raise ConfigError(f"Config file not found: {source}")

        # Determine model directory from config file location
        if model_dir is None:
            model_dir = source.parent

        # Load based on file extension
        if source.suffix in (".yaml", ".yml"):
            config_dict = load_yaml(source)
        elif source.suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib

            with open(source, "rb") as f:
                config_dict = tomllib.load(f)
        else:
            raise ConfigError(f"Unsupported config format: {source.suffix}")

    try:
        config = ModelConfig.model_validate(config_dict)
        config.model_dir = Path(model_dir)
        return config
    except Exception as e:
        raise ConfigError(f"Invalid configuration: {e}") from e


def find_config(model_dir: Path) -> Path | None:
    """Find the configuration file in a model directory.

    Looks for model.yaml, model.yml, or model.toml.

    Args:
        model_dir: Directory to search.

    Returns:
        Path to config file if found, None otherwise.
    """
    for filename in ("model.yaml", "model.yml", "model.toml"):
        config_path = model_dir / filename
        if config_path.exists():
            return config_path
    return None


def load_config_from_dir(model_dir: Path | str) -> ModelConfig:
    """Load configuration from a model directory.

    Args:
        model_dir: Directory containing the model and config.

    Returns:
        Validated ModelConfig.

    Raises:
        ConfigError: If no config file is found or config is invalid.
    """
    model_dir = Path(model_dir)

    if not model_dir.is_dir():
        raise ConfigError(f"Not a directory: {model_dir}")

    config_path = find_config(model_dir)
    if config_path is None:
        raise ConfigError(f"No config file found in {model_dir}")

    return load_config(config_path, model_dir=model_dir)
