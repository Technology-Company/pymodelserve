"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest

from pymodelserve.config.loader import ConfigError, load_config
from pymodelserve.config.schema import ClientConfig, HandlerConfig, ModelConfig


class TestModelConfig:
    """Tests for ModelConfig schema."""

    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = ModelConfig(
            name="test_model",
            client=ClientConfig(module="model", class_name="TestClient"),
        )
        assert config.name == "test_model"
        assert config.version == "1.0.0"
        assert config.client.module == "model"
        assert config.client.class_name == "TestClient"

    def test_full_config(self):
        """Test creating config with all fields."""
        config = ModelConfig(
            name="full_model",
            version="2.0.0",
            python=">=3.11",
            client=ClientConfig(module="my_model", class_name="MyClient"),
            requirements="deps.txt",
            handlers=[HandlerConfig(name="predict", input={"x": "float"}, output={"y": "float"})],
        )
        assert config.name == "full_model"
        assert config.version == "2.0.0"
        assert len(config.handlers) == 1
        assert config.handlers[0].name == "predict"

    def test_invalid_name(self):
        """Test that invalid names are rejected."""
        with pytest.raises(ValueError):
            ModelConfig(
                name="invalid name with spaces",
                client=ClientConfig(module="model", class_name="Client"),
            )

    def test_get_handler_names(self):
        """Test getting handler names."""
        config = ModelConfig(
            name="test",
            client=ClientConfig(module="m", class_name="C"),
            handlers=[
                HandlerConfig(name="a"),
                HandlerConfig(name="b"),
            ],
        )
        assert config.get_handler_names() == ["a", "b"]

    def test_get_handler(self):
        """Test getting specific handler."""
        config = ModelConfig(
            name="test",
            client=ClientConfig(module="m", class_name="C"),
            handlers=[
                HandlerConfig(name="predict", input={"x": "int"}),
            ],
        )
        handler = config.get_handler("predict")
        assert handler is not None
        assert handler.name == "predict"
        assert config.get_handler("nonexistent") is None


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_yaml_file(self):
        """Test loading a YAML config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model.yaml"
            config_path.write_text("""
name: yaml_test
version: "1.0.0"
client:
  module: model
  class: TestClient
""")
            config = load_config(config_path)
            assert config.name == "yaml_test"
            assert config.client.class_name == "TestClient"
            assert config.model_dir == Path(tmpdir)

    def test_load_from_dict(self):
        """Test loading config from dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(
                {
                    "name": "dict_test",
                    "client": {"module": "m", "class": "C"},
                },
                model_dir=Path(tmpdir),
            )
            assert config.name == "dict_test"

    def test_load_missing_file(self):
        """Test error on missing config file."""
        with pytest.raises(ConfigError):
            load_config(Path("/nonexistent/model.yaml"))

    def test_load_invalid_yaml(self):
        """Test error on invalid YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "model.yaml"
            config_path.write_text("invalid: yaml: content: [")
            with pytest.raises(ConfigError):
                load_config(config_path)
