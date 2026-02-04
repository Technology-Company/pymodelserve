"""Tests for model discovery and registry."""

import pytest
from pathlib import Path

from pymodelserve.discovery.finder import (
    discover_models,
    ModelRegistry,
    DiscoveryError,
)
from pymodelserve.config.schema import ModelConfig, ClientConfig


class TestDiscoverModels:
    """Tests for discover_models function."""

    def test_discover_single_model(self, tmp_path):
        """Test discovering a single model."""
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()

        config_file = model_dir / "model.yaml"
        config_file.write_text("""
name: test_model
client:
  module: model
  class: TestClient
""")

        models = discover_models(tmp_path)

        assert "test_model" in models
        assert models["test_model"].name == "test_model"

    def test_discover_multiple_models(self, tmp_path):
        """Test discovering multiple models."""
        for name in ["model_a", "model_b", "model_c"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            (model_dir / "model.yaml").write_text(f"""
name: {name}
client:
  module: model
  class: Client
""")

        models = discover_models(tmp_path)

        assert len(models) == 3
        assert "model_a" in models
        assert "model_b" in models
        assert "model_c" in models

    def test_discover_nested_models(self, tmp_path):
        """Test discovering models in nested directories."""
        nested = tmp_path / "level1" / "level2" / "my_model"
        nested.mkdir(parents=True)
        (nested / "model.yaml").write_text("""
name: nested_model
client:
  module: model
  class: Client
""")

        models = discover_models(tmp_path, recursive=True)

        assert "nested_model" in models

    def test_discover_respects_max_depth(self, tmp_path):
        """Test that max_depth is respected."""
        deep = tmp_path / "l1" / "l2" / "l3" / "l4" / "l5"
        deep.mkdir(parents=True)
        (deep / "model.yaml").write_text("""
name: deep_model
client:
  module: model
  class: Client
""")

        models = discover_models(tmp_path, max_depth=2)

        assert "deep_model" not in models

    def test_discover_non_recursive(self, tmp_path):
        """Test non-recursive discovery."""
        # Model at root level
        root_model = tmp_path / "root_model"
        root_model.mkdir()
        (root_model / "model.yaml").write_text("""
name: root
client:
  module: model
  class: Client
""")

        # Model in subdirectory
        sub_model = tmp_path / "subdir" / "sub_model"
        sub_model.mkdir(parents=True)
        (sub_model / "model.yaml").write_text("""
name: sub
client:
  module: model
  class: Client
""")

        models = discover_models(tmp_path, recursive=False)

        assert "root" in models
        assert "sub" not in models

    def test_discover_skips_hidden_dirs(self, tmp_path):
        """Test that hidden directories are skipped."""
        hidden = tmp_path / ".hidden" / "model"
        hidden.mkdir(parents=True)
        (hidden / "model.yaml").write_text("""
name: hidden_model
client:
  module: model
  class: Client
""")

        models = discover_models(tmp_path)

        assert "hidden_model" not in models

    def test_discover_invalid_directory(self):
        """Test error on invalid directory."""
        with pytest.raises(DiscoveryError):
            discover_models(Path("/nonexistent/path"))

    def test_discover_handles_invalid_config(self, tmp_path):
        """Test that invalid configs are skipped with warning."""
        model_dir = tmp_path / "bad_model"
        model_dir.mkdir()
        (model_dir / "model.yaml").write_text("invalid: yaml: [")

        # Should not raise, just skip
        models = discover_models(tmp_path)
        assert len(models) == 0

    def test_discover_yml_extension(self, tmp_path):
        """Test discovering .yml files."""
        model_dir = tmp_path / "yml_model"
        model_dir.mkdir()
        (model_dir / "model.yml").write_text("""
name: yml_model
client:
  module: model
  class: Client
""")

        models = discover_models(tmp_path)

        assert "yml_model" in models


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_from_config(self, tmp_path):
        """Test registering a model from config."""
        config = ModelConfig(
            name="test_model",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        manager = registry.register(config=config)

        assert "test_model" in registry
        assert registry.get("test_model") is manager

    def test_register_from_dir(self, tmp_path):
        """Test registering a model from directory."""
        (tmp_path / "model.yaml").write_text("""
name: dir_model
client:
  module: model
  class: Client
""")

        registry = ModelRegistry()
        manager = registry.register(model_dir=tmp_path)

        assert "dir_model" in registry
        assert manager.name == "dir_model"

    def test_register_with_custom_name(self, tmp_path):
        """Test registering with custom name."""
        config = ModelConfig(
            name="original_name",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        registry.register(name="custom_name", config=config)

        assert "custom_name" in registry
        assert "original_name" not in registry

    def test_register_duplicate_raises(self, tmp_path):
        """Test that duplicate registration raises."""
        config = ModelConfig(
            name="dup_model",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        registry.register(config=config)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(config=config)

    def test_unregister(self, tmp_path):
        """Test unregistering a model."""
        config = ModelConfig(
            name="to_remove",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        registry.register(config=config)

        assert "to_remove" in registry

        registry.unregister("to_remove")

        assert "to_remove" not in registry

    def test_unregister_unknown_raises(self):
        """Test unregistering unknown model raises."""
        registry = ModelRegistry()

        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_unknown_raises(self):
        """Test getting unknown model raises."""
        registry = ModelRegistry()

        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_bracket_access(self, tmp_path):
        """Test bracket notation access."""
        config = ModelConfig(
            name="bracket_test",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        manager = registry.register(config=config)

        assert registry["bracket_test"] is manager

    def test_contains(self, tmp_path):
        """Test 'in' operator."""
        config = ModelConfig(
            name="contains_test",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        registry.register(config=config)

        assert "contains_test" in registry
        assert "nonexistent" not in registry

    def test_iter(self, tmp_path):
        """Test iteration over registry."""
        registry = ModelRegistry()

        for name in ["a", "b", "c"]:
            config = ModelConfig(
                name=name,
                client=ClientConfig(module="model", class_name="Client"),
            )
            config.model_dir = tmp_path
            registry.register(config=config)

        names = list(registry)

        assert set(names) == {"a", "b", "c"}

    def test_len(self, tmp_path):
        """Test len() on registry."""
        registry = ModelRegistry()

        assert len(registry) == 0

        for name in ["x", "y"]:
            config = ModelConfig(
                name=name,
                client=ClientConfig(module="model", class_name="Client"),
            )
            config.model_dir = tmp_path
            registry.register(config=config)

        assert len(registry) == 2

    def test_names_property(self, tmp_path):
        """Test names property."""
        registry = ModelRegistry()

        for name in ["first", "second"]:
            config = ModelConfig(
                name=name,
                client=ClientConfig(module="model", class_name="Client"),
            )
            config.model_dir = tmp_path
            registry.register(config=config)

        assert set(registry.names) == {"first", "second"}

    def test_get_config(self, tmp_path):
        """Test getting model config."""
        config = ModelConfig(
            name="config_test",
            version="2.0.0",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        registry.register(config=config)

        retrieved = registry.get_config("config_test")

        assert retrieved.name == "config_test"
        assert retrieved.version == "2.0.0"

    def test_register_from_dir_discovery(self, tmp_path):
        """Test register_from_dir with discovery."""
        for name in ["model1", "model2"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            (model_dir / "model.yaml").write_text(f"""
name: {name}
client:
  module: model
  class: Client
""")

        registry = ModelRegistry()
        registered = registry.register_from_dir(tmp_path)

        assert set(registered) == {"model1", "model2"}
        assert "model1" in registry
        assert "model2" in registry

    def test_status(self, tmp_path):
        """Test status method."""
        config = ModelConfig(
            name="status_test",
            version="1.2.3",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        registry = ModelRegistry()
        registry.register(config=config)

        status = registry.status()

        assert "status_test" in status
        assert status["status_test"]["running"] is False
        assert status["status_test"]["version"] == "1.2.3"
