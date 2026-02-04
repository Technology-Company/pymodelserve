"""Tests for ModelManager."""

import pytest
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from pymodelserve.core.manager import (
    ModelManager,
    ModelManagerError,
    ModelNotStartedError,
    ModelStartupError,
    ModelRequestError,
)
from pymodelserve.config.schema import ModelConfig, ClientConfig


class TestModelManagerInit:
    """Tests for ModelManager initialization."""

    def test_from_yaml(self, tmp_path):
        """Test creating manager from YAML file."""
        config_file = tmp_path / "model.yaml"
        config_file.write_text("""
name: yaml_model
client:
  module: model
  class: TestClient
""")

        manager = ModelManager.from_yaml(config_file)

        assert manager.name == "yaml_model"
        assert manager.model_dir == tmp_path

    def test_from_dir(self, tmp_path):
        """Test creating manager from directory."""
        (tmp_path / "model.yaml").write_text("""
name: dir_model
client:
  module: model
  class: TestClient
""")

        manager = ModelManager.from_dir(tmp_path)

        assert manager.name == "dir_model"

    def test_from_config_dict(self, tmp_path):
        """Test creating manager from config dict."""
        manager = ModelManager.from_config(
            {
                "name": "dict_model",
                "client": {"module": "model", "class": "Client"},
            },
            model_dir=tmp_path,
        )

        assert manager.name == "dict_model"

    def test_properties(self, tmp_path):
        """Test manager properties."""
        config = ModelConfig(
            name="prop_test",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        manager = ModelManager(config)

        assert manager.name == "prop_test"
        assert manager.model_dir == tmp_path
        assert manager.is_running is False

    def test_repr(self, tmp_path):
        """Test string representation."""
        config = ModelConfig(
            name="repr_test",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = tmp_path

        manager = ModelManager(config)

        assert "repr_test" in repr(manager)
        assert "stopped" in repr(manager)


@pytest.mark.slow
class TestModelManagerLifecycle:
    """Tests for ModelManager lifecycle (start/stop).

    These tests create real venvs and spawn subprocesses.
    """

    @pytest.fixture
    def simple_model_dir(self, tmp_path):
        """Create a simple model directory for testing."""
        # Get the project root for installing pymodelserve
        project_root = Path(__file__).parent.parent

        # Create model.yaml
        (tmp_path / "model.yaml").write_text("""
name: simple_test
client:
  module: model
  class: SimpleClient
""")

        # Create requirements.txt - install local pymodelserve
        (tmp_path / "requirements.txt").write_text(f"-e {project_root}\n")

        # Create model.py
        (tmp_path / "model.py").write_text('''
from pymodelserve import ModelClient, handler

class SimpleClient(ModelClient):
    @handler("echo")
    def echo(self, message: str = "default") -> dict:
        return {"echoed": message}

    @handler("add")
    def add(self, a: int, b: int) -> dict:
        return {"result": a + b}

if __name__ == "__main__":
    SimpleClient().run()
''')

        return tmp_path

    def test_start_creates_venv(self, simple_model_dir):
        """Test that start creates venv."""
        manager = ModelManager.from_dir(simple_model_dir)

        venv_dir = simple_model_dir / "model_venv"
        assert not venv_dir.exists()

        try:
            manager.start(timeout=60)
            assert venv_dir.exists()
            assert manager.is_running
        finally:
            manager.stop()

    def test_start_stop(self, simple_model_dir):
        """Test basic start/stop cycle."""
        manager = ModelManager.from_dir(simple_model_dir)

        manager.start(timeout=60)
        assert manager.is_running

        manager.stop()
        assert not manager.is_running

    def test_start_twice_is_safe(self, simple_model_dir):
        """Test that starting twice doesn't crash."""
        manager = ModelManager.from_dir(simple_model_dir)

        try:
            manager.start(timeout=60)
            manager.start(timeout=60)  # Should be no-op
            assert manager.is_running
        finally:
            manager.stop()

    def test_stop_without_start_is_safe(self, simple_model_dir):
        """Test that stopping without starting is safe."""
        manager = ModelManager.from_dir(simple_model_dir)
        manager.stop()  # Should not raise

    def test_context_manager(self, simple_model_dir):
        """Test context manager usage."""
        with ModelManager.from_dir(simple_model_dir) as manager:
            assert manager.is_running

        assert not manager.is_running

    def test_ping(self, simple_model_dir):
        """Test ping functionality."""
        with ModelManager.from_dir(simple_model_dir) as manager:
            assert manager.ping() is True

    def test_request_echo(self, simple_model_dir):
        """Test basic request."""
        with ModelManager.from_dir(simple_model_dir) as manager:
            result = manager.request("echo", {"message": "hello"})

            assert result["echoed"] == "hello"

    def test_request_add(self, simple_model_dir):
        """Test request with computation."""
        with ModelManager.from_dir(simple_model_dir) as manager:
            result = manager.request("add", {"a": 5, "b": 3})

            assert result["result"] == 8

    def test_request_without_start_raises(self, simple_model_dir):
        """Test that request without start raises."""
        manager = ModelManager.from_dir(simple_model_dir)

        with pytest.raises(ModelNotStartedError):
            manager.request("echo", {})

    def test_restart(self, simple_model_dir):
        """Test restart functionality."""
        manager = ModelManager.from_dir(simple_model_dir)

        try:
            manager.start(timeout=60)
            assert manager.is_running

            manager.restart()
            assert manager.is_running

            # Should still work after restart
            result = manager.request("echo", {"message": "after restart"})
            assert result["echoed"] == "after restart"
        finally:
            manager.stop()


@pytest.mark.slow
class TestModelManagerErrors:
    """Tests for error handling.

    Some of these tests start model processes.
    """

    def test_missing_model_dir_raises(self, tmp_path):
        """Test error when model_dir is not set."""
        config = ModelConfig(
            name="no_dir",
            client=ClientConfig(module="model", class_name="Client"),
        )
        # Don't set model_dir

        manager = ModelManager(config)

        with pytest.raises(ModelManagerError):
            _ = manager.model_dir

    def test_invalid_client_module_fails(self, tmp_path):
        """Test error when client module doesn't exist."""
        (tmp_path / "model.yaml").write_text("""
name: bad_module
client:
  module: nonexistent_module
  class: Client
""")
        (tmp_path / "requirements.txt").write_text("")

        manager = ModelManager.from_dir(tmp_path)

        with pytest.raises(ModelStartupError):
            manager.start(timeout=10)

    def test_request_error_handling(self, tmp_path):
        """Test request error for unknown handler."""
        (tmp_path / "model.yaml").write_text("""
name: error_test
client:
  module: model
  class: ErrorClient
""")
        (tmp_path / "requirements.txt").write_text("")
        (tmp_path / "model.py").write_text('''
from pymodelserve import ModelClient

class ErrorClient(ModelClient):
    pass

if __name__ == "__main__":
    ErrorClient().run()
''')

        with ModelManager.from_dir(tmp_path) as manager:
            # Unknown handler should return error in response
            with pytest.raises(ModelRequestError, match="Unknown message"):
                manager.request("nonexistent_handler", {})
