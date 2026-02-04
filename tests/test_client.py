"""Tests for ModelClient."""

import pytest
from unittest.mock import MagicMock, patch

from pymodelserve.core.client import ModelClient, handler, run_client


class TestHandlerDecorator:
    """Tests for @handler decorator."""

    def test_handler_sets_attribute(self):
        """Test that @handler sets _handler_name attribute."""

        @handler("test_handler")
        def my_func(self):
            return {"result": "ok"}

        assert hasattr(my_func, "_handler_name")
        assert my_func._handler_name == "test_handler"

    def test_handler_preserves_function(self):
        """Test that @handler preserves function behavior."""

        class TestClass:
            @handler("compute")
            def compute(self, x: int) -> dict:
                return {"result": x * 2}

        obj = TestClass()
        result = obj.compute(5)
        assert result == {"result": 10}


class TestModelClient:
    """Tests for ModelClient base class."""

    def test_discover_decorated_handlers(self):
        """Test that decorated handlers are discovered."""

        class TestClient(ModelClient):
            @handler("foo")
            def foo_handler(self):
                return {"status": "foo"}

            @handler("bar")
            def bar_handler(self):
                return {"status": "bar"}

        client = TestClient()
        handlers = client.get_handlers()

        assert "foo" in handlers
        assert "bar" in handlers
        assert "ping" in handlers  # Built-in
        assert "shutdown" in handlers  # Built-in

    def test_discover_handle_prefix_handlers(self):
        """Test that handle_* methods are discovered."""

        class TestClient(ModelClient):
            def handle_process(self, data: dict) -> dict:
                return {"processed": True}

            def handle_info(self) -> dict:
                return {"version": "1.0"}

        client = TestClient()
        handlers = client.get_handlers()

        assert "process" in handlers
        assert "info" in handlers

    def test_builtin_ping_handler(self):
        """Test built-in ping handler."""

        class TestClient(ModelClient):
            @handler("custom")
            def custom(self):
                return {}

        client = TestClient()
        result = client.handle_message("ping", {})

        assert result["status"] == "pong"
        assert "handlers" in result
        assert "custom" in result["handlers"]
        assert "ping" in result["handlers"]

    def test_builtin_shutdown_handler(self):
        """Test built-in shutdown handler."""
        client = ModelClient()
        client._running = True

        result = client.handle_message("shutdown", {})

        assert result["status"] == "shutting_down"
        assert client._running is False

    def test_handle_message_calls_handler(self):
        """Test that handle_message dispatches correctly."""

        class TestClient(ModelClient):
            @handler("echo")
            def echo(self, message: str) -> dict:
                return {"echoed": message}

        client = TestClient()
        result = client.handle_message("echo", {"message": "hello"})

        assert result["echoed"] == "hello"

    def test_handle_message_unknown_handler(self):
        """Test handling unknown message type."""
        client = ModelClient()
        result = client.handle_message("nonexistent", {})

        assert "error" in result
        assert "Unknown message type" in result["error"]
        assert "available_handlers" in result

    def test_handle_message_wrong_args(self):
        """Test handling wrong arguments."""

        class TestClient(ModelClient):
            @handler("strict")
            def strict(self, required_arg: str) -> dict:
                return {"got": required_arg}

        client = TestClient()
        result = client.handle_message("strict", {})  # Missing required_arg

        assert "error" in result
        assert "argument" in result["error"].lower()

    def test_handle_message_exception(self):
        """Test handling exception in handler."""

        class TestClient(ModelClient):
            @handler("fail")
            def fail(self) -> dict:
                raise ValueError("intentional error")

        client = TestClient()
        result = client.handle_message("fail", {})

        assert "error" in result
        assert "intentional error" in result["error"]
        assert "traceback" in result

    def test_handle_message_non_dict_result(self):
        """Test that non-dict results are wrapped."""

        class TestClient(ModelClient):
            @handler("simple")
            def simple(self) -> str:
                return "just a string"

        client = TestClient()
        result = client.handle_message("simple", {})

        assert result == {"result": "just a string"}

    def test_setup_teardown_hooks(self):
        """Test that setup and teardown can be overridden."""
        setup_called = []
        teardown_called = []

        class TestClient(ModelClient):
            def setup(self):
                setup_called.append(True)

            def teardown(self):
                teardown_called.append(True)

        client = TestClient()
        client.setup()
        client.teardown()

        assert len(setup_called) == 1
        assert len(teardown_called) == 1

    def test_combined_handler_styles(self):
        """Test mixing @handler and handle_* styles."""

        class TestClient(ModelClient):
            @handler("decorated")
            def my_decorated(self):
                return {"style": "decorator"}

            def handle_prefixed(self):
                return {"style": "prefix"}

        client = TestClient()

        result1 = client.handle_message("decorated", {})
        result2 = client.handle_message("prefixed", {})

        assert result1["style"] == "decorator"
        assert result2["style"] == "prefix"
