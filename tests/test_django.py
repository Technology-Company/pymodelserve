"""Tests for Django integration."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# Configure Django settings before importing Django modules
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tests.django_settings")

import django

django.setup()

from django.test import RequestFactory  # noqa: E402

from pymodelserve.config.schema import ClientConfig, ModelConfig  # noqa: E402
from pymodelserve.contrib.django.registry import (  # noqa: E402
    get_model,
    get_registry,
    shutdown_models,
)
from pymodelserve.contrib.django.settings import (  # noqa: E402
    get_health_check_interval,
    get_models_dir,
    get_settings,
    is_auto_start_enabled,
)
from pymodelserve.contrib.django.views import (  # noqa: E402
    GenericModelView,
    ModelAPIView,
    ModelStatusView,
)


class TestDjangoSettings:
    """Tests for Django settings integration."""

    def test_get_settings(self):
        """Test getting MLSERVE settings."""
        settings = get_settings()
        assert isinstance(settings, dict)
        assert "auto_start" in settings
        assert settings["auto_start"] is False

    def test_get_health_check_interval(self):
        """Test getting health check interval."""
        interval = get_health_check_interval()
        assert interval == 30

    def test_is_auto_start_enabled(self):
        """Test checking auto_start setting."""
        assert is_auto_start_enabled() is False

    def test_get_models_dir_none(self):
        """Test getting models_dir when not set."""
        models_dir = get_models_dir()
        assert models_dir is None


class TestDjangoRegistry:
    """Tests for Django global registry."""

    def test_get_registry_creates_singleton(self):
        """Test that get_registry creates a singleton."""
        # Reset the global registry
        import pymodelserve.contrib.django.registry as reg_module

        reg_module._registry = None

        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    def test_get_model_not_found(self):
        """Test get_model raises KeyError for unknown model."""
        import pymodelserve.contrib.django.registry as reg_module

        reg_module._registry = None
        get_registry()  # Initialize empty registry

        with pytest.raises(KeyError):
            get_model("nonexistent_model")

    def test_shutdown_models(self):
        """Test shutdown_models clears registry."""
        import pymodelserve.contrib.django.registry as reg_module

        reg_module._registry = None
        registry = get_registry()

        # Register a mock model
        config = ModelConfig(
            name="test_model",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = Path("/tmp")

        with patch.object(registry, "stop_all"):
            shutdown_models()
            # Registry should be None after shutdown
            assert reg_module._registry is None


class TestModelAPIView:
    """Tests for ModelAPIView."""

    @pytest.fixture
    def factory(self):
        """Create request factory."""
        return RequestFactory()

    def test_get_handler_input_from_get(self, factory):
        """Test extracting input from GET request."""

        class TestView(ModelAPIView):
            model_name = "test"
            handler = "test_handler"

        view = TestView()
        request = factory.get("/test/?param1=value1&param2=value2")

        input_data = view.get_handler_input(request)

        assert input_data["param1"] == "value1"
        assert input_data["param2"] == "value2"

    def test_get_handler_input_from_post_json(self, factory):
        """Test extracting input from POST JSON request."""

        class TestView(ModelAPIView):
            model_name = "test"
            handler = "test_handler"

        view = TestView()
        request = factory.post(
            "/test/",
            data=json.dumps({"key": "value"}),
            content_type="application/json",
        )

        input_data = view.get_handler_input(request)

        assert input_data["key"] == "value"

    def test_get_handler_input_from_post_form(self, factory):
        """Test extracting input from POST form request."""

        class TestView(ModelAPIView):
            model_name = "test"
            handler = "test_handler"

        view = TestView()
        request = factory.post("/test/", data={"field": "data"})

        input_data = view.get_handler_input(request)

        assert input_data["field"] == "data"

    def test_format_response(self, factory):
        """Test response formatting."""

        class TestView(ModelAPIView):
            model_name = "test"
            handler = "test_handler"

        view = TestView()
        result = {"result": "success"}

        formatted = view.format_response(result)

        assert formatted == result

    def test_model_name_required(self, factory):
        """Test that model_name is required."""

        class TestView(ModelAPIView):
            handler = "test_handler"

        view = TestView()
        request = factory.get("/test/")

        with pytest.raises(ValueError, match="model_name not set"):
            view.get_model_name(request)

    def test_handler_required(self, factory):
        """Test that handler is required."""

        class TestView(ModelAPIView):
            model_name = "test"

        view = TestView()
        request = factory.get("/test/")

        with pytest.raises(ValueError, match="handler not set"):
            view.get_handler_name(request)


class TestGenericModelView:
    """Tests for GenericModelView."""

    @pytest.fixture
    def factory(self):
        """Create request factory."""
        return RequestFactory()

    def test_model_not_found(self, factory):
        """Test 404 when model not found."""
        import pymodelserve.contrib.django.registry as reg_module

        reg_module._registry = None
        get_registry()  # Initialize empty registry

        view = GenericModelView.as_view()
        request = factory.get("/test/")

        response = view(request, model_name="nonexistent", handler="test")

        assert response.status_code == 404
        data = json.loads(response.content)
        assert "not found" in data["error"]

    def test_get_handler_input(self, factory):
        """Test handler input extraction."""
        view = GenericModelView()

        # GET request
        request = factory.get("/test/?x=1&y=2")
        input_data = view.get_handler_input(request)
        assert input_data["x"] == "1"
        assert input_data["y"] == "2"

        # POST JSON request
        request = factory.post(
            "/test/",
            data=json.dumps({"a": "b"}),
            content_type="application/json",
        )
        input_data = view.get_handler_input(request)
        assert input_data["a"] == "b"


class TestModelStatusView:
    """Tests for ModelStatusView."""

    @pytest.fixture
    def factory(self):
        """Create request factory."""
        return RequestFactory()

    def test_status_empty_registry(self, factory):
        """Test status with empty registry."""
        import pymodelserve.contrib.django.registry as reg_module

        reg_module._registry = None
        get_registry()  # Initialize empty registry

        view = ModelStatusView.as_view()
        request = factory.get("/status/")

        response = view(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert "models" in data
        assert data["models"] == {}

    def test_status_with_models(self, factory):
        """Test status with registered models."""
        import pymodelserve.contrib.django.registry as reg_module

        reg_module._registry = None
        registry = get_registry()

        # Register a mock model
        config = ModelConfig(
            name="status_test",
            version="1.0.0",
            client=ClientConfig(module="model", class_name="Client"),
        )
        config.model_dir = Path("/tmp")
        registry.register(config=config)

        view = ModelStatusView.as_view()
        request = factory.get("/status/")

        response = view(request)

        assert response.status_code == 200
        data = json.loads(response.content)
        assert "status_test" in data["models"]
        assert data["models"]["status_test"]["version"] == "1.0.0"
