"""Django views for pymodelserve."""

from __future__ import annotations

import json
import logging
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.views import View

from pymodelserve.contrib.django.registry import get_model, get_registry
from pymodelserve.core.manager import ModelManagerError, ModelNotStartedError, ModelRequestError

logger = logging.getLogger(__name__)


class ModelAPIView(View):
    """Base view for model API endpoints.

    Subclass this to create endpoints for specific models.

    Example:
        class ClassifyImageView(ModelAPIView):
            model_name = "fruit_classifier"
            handler = "classify"

            def get_handler_input(self, request: HttpRequest) -> dict:
                image = request.FILES["image"]
                path = save_uploaded_file(image)
                return {"image_path": str(path)}
    """

    model_name: str | None = None
    handler: str | None = None

    def get_model_name(self, request: HttpRequest, **kwargs: Any) -> str:
        """Get the model name for this request.

        Override to dynamically determine model name.
        """
        if self.model_name is None:
            raise ValueError("model_name not set")
        return self.model_name

    def get_handler_name(self, request: HttpRequest, **kwargs: Any) -> str:
        """Get the handler name for this request.

        Override to dynamically determine handler name.
        """
        if self.handler is None:
            raise ValueError("handler not set")
        return self.handler

    def get_handler_input(self, request: HttpRequest, **kwargs: Any) -> dict[str, Any]:
        """Extract handler input from the request.

        Override this to customize input extraction.

        Default behavior:
        - GET: Use query parameters
        - POST with JSON: Parse JSON body
        - POST with form: Use form data
        """
        if request.method == "GET":
            return dict(request.GET.items())

        content_type = request.content_type or ""

        if "application/json" in content_type:
            try:
                return json.loads(request.body)
            except json.JSONDecodeError:
                return {}

        return dict(request.POST.items())

    def format_response(self, result: dict[str, Any]) -> dict[str, Any]:
        """Format the handler result for the response.

        Override to customize response format.
        """
        return result

    def handle_error(self, error: Exception) -> JsonResponse:
        """Handle errors and return appropriate response.

        Override to customize error handling.
        """
        if isinstance(error, ModelNotStartedError):
            return JsonResponse(
                {"error": "Model not running", "detail": str(error)},
                status=503,
            )
        elif isinstance(error, ModelRequestError):
            return JsonResponse(
                {"error": "Model request failed", "detail": str(error)},
                status=500,
            )
        elif isinstance(error, ModelManagerError):
            return JsonResponse(
                {"error": "Model error", "detail": str(error)},
                status=500,
            )
        else:
            logger.exception("Unexpected error in model view")
            return JsonResponse(
                {"error": "Internal error"},
                status=500,
            )

    def dispatch(self, request: HttpRequest, *args: Any, **kwargs: Any) -> JsonResponse:
        """Handle the request."""
        try:
            model_name = self.get_model_name(request, **kwargs)
            handler_name = self.get_handler_name(request, **kwargs)
            handler_input = self.get_handler_input(request, **kwargs)

            model = get_model(model_name)
            result = model.request(handler_name, handler_input)

            return JsonResponse(self.format_response(result))

        except Exception as e:
            return self.handle_error(e)


class GenericModelView(View):
    """Generic view that routes to any model and handler.

    Use with URL patterns like:
        path("api/models/<str:model_name>/<str:handler>/", GenericModelView.as_view())

    Requests:
        GET /api/models/fruit_classifier/classify/?image_path=/path/to/image
        POST /api/models/fruit_classifier/classify/
            {"image_path": "/path/to/image"}
    """

    def get_handler_input(self, request: HttpRequest) -> dict[str, Any]:
        """Extract handler input from request."""
        if request.method == "GET":
            return dict(request.GET.items())

        content_type = request.content_type or ""

        if "application/json" in content_type:
            try:
                return json.loads(request.body)
            except json.JSONDecodeError:
                return {}

        return dict(request.POST.items())

    def dispatch(
        self, request: HttpRequest, model_name: str, handler: str, **kwargs: Any
    ) -> JsonResponse:
        """Handle the request."""
        try:
            handler_input = self.get_handler_input(request)

            model = get_model(model_name)
            result = model.request(handler, handler_input)

            return JsonResponse(result)

        except KeyError:
            return JsonResponse(
                {"error": f"Model '{model_name}' not found"},
                status=404,
            )
        except ModelNotStartedError:
            return JsonResponse(
                {"error": f"Model '{model_name}' not running"},
                status=503,
            )
        except ModelRequestError as e:
            return JsonResponse(
                {"error": "Request failed", "detail": str(e)},
                status=500,
            )
        except Exception as e:
            logger.exception("Error in generic model view")
            return JsonResponse(
                {"error": "Internal error"},
                status=500,
            )


class ModelStatusView(View):
    """View to check status of all models."""

    def get(self, request: HttpRequest) -> JsonResponse:
        """Return status of all registered models."""
        registry = get_registry()
        return JsonResponse({"models": registry.status()})
