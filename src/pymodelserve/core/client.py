"""ModelClient base class for model subprocess implementation."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

from pymodelserve.core.ipc import NamedPipeClient

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Registry for handler methods
_HANDLER_REGISTRY: dict[str, str] = {}


def handler(name: str) -> Callable[[F], F]:
    """Decorator to register a method as a message handler.

    Args:
        name: The message type this handler responds to.

    Example:
        class MyClient(ModelClient):
            @handler("classify")
            def classify(self, image_path: str) -> dict:
                return {"class": "cat", "confidence": 0.95}
    """

    def decorator(func: F) -> F:
        # Store handler name as function attribute
        setattr(func, "_handler_name", name)

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return func(self, *args, **kwargs)

        setattr(wrapper, "_handler_name", name)
        return wrapper  # type: ignore

    return decorator


class ModelClient:
    """Base class for model implementations that run in subprocess.

    Subclass this to create your model client. Implement handlers using
    either the @handler decorator or the handle_{message_type} naming pattern.

    Example:
        class FruitClassifier(ModelClient):
            def __init__(self):
                super().__init__()
                self.model = self.load_model()

            def load_model(self):
                import tensorflow as tf
                return tf.keras.models.load_model("weights/model.keras")

            @handler("classify")
            def classify(self, image_path: str) -> dict:
                # Process image and return prediction
                return {"class": "apple", "confidence": 0.92}

        if __name__ == "__main__":
            FruitClassifier().run()
    """

    def __init__(self) -> None:
        self._pipe_dir: str | None = os.environ.get("PYMODELSERVE_PIPE_DIR")
        self._ipc: NamedPipeClient | None = None
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._running = False

        # Discover and register handlers
        self._discover_handlers()

    def _discover_handlers(self) -> None:
        """Discover handler methods from class."""
        for name in dir(self):
            if name.startswith("_"):
                continue

            method = getattr(self, name, None)
            if not callable(method):
                continue

            # Check for @handler decorator
            handler_name = getattr(method, "_handler_name", None)
            if handler_name:
                self._handlers[handler_name] = method
                logger.debug(f"Registered handler: {handler_name} -> {name}")
                continue

            # Check for handle_{message} pattern
            if name.startswith("handle_"):
                handler_name = name[7:]  # Remove "handle_" prefix
                self._handlers[handler_name] = method
                logger.debug(f"Registered handler: {handler_name} -> {name}")

    def get_handlers(self) -> list[str]:
        """Get list of available handler names."""
        return list(self._handlers.keys())

    def handle_ping(self, **kwargs: Any) -> dict[str, Any]:
        """Built-in ping handler for health checks."""
        return {"status": "pong", "handlers": self.get_handlers()}

    def handle_shutdown(self, **kwargs: Any) -> dict[str, Any]:
        """Built-in shutdown handler."""
        self._running = False
        return {"status": "shutting_down"}

    def handle_message(self, message: str, data: dict[str, Any]) -> dict[str, Any]:
        """Dispatch message to appropriate handler.

        Args:
            message: The message type.
            data: Message data/payload.

        Returns:
            Handler response.
        """
        handler_func = self._handlers.get(message)

        if handler_func is None:
            return {
                "error": f"Unknown message type: {message}",
                "available_handlers": self.get_handlers(),
            }

        try:
            # Call handler with data dict unpacked as kwargs
            result = handler_func(**data)

            # Ensure result is a dict
            if not isinstance(result, dict):
                result = {"result": result}

            return result

        except TypeError as e:
            # Likely wrong arguments passed
            return {"error": f"Handler argument error: {e}"}
        except Exception as e:
            logger.exception(f"Handler error for {message}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def setup(self) -> None:
        """Called before starting the message loop.

        Override to perform any initialization needed after __init__.
        This is called after IPC is established but before processing messages.
        """
        pass

    def teardown(self) -> None:
        """Called after the message loop exits.

        Override to perform cleanup (close files, release resources, etc.).
        """
        pass

    def run(self) -> None:
        """Run the client message processing loop.

        This is the main entry point when the model subprocess starts.
        """
        if self._pipe_dir is None:
            logger.error("PYMODELSERVE_PIPE_DIR not set")
            sys.exit(1)

        logger.info(f"Starting model client, pipe_dir={self._pipe_dir}")

        try:
            self._ipc = NamedPipeClient(self._pipe_dir)
            self._ipc.connect()

            logger.info("IPC connected, calling setup()")
            self.setup()

            self._running = True
            logger.info("Entering message loop")

            while self._running:
                request = self._ipc.receive()

                if request is None:
                    logger.info("Pipe closed, exiting")
                    break

                message = request.get("message", "")
                data = request.get("data", {})

                logger.debug(f"Received: {message}")

                response = self.handle_message(message, data)

                self._ipc.send(response)

        except Exception as e:
            logger.exception("Model client error")
            sys.exit(1)
        finally:
            logger.info("Calling teardown()")
            try:
                self.teardown()
            except Exception:
                logger.exception("Error in teardown")

            if self._ipc is not None:
                self._ipc.close()

        logger.info("Model client exiting")


def run_client(client_class: type[ModelClient]) -> None:
    """Convenience function to run a client class.

    Args:
        client_class: The ModelClient subclass to instantiate and run.

    Example:
        from pymodelserve import ModelClient, run_client, handler

        class MyModel(ModelClient):
            @handler("predict")
            def predict(self, x: float) -> dict:
                return {"y": x * 2}

        if __name__ == "__main__":
            run_client(MyModel)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    client = client_class()
    client.run()
