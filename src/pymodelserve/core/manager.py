"""ModelManager for managing model subprocesses."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from pymodelserve.config.loader import load_config, load_config_from_dir
from pymodelserve.config.schema import ModelConfig
from pymodelserve.core.ipc import IPCError, NamedPipeServer
from pymodelserve.core.venv import VenvManager

logger = logging.getLogger(__name__)


class ModelManagerError(Exception):
    """Base exception for ModelManager errors."""


class ModelNotStartedError(ModelManagerError):
    """Raised when trying to use a model that hasn't been started."""


class ModelStartupError(ModelManagerError):
    """Raised when a model fails to start."""


class ModelRequestError(ModelManagerError):
    """Raised when a request to the model fails."""


class ModelManager:
    """Manages a model running in an isolated subprocess.

    The ModelManager handles:
    - Virtual environment creation and dependency installation
    - Subprocess spawning with IPC via named pipes
    - Health checking and automatic restart
    - Graceful shutdown

    Example:
        # Using context manager (recommended)
        with ModelManager.from_yaml("./models/fruit/model.yaml") as model:
            result = model.request("classify", {"image_path": "/path/to/image.jpg"})

        # Manual lifecycle management
        manager = ModelManager.from_yaml("./models/fruit/model.yaml")
        manager.start()
        result = manager.request("classify", {"image_path": "/path/to/image.jpg"})
        manager.stop()
    """

    def __init__(
        self,
        config: ModelConfig,
        auto_setup_venv: bool = True,
    ):
        """Initialize ModelManager.

        Args:
            config: Model configuration.
            auto_setup_venv: Automatically create venv and install deps on start.
        """
        self.config = config
        self.auto_setup_venv = auto_setup_venv

        self._venv: VenvManager | None = None
        self._ipc: NamedPipeServer | None = None
        self._process: subprocess.Popen[str] | None = None
        self._is_started = False
        self._stderr_thread: threading.Thread | None = None
        self._stderr_output: list[str] = []

    @classmethod
    def from_yaml(cls, config_path: str | Path, **kwargs: Any) -> ModelManager:
        """Create ModelManager from a YAML config file.

        Args:
            config_path: Path to model.yaml file.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            ModelManager instance.
        """
        config = load_config(Path(config_path))
        return cls(config, **kwargs)

    @classmethod
    def from_dir(cls, model_dir: str | Path, **kwargs: Any) -> ModelManager:
        """Create ModelManager from a model directory.

        Looks for model.yaml, model.yml, or model.toml in the directory.

        Args:
            model_dir: Path to directory containing model and config.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            ModelManager instance.
        """
        config = load_config_from_dir(Path(model_dir))
        return cls(config, **kwargs)

    @classmethod
    def from_config(
        cls, config_dict: dict[str, Any], model_dir: str | Path, **kwargs: Any
    ) -> ModelManager:
        """Create ModelManager from a config dictionary.

        Args:
            config_dict: Configuration dictionary.
            model_dir: Path to model directory.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            ModelManager instance.
        """
        config = load_config(config_dict, model_dir=Path(model_dir))
        return cls(config, **kwargs)

    @property
    def name(self) -> str:
        """Model name from config."""
        return self.config.name

    @property
    def model_dir(self) -> Path:
        """Model directory."""
        if self.config.model_dir is None:
            raise ModelManagerError("Model directory not set")
        return self.config.model_dir

    @property
    def is_running(self) -> bool:
        """Check if the model process is running."""
        return self._is_started and self._process is not None and self._process.poll() is None

    def setup_venv(self, force: bool = False) -> VenvManager:
        """Set up virtual environment and install dependencies.

        Args:
            force: Force recreation of venv even if it exists.

        Returns:
            VenvManager instance.
        """
        logger.info(f"Setting up venv for model '{self.name}'")

        self._venv = VenvManager(self.model_dir)

        if not self._venv.exists or force:
            self._venv.create(force=force)

            req_path = self.config.get_requirements_path()
            if req_path and req_path.exists():
                self._venv.install_requirements(requirements_file=req_path)
            else:
                logger.warning(f"No requirements file found at {req_path}")

        return self._venv

    def start(self, timeout: float = 30.0) -> None:
        """Start the model subprocess.

        Args:
            timeout: Timeout in seconds for startup (including health check).

        Raises:
            ModelStartupError: If the model fails to start.
        """
        if self._is_started:
            logger.warning(f"Model '{self.name}' already started")
            return

        logger.info(f"Starting model '{self.name}'")

        # Setup venv if needed
        if self.auto_setup_venv:
            self.setup_venv()
        elif self._venv is None:
            self._venv = VenvManager(self.model_dir)

        if not self._venv.exists:
            raise ModelStartupError("Virtual environment not found. Call setup_venv() first.")

        # Setup IPC
        self._ipc = NamedPipeServer()
        pipe_config = self._ipc.setup()

        # Build environment for subprocess
        env = os.environ.copy()
        env["PYMODELSERVE_PIPE_DIR"] = str(pipe_config.pipe_dir)

        # GPU configuration if specified
        if self.config.resources.gpu_ids is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.config.resources.gpu_ids)

        # Start subprocess
        client_module = self.config.client.module
        logger.info(f"Spawning subprocess for module '{client_module}'")

        self._process = self._venv.run_module(
            client_module,
            env=env,
        )

        # Start stderr reader thread
        self._stderr_output = []
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            daemon=True,
        )
        self._stderr_thread.start()

        # Connect IPC (this blocks until subprocess opens pipes)
        try:
            self._ipc.connect()
        except Exception as e:
            self._cleanup()
            raise ModelStartupError(f"Failed to connect IPC: {e}") from e

        self._is_started = True

        # Verify startup with ping
        try:
            start_time = time.time()
            response = self._ipc.request("ping", {})

            if response.get("status") != "pong":
                raise ModelStartupError(f"Unexpected ping response: {response}")

            elapsed = time.time() - start_time
            logger.info(
                f"Model '{self.name}' started successfully in {elapsed:.2f}s. "
                f"Available handlers: {response.get('handlers', [])}"
            )

        except Exception as e:
            self._cleanup()
            stderr = "\n".join(self._stderr_output)
            raise ModelStartupError(
                f"Model failed to respond to ping: {e}\nStderr: {stderr}"
            ) from e

    def _read_stderr(self) -> None:
        """Read stderr from subprocess in background thread."""
        if self._process is None or self._process.stderr is None:
            return

        for line in self._process.stderr:
            line = line.rstrip()
            self._stderr_output.append(line)
            logger.debug(f"[{self.name}] {line}")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the model subprocess gracefully.

        Args:
            timeout: Timeout in seconds to wait for graceful shutdown.
        """
        if not self._is_started:
            return

        logger.info(f"Stopping model '{self.name}'")

        # Send shutdown message
        if self._ipc is not None and self.is_running:
            try:
                self._ipc.request("shutdown", {})
            except Exception as e:
                logger.warning(f"Error sending shutdown: {e}")

        # Wait for process to exit
        if self._process is not None:
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(f"Model '{self.name}' didn't exit gracefully, terminating")
                self._process.terminate()
                try:
                    self._process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Model '{self.name}' didn't terminate, killing")
                    self._process.kill()

        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_started = False

        if self._ipc is not None:
            self._ipc.close()
            self._ipc = None

        self._process = None

    def request(
        self,
        handler: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a request to the model.

        Args:
            handler: Handler name to invoke.
            data: Request data (passed as kwargs to handler).

        Returns:
            Handler response.

        Raises:
            ModelNotStartedError: If model is not running.
            ModelRequestError: If request fails.
        """
        if not self._is_started:
            raise ModelNotStartedError("Model not started. Call start() first.")

        if not self.is_running:
            stderr = "\n".join(self._stderr_output[-20:])
            raise ModelRequestError(f"Model process died unexpectedly.\nLast stderr:\n{stderr}")

        if self._ipc is None:
            raise ModelNotStartedError("IPC not initialized")

        try:
            response = self._ipc.request(handler, data or {})

            if "error" in response:
                raise ModelRequestError(f"Handler error: {response['error']}")

            return response

        except IPCError as e:
            raise ModelRequestError(f"IPC error: {e}") from e

    def ping(self) -> bool:
        """Check if the model is healthy.

        Returns:
            True if model responds to ping, False otherwise.
        """
        if not self.is_running:
            return False

        try:
            response = self.request("ping", {})
            return response.get("status") == "pong"
        except Exception:
            return False

    def restart(self) -> None:
        """Restart the model subprocess."""
        logger.info(f"Restarting model '{self.name}'")
        self.stop()
        self.start()

    def __enter__(self) -> ModelManager:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"<ModelManager name='{self.name}' status='{status}'>"
