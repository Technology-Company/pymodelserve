"""Health monitoring for model processes."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pymodelserve.core.manager import ModelManager
    from pymodelserve.discovery.finder import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status for a model."""

    name: str
    healthy: bool
    last_check: datetime | None = None
    consecutive_failures: int = 0
    last_error: str | None = None
    response_time_ms: float | None = None


@dataclass
class HealthCheckResult:
    """Result of a single health check."""

    healthy: bool
    response_time_ms: float
    error: str | None = None


class HealthChecker:
    """Monitors health of model processes and handles recovery.

    Example:
        from pymodelserve import ModelRegistry
        from pymodelserve.health import HealthChecker

        registry = ModelRegistry()
        registry.register("fruit", "./models/fruit/")
        registry.start_all()

        # Start health monitoring
        checker = HealthChecker(registry, interval=30)
        checker.start()

        # ... application runs ...

        checker.stop()
        registry.stop_all()
    """

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        managers: list[ModelManager] | None = None,
        interval: float = 30.0,
        timeout: float = 5.0,
        max_failures: int = 3,
        auto_restart: bool = True,
        on_failure: Callable[[str, HealthStatus], None] | None = None,
        on_restart: Callable[[str], None] | None = None,
    ):
        """Initialize HealthChecker.

        Args:
            registry: ModelRegistry to monitor (alternative to managers list).
            managers: List of ModelManagers to monitor.
            interval: Seconds between health checks.
            timeout: Timeout for each health check.
            max_failures: Consecutive failures before restart.
            auto_restart: Automatically restart failed models.
            on_failure: Callback when a model fails health check.
            on_restart: Callback when a model is restarted.
        """
        if registry is None and managers is None:
            raise ValueError("Must provide either registry or managers")

        self._registry = registry
        self._managers: list[ModelManager] = managers or []
        self.interval = interval
        self.timeout = timeout
        self.max_failures = max_failures
        self.auto_restart = auto_restart
        self.on_failure = on_failure
        self.on_restart = on_restart

        self._status: dict[str, HealthStatus] = {}
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _get_managers(self) -> dict[str, ModelManager]:
        """Get all managers to check."""
        if self._registry is not None:
            return {name: self._registry.get(name) for name in self._registry.names}
        return {m.name: m for m in self._managers}

    def check_one(self, manager: ModelManager) -> HealthCheckResult:
        """Check health of a single model.

        Args:
            manager: ModelManager to check.

        Returns:
            HealthCheckResult with status.
        """
        start_time = time.time()

        try:
            healthy = manager.ping()
            response_time = (time.time() - start_time) * 1000

            return HealthCheckResult(
                healthy=healthy,
                response_time_ms=response_time,
                error=None if healthy else "Ping failed",
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                healthy=False,
                response_time_ms=response_time,
                error=str(e),
            )

    def check_all(self) -> dict[str, HealthStatus]:
        """Check health of all monitored models.

        Returns:
            Dictionary mapping model names to HealthStatus.
        """
        managers = self._get_managers()

        for name, manager in managers.items():
            result = self.check_one(manager)

            # Update status
            if name not in self._status:
                self._status[name] = HealthStatus(name=name, healthy=True)

            status = self._status[name]
            status.last_check = datetime.now()
            status.healthy = result.healthy
            status.response_time_ms = result.response_time_ms

            if result.healthy:
                status.consecutive_failures = 0
                status.last_error = None
            else:
                status.consecutive_failures += 1
                status.last_error = result.error

                logger.warning(
                    f"Health check failed for '{name}' "
                    f"({status.consecutive_failures}/{self.max_failures}): {result.error}"
                )

                # Trigger failure callback
                if self.on_failure:
                    try:
                        self.on_failure(name, status)
                    except Exception as e:
                        logger.error(f"Error in on_failure callback: {e}")

                # Handle restart
                if self.auto_restart and status.consecutive_failures >= self.max_failures:
                    self._restart_model(name, manager)

        return self._status.copy()

    def _restart_model(self, name: str, manager: ModelManager) -> None:
        """Restart a failed model.

        Args:
            name: Model name.
            manager: ModelManager instance.
        """
        logger.info(f"Restarting model '{name}' after {self.max_failures} failures")

        try:
            manager.restart()

            # Reset failure count on successful restart
            if name in self._status:
                self._status[name].consecutive_failures = 0
                self._status[name].healthy = True

            # Trigger restart callback
            if self.on_restart:
                try:
                    self.on_restart(name)
                except Exception as e:
                    logger.error(f"Error in on_restart callback: {e}")

        except Exception as e:
            logger.error(f"Failed to restart model '{name}': {e}")

    def _monitoring_loop(self) -> None:
        """Background thread for periodic health checks."""
        while not self._stop_event.is_set():
            try:
                self.check_all()
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

            # Wait for interval or stop event
            self._stop_event.wait(self.interval)

    def start(self) -> None:
        """Start background health monitoring."""
        if self._running:
            logger.warning("Health checker already running")
            return

        logger.info(
            f"Starting health checker (interval={self.interval}s, max_failures={self.max_failures})"
        )

        self._running = True
        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._monitoring_loop,
            name="pymodelserve-health",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Stop background health monitoring.

        Args:
            timeout: Time to wait for thread to stop.
        """
        if not self._running:
            return

        logger.info("Stopping health checker")
        self._running = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def get_status(self, name: str | None = None) -> HealthStatus | dict[str, HealthStatus]:
        """Get health status.

        Args:
            name: Specific model name, or None for all.

        Returns:
            HealthStatus for specific model, or dict of all statuses.
        """
        if name is not None:
            return self._status.get(
                name, HealthStatus(name=name, healthy=False, last_error="Not monitored")
            )
        return self._status.copy()

    @property
    def is_running(self) -> bool:
        """Check if health checker is running."""
        return self._running

    def __enter__(self) -> HealthChecker:
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()
