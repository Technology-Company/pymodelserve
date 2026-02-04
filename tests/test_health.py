"""Tests for health monitoring."""

import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from pymodelserve.health.checker import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
)


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_default_values(self):
        """Test default values."""
        status = HealthStatus(name="test", healthy=True)

        assert status.name == "test"
        assert status.healthy is True
        assert status.last_check is None
        assert status.consecutive_failures == 0
        assert status.last_error is None
        assert status.response_time_ms is None

    def test_with_all_values(self):
        """Test with all values set."""
        now = datetime.now()
        status = HealthStatus(
            name="model",
            healthy=False,
            last_check=now,
            consecutive_failures=3,
            last_error="timeout",
            response_time_ms=150.5,
        )

        assert status.consecutive_failures == 3
        assert status.last_error == "timeout"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_healthy_result(self):
        """Test healthy result."""
        result = HealthCheckResult(healthy=True, response_time_ms=10.5)

        assert result.healthy is True
        assert result.response_time_ms == 10.5
        assert result.error is None

    def test_unhealthy_result(self):
        """Test unhealthy result."""
        result = HealthCheckResult(
            healthy=False,
            response_time_ms=5000.0,
            error="timeout",
        )

        assert result.healthy is False
        assert result.error == "timeout"


class TestHealthChecker:
    """Tests for HealthChecker."""

    def test_init_requires_registry_or_managers(self):
        """Test that init requires registry or managers."""
        with pytest.raises(ValueError, match="Must provide"):
            HealthChecker()

    def test_init_with_managers(self):
        """Test init with managers list."""
        manager = MagicMock()
        manager.name = "test"

        checker = HealthChecker(managers=[manager])

        assert not checker.is_running

    def test_init_with_registry(self):
        """Test init with registry."""
        registry = MagicMock()
        registry.names = ["model1", "model2"]

        checker = HealthChecker(registry=registry)

        assert not checker.is_running

    def test_check_one_healthy(self):
        """Test checking a healthy model."""
        manager = MagicMock()
        manager.name = "healthy_model"
        manager.ping.return_value = True

        checker = HealthChecker(managers=[manager])
        result = checker.check_one(manager)

        assert result.healthy is True
        assert result.error is None
        assert result.response_time_ms >= 0

    def test_check_one_unhealthy(self):
        """Test checking an unhealthy model."""
        manager = MagicMock()
        manager.name = "unhealthy_model"
        manager.ping.return_value = False

        checker = HealthChecker(managers=[manager])
        result = checker.check_one(manager)

        assert result.healthy is False
        assert result.error == "Ping failed"

    def test_check_one_exception(self):
        """Test checking a model that throws exception."""
        manager = MagicMock()
        manager.name = "error_model"
        manager.ping.side_effect = Exception("connection failed")

        checker = HealthChecker(managers=[manager])
        result = checker.check_one(manager)

        assert result.healthy is False
        assert "connection failed" in result.error

    def test_check_all_updates_status(self):
        """Test that check_all updates status."""
        manager = MagicMock()
        manager.name = "status_model"
        manager.ping.return_value = True

        checker = HealthChecker(managers=[manager])
        status = checker.check_all()

        assert "status_model" in status
        assert status["status_model"].healthy is True
        assert status["status_model"].last_check is not None

    def test_check_all_tracks_failures(self):
        """Test that failures are tracked."""
        manager = MagicMock()
        manager.name = "failing_model"
        manager.ping.return_value = False

        checker = HealthChecker(managers=[manager], auto_restart=False)

        # First failure
        checker.check_all()
        assert checker._status["failing_model"].consecutive_failures == 1

        # Second failure
        checker.check_all()
        assert checker._status["failing_model"].consecutive_failures == 2

    def test_check_all_resets_failures_on_success(self):
        """Test that success resets failure count."""
        manager = MagicMock()
        manager.name = "recovering_model"
        manager.ping.return_value = False

        checker = HealthChecker(managers=[manager], auto_restart=False)

        # Two failures
        checker.check_all()
        checker.check_all()
        assert checker._status["recovering_model"].consecutive_failures == 2

        # Recovery
        manager.ping.return_value = True
        checker.check_all()
        assert checker._status["recovering_model"].consecutive_failures == 0
        assert checker._status["recovering_model"].healthy is True

    def test_auto_restart_on_max_failures(self):
        """Test auto restart after max failures."""
        manager = MagicMock()
        manager.name = "restart_model"
        manager.ping.return_value = False

        checker = HealthChecker(
            managers=[manager],
            max_failures=2,
            auto_restart=True,
        )

        # First failure - no restart
        checker.check_all()
        manager.restart.assert_not_called()

        # Second failure - should restart
        checker.check_all()
        manager.restart.assert_called_once()

    def test_no_restart_when_disabled(self):
        """Test no restart when auto_restart=False."""
        manager = MagicMock()
        manager.name = "no_restart_model"
        manager.ping.return_value = False

        checker = HealthChecker(
            managers=[manager],
            max_failures=1,
            auto_restart=False,
        )

        checker.check_all()
        checker.check_all()
        checker.check_all()

        manager.restart.assert_not_called()

    def test_on_failure_callback(self):
        """Test on_failure callback."""
        manager = MagicMock()
        manager.name = "callback_model"
        manager.ping.return_value = False

        callback_calls = []

        def on_failure(name, status):
            callback_calls.append((name, status.consecutive_failures))

        checker = HealthChecker(
            managers=[manager],
            auto_restart=False,
            on_failure=on_failure,
        )

        checker.check_all()
        checker.check_all()

        assert len(callback_calls) == 2
        assert callback_calls[0] == ("callback_model", 1)
        assert callback_calls[1] == ("callback_model", 2)

    def test_on_restart_callback(self):
        """Test on_restart callback."""
        manager = MagicMock()
        manager.name = "restart_callback_model"
        manager.ping.return_value = False

        restart_calls = []

        def on_restart(name):
            restart_calls.append(name)

        checker = HealthChecker(
            managers=[manager],
            max_failures=1,
            auto_restart=True,
            on_restart=on_restart,
        )

        checker.check_all()

        assert restart_calls == ["restart_callback_model"]

    def test_get_status_specific_model(self):
        """Test getting status for specific model."""
        manager = MagicMock()
        manager.name = "specific_model"
        manager.ping.return_value = True

        checker = HealthChecker(managers=[manager])
        checker.check_all()

        status = checker.get_status("specific_model")

        assert status.name == "specific_model"
        assert status.healthy is True

    def test_get_status_all_models(self):
        """Test getting status for all models."""
        managers = [MagicMock(), MagicMock()]
        managers[0].name = "model_a"
        managers[0].ping.return_value = True
        managers[1].name = "model_b"
        managers[1].ping.return_value = False

        checker = HealthChecker(managers=managers, auto_restart=False)
        checker.check_all()

        status = checker.get_status()

        assert "model_a" in status
        assert "model_b" in status
        assert status["model_a"].healthy is True
        assert status["model_b"].healthy is False

    def test_get_status_unmonitored_model(self):
        """Test getting status for unmonitored model."""
        checker = HealthChecker(managers=[MagicMock()])

        status = checker.get_status("unknown_model")

        assert status.name == "unknown_model"
        assert status.healthy is False
        assert "Not monitored" in status.last_error

    def test_start_stop(self):
        """Test starting and stopping the checker."""
        manager = MagicMock()
        manager.name = "test"
        manager.ping.return_value = True

        checker = HealthChecker(managers=[manager], interval=0.1)

        assert not checker.is_running

        checker.start()
        assert checker.is_running

        time.sleep(0.15)  # Let it run a check

        checker.stop()
        assert not checker.is_running

    def test_context_manager(self):
        """Test context manager usage."""
        manager = MagicMock()
        manager.name = "ctx_model"
        manager.ping.return_value = True

        with HealthChecker(managers=[manager], interval=1) as checker:
            assert checker.is_running

        assert not checker.is_running
