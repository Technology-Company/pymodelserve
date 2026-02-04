"""Django management command to serve ML models."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from pymodelserve.contrib.django.registry import get_registry, shutdown_models
from pymodelserve.contrib.django.settings import get_health_check_interval, get_models_dir
from pymodelserve.discovery.finder import discover_models
from pymodelserve.health.checker import HealthChecker


class Command(BaseCommand):
    """Management command to serve ML models."""

    help = "Start serving ML models with health monitoring"

    def add_arguments(self, parser: Any) -> None:
        """Add command arguments."""
        parser.add_argument(
            "--models-dir",
            type=str,
            help="Directory containing models (overrides settings)",
        )
        parser.add_argument(
            "--health-interval",
            type=int,
            default=None,
            help="Health check interval in seconds",
        )
        parser.add_argument(
            "--no-health-check",
            action="store_true",
            help="Disable health monitoring",
        )
        parser.add_argument(
            "--model",
            type=str,
            action="append",
            dest="models",
            help="Specific model name(s) to start",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Execute the command."""
        # Determine models directory
        models_dir = options.get("models_dir")
        if models_dir:
            models_dir = Path(models_dir)
        else:
            models_dir = get_models_dir()

        if models_dir is None:
            raise CommandError(
                "No models directory configured. "
                "Set MLSERVE['models_dir'] in settings or use --models-dir"
            )

        if not models_dir.is_dir():
            raise CommandError(f"Models directory not found: {models_dir}")

        # Discover and register models
        self.stdout.write(f"Discovering models in {models_dir}...")
        configs = discover_models(models_dir)

        if not configs:
            raise CommandError("No models found!")

        registry = get_registry()
        specific_models = options.get("models")

        for name, config in configs.items():
            if specific_models and name not in specific_models:
                continue
            registry.register(config=config)
            self.stdout.write(f"  Registered: {name}")

        # Start models
        self.stdout.write("\nStarting models...")
        results = registry.start_all()

        started = 0
        for name, error in results.items():
            if error is None:
                self.stdout.write(self.style.SUCCESS(f"  ✓ {name} started"))
                started += 1
            else:
                self.stdout.write(self.style.ERROR(f"  ✗ {name} failed: {error}"))

        if started == 0:
            raise CommandError("All models failed to start!")

        # Start health checker
        checker: HealthChecker | None = None
        if not options.get("no_health_check"):
            interval = options.get("health_interval") or get_health_check_interval()
            self.stdout.write(f"\nStarting health monitor (interval={interval}s)...")

            checker = HealthChecker(
                registry=registry,
                interval=interval,
                auto_restart=True,
                on_failure=self._on_health_failure,
                on_restart=self._on_model_restart,
            )
            checker.start()

        self.stdout.write(self.style.SUCCESS("\nModels are running. Press Ctrl+C to stop.\n"))

        # Print status
        self._print_status(registry)

        # Setup signal handlers
        def shutdown(sig: int, frame: Any) -> None:
            self.stdout.write("\nShutting down...")
            if checker:
                checker.stop()
            shutdown_models()
            self.stdout.write(self.style.SUCCESS("Done."))
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Wait
        signal.pause()

    def _on_health_failure(self, name: str, status: Any) -> None:
        """Called when a model fails health check."""
        self.stdout.write(
            self.style.WARNING(
                f"Health check failed for '{name}' ({status.consecutive_failures} failures)"
            )
        )

    def _on_model_restart(self, name: str) -> None:
        """Called when a model is restarted."""
        self.stdout.write(self.style.SUCCESS(f"Model '{name}' restarted"))

    def _print_status(self, registry: Any) -> None:
        """Print status table."""
        status = registry.status()

        self.stdout.write("\nModel Status:")
        self.stdout.write("-" * 60)

        for name, info in sorted(status.items()):
            running = "Running" if info["running"] else "Stopped"
            handlers = ", ".join(info["handlers"]) or "(none)"
            style = self.style.SUCCESS if info["running"] else self.style.ERROR
            self.stdout.write(f"  {name}: {style(running)} | v{info['version']} | {handlers}")

        self.stdout.write("-" * 60)
