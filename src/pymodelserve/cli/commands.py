"""CLI commands for pymodelserve (pml command)."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

console = Console()


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.version_option(package_name="pymodelserve")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """pymodelserve - Run ML models in isolated subprocess environments.

    Use 'pml COMMAND --help' for more information on a specific command.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--all", "serve_all", is_flag=True, help="Serve all models in directory")
@click.option("--health-interval", default=30, help="Health check interval in seconds")
@click.option("--no-health-check", is_flag=True, help="Disable health monitoring")
@click.pass_context
def serve(
    ctx: click.Context,
    model_path: str,
    serve_all: bool,
    health_interval: int,
    no_health_check: bool,
) -> None:
    """Start serving a model or all models in a directory.

    MODEL_PATH can be a model directory (containing model.yaml) or a
    directory containing multiple model subdirectories (use --all).

    Examples:

        pml serve ./models/fruit_classifier/

        pml serve ./models/ --all
    """
    from pymodelserve.discovery.finder import ModelRegistry, discover_models
    from pymodelserve.health.checker import HealthChecker

    model_path = Path(model_path)
    registry = ModelRegistry()

    if serve_all:
        console.print(f"[blue]Discovering models in {model_path}...[/blue]")
        configs = discover_models(model_path)

        if not configs:
            console.print("[red]No models found![/red]")
            sys.exit(1)

        for name, config in configs.items():
            registry.register(config=config)
            console.print(f"  [green]✓[/green] Found model: {name}")
    else:
        # Single model
        from pymodelserve.config.loader import find_config

        if find_config(model_path) is None:
            console.print(f"[red]No model.yaml found in {model_path}[/red]")
            sys.exit(1)

        registry.register(model_dir=model_path)

    # Start all models
    console.print("\n[blue]Starting models...[/blue]")
    results = registry.start_all()

    for name, error in results.items():
        if error is None:
            console.print(f"  [green]✓[/green] {name} started")
        else:
            console.print(f"  [red]✗[/red] {name} failed: {error}")

    failed = sum(1 for e in results.values() if e is not None)
    if failed == len(results):
        console.print("[red]All models failed to start![/red]")
        sys.exit(1)

    # Start health checker
    checker: HealthChecker | None = None
    if not no_health_check:
        console.print(f"\n[blue]Starting health monitor (interval={health_interval}s)...[/blue]")
        checker = HealthChecker(
            registry=registry,
            interval=health_interval,
            auto_restart=True,
        )
        checker.start()

    console.print("\n[green]Models are running. Press Ctrl+C to stop.[/green]\n")

    # Show status table
    _print_status_table(registry)

    # Wait for interrupt
    def shutdown(sig: int, frame: Any) -> None:
        console.print("\n[yellow]Shutting down...[/yellow]")
        if checker:
            checker.stop()
        registry.stop_all()
        console.print("[green]Done.[/green]")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Keep running
    signal.pause()


@main.command("list")
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive/--no-recursive", default=True, help="Search subdirectories")
def list_models(path: str, recursive: bool) -> None:
    """List all models found in a directory.

    Examples:

        pml list ./models/

        pml list ./models/ --no-recursive
    """
    from pymodelserve.discovery.finder import discover_models

    path = Path(path)
    console.print(f"[blue]Scanning {path} for models...[/blue]\n")

    configs = discover_models(path, recursive=recursive)

    if not configs:
        console.print("[yellow]No models found.[/yellow]")
        return

    table = Table(title="Discovered Models")
    table.add_column("Name", style="cyan")
    table.add_column("Version")
    table.add_column("Handlers")
    table.add_column("Path")

    for name, config in sorted(configs.items()):
        handlers = ", ".join(config.get_handler_names()) or "(none)"
        rel_path = str(config.model_dir.relative_to(path) if config.model_dir else "")
        table.add_row(name, config.version, handlers, rel_path)

    console.print(table)


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--timeout", default=5.0, help="Health check timeout in seconds")
def health(model_path: str, timeout: float) -> None:
    """Check health of a model.

    Examples:

        pml health ./models/fruit_classifier/
    """
    from pymodelserve.core.manager import ModelManager

    model_path = Path(model_path)

    console.print(f"[blue]Checking health of model at {model_path}...[/blue]\n")

    try:
        manager = ModelManager.from_dir(model_path)
        manager.start(timeout=30)

        if manager.ping():
            console.print("[green]✓ Model is healthy[/green]")
        else:
            console.print("[red]✗ Model failed health check[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        sys.exit(1)
    finally:
        if "manager" in locals():
            manager.stop()


@main.command()
@click.argument("name")
@click.option("--framework", type=click.Choice(["tensorflow", "pytorch", "generic"]), default="generic")
@click.option("--output", "-o", type=click.Path(), help="Output directory (default: ./NAME)")
def init(name: str, framework: str, output: str | None) -> None:
    """Create a new model scaffold.

    Creates a new model directory with model.yaml, model.py, and requirements.txt.

    Examples:

        pml init my_classifier --framework tensorflow

        pml init sentiment_model --framework pytorch -o ./models/sentiment/
    """
    output_dir = Path(output) if output else Path(f"./{name}")

    if output_dir.exists():
        console.print(f"[red]Directory already exists: {output_dir}[/red]")
        sys.exit(1)

    output_dir.mkdir(parents=True)

    # Create model.yaml
    handlers = "classify" if framework != "generic" else "predict"
    yaml_content = f"""name: {name}
version: "1.0.0"
python: ">=3.11"

client:
  module: model
  class: {name.title().replace('_', '')}Client

requirements: requirements.txt

handlers:
  - name: {handlers}
    input:
      data: object
    output:
      result: object

health:
  interval: 30
  timeout: 5
"""
    (output_dir / "model.yaml").write_text(yaml_content)

    # Create requirements.txt based on framework
    requirements = {
        "tensorflow": "tensorflow>=2.15\nnumpy>=1.24\nPillow>=10.0",
        "pytorch": "torch>=2.1\ntorchvision>=0.16\nnumpy>=1.24\nPillow>=10.0",
        "generic": "numpy>=1.24",
    }
    (output_dir / "requirements.txt").write_text(requirements[framework])

    # Create model.py
    class_name = f"{name.title().replace('_', '')}Client"

    if framework == "tensorflow":
        model_py = f'''"""Model client for {name}."""

from pymodelserve import ModelClient, handler


class {class_name}(ModelClient):
    """TensorFlow model client."""

    def setup(self) -> None:
        """Load the model on startup."""
        import tensorflow as tf

        # Load your model here
        # self.model = tf.keras.models.load_model("weights/model.keras")
        self.model = None  # Replace with actual model

    @handler("classify")
    def classify(self, data: dict) -> dict:
        """Classify input data.

        Args:
            data: Input data for classification.

        Returns:
            Classification result.
        """
        # Implement your classification logic here
        # predictions = self.model.predict(...)
        return {{"class": "example", "confidence": 0.95}}


if __name__ == "__main__":
    {class_name}().run()
'''
    elif framework == "pytorch":
        model_py = f'''"""Model client for {name}."""

from pymodelserve import ModelClient, handler


class {class_name}(ModelClient):
    """PyTorch model client."""

    def setup(self) -> None:
        """Load the model on startup."""
        import torch

        # Load your model here
        # self.model = torch.load("weights/model.pt")
        self.model = None  # Replace with actual model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @handler("classify")
    def classify(self, data: dict) -> dict:
        """Classify input data.

        Args:
            data: Input data for classification.

        Returns:
            Classification result.
        """
        # Implement your classification logic here
        # with torch.no_grad():
        #     output = self.model(...)
        return {{"class": "example", "confidence": 0.95}}


if __name__ == "__main__":
    {class_name}().run()
'''
    else:
        model_py = f'''"""Model client for {name}."""

from pymodelserve import ModelClient, handler


class {class_name}(ModelClient):
    """Generic model client."""

    def setup(self) -> None:
        """Initialize on startup."""
        # Load your model or initialize resources here
        pass

    @handler("predict")
    def predict(self, data: dict) -> dict:
        """Make a prediction.

        Args:
            data: Input data.

        Returns:
            Prediction result.
        """
        # Implement your prediction logic here
        return {{"result": data}}


if __name__ == "__main__":
    {class_name}().run()
'''

    (output_dir / "model.py").write_text(model_py)

    # Create weights directory
    (output_dir / "weights").mkdir()
    (output_dir / "weights" / ".gitkeep").write_text("")

    console.print(f"[green]✓ Created model scaffold at {output_dir}[/green]")
    console.print(f"\nNext steps:")
    console.print(f"  1. Add your model weights to {output_dir}/weights/")
    console.print(f"  2. Edit {output_dir}/model.py to load and use your model")
    console.print(f"  3. Run: pml serve {output_dir}")


@main.command()
@click.argument("model_path", type=click.Path(exists=True))
def test(model_path: str) -> None:
    """Test a model by starting it and running a health check.

    Examples:

        pml test ./models/fruit_classifier/
    """
    from pymodelserve.core.manager import ModelManager

    model_path = Path(model_path)

    console.print(f"[blue]Testing model at {model_path}...[/blue]\n")

    try:
        console.print("1. Loading configuration...")
        manager = ModelManager.from_dir(model_path)
        console.print(f"   [green]✓[/green] Config loaded: {manager.name}")

        console.print("2. Setting up virtual environment...")
        manager.setup_venv()
        console.print(f"   [green]✓[/green] Venv ready")

        console.print("3. Starting model...")
        manager.start(timeout=60)
        console.print(f"   [green]✓[/green] Model started")

        console.print("4. Running health check...")
        if manager.ping():
            console.print(f"   [green]✓[/green] Health check passed")
        else:
            console.print(f"   [red]✗[/red] Health check failed")
            sys.exit(1)

        console.print("\n[green]All tests passed![/green]")

    except Exception as e:
        console.print(f"\n[red]✗ Test failed: {e}[/red]")
        sys.exit(1)
    finally:
        if "manager" in locals():
            console.print("\n5. Stopping model...")
            manager.stop()
            console.print(f"   [green]✓[/green] Stopped")


def _print_status_table(registry: Any) -> None:
    """Print status table for registry."""
    table = Table(title="Running Models")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Version")
    table.add_column("Handlers")

    status = registry.status()
    for name, info in sorted(status.items()):
        running = "[green]Running[/green]" if info["running"] else "[red]Stopped[/red]"
        handlers = ", ".join(info["handlers"]) or "(none)"
        table.add_row(name, running, info["version"], handlers)

    console.print(table)


if __name__ == "__main__":
    main()
