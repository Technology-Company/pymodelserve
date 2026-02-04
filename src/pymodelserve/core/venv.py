"""Virtual environment management for isolated model execution."""

from __future__ import annotations

import logging
import subprocess
import venv
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)


class VenvError(Exception):
    """Base exception for virtual environment errors."""


class VenvCreationError(VenvError):
    """Raised when venv creation fails."""


class DependencyInstallError(VenvError):
    """Raised when dependency installation fails."""


class VenvManager:
    """Manages virtual environments for model isolation."""

    DEFAULT_VENV_NAME = "model_venv"

    def __init__(
        self,
        model_dir: Path,
        venv_name: str = DEFAULT_VENV_NAME,
        python_version: str | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.venv_name = venv_name
        self.python_version = python_version
        self._venv_dir: Path | None = None

    @property
    def venv_dir(self) -> Path:
        """Get the virtual environment directory."""
        if self._venv_dir is None:
            self._venv_dir = self.model_dir / self.venv_name
        return self._venv_dir

    @property
    def python_path(self) -> Path:
        """Get the path to the Python executable in the venv."""
        return self.venv_dir / "bin" / "python"

    @property
    def pip_path(self) -> Path:
        """Get the path to the pip executable in the venv."""
        return self.venv_dir / "bin" / "pip"

    @property
    def exists(self) -> bool:
        """Check if the virtual environment exists."""
        return self.venv_dir.exists() and self.python_path.exists()

    def create(self, force: bool = False) -> None:
        """Create the virtual environment.

        Args:
            force: If True, recreate even if it exists.
        """
        if self.exists and not force:
            logger.debug(f"Venv already exists at {self.venv_dir}")
            return

        if self.exists and force:
            logger.info(f"Removing existing venv at {self.venv_dir}")
            import shutil

            shutil.rmtree(self.venv_dir)

        logger.info(f"Creating virtual environment at {self.venv_dir}")

        try:
            # Use the venv module to create the virtual environment
            venv.create(
                self.venv_dir,
                with_pip=True,
                clear=force,
                symlinks=True,
            )
        except Exception as e:
            raise VenvCreationError(f"Failed to create venv: {e}") from e

        # Upgrade pip to latest version
        self._run_pip(["install", "--upgrade", "pip"])

    def install_requirements(
        self,
        requirements_file: Path | str | None = None,
        packages: Sequence[str] | None = None,
    ) -> None:
        """Install dependencies from requirements file or package list.

        Args:
            requirements_file: Path to requirements.txt file.
            packages: List of packages to install.
        """
        if not self.exists:
            raise VenvError("Virtual environment does not exist. Call create() first.")

        if requirements_file is not None:
            req_path = Path(requirements_file)
            if not req_path.is_absolute():
                req_path = self.model_dir / req_path

            if not req_path.exists():
                raise DependencyInstallError(f"Requirements file not found: {req_path}")

            logger.info(f"Installing requirements from {req_path}")
            self._run_pip(["install", "-r", str(req_path)])

        if packages:
            logger.info(f"Installing packages: {packages}")
            self._run_pip(["install"] + list(packages))

    def install_package(self, package: str) -> None:
        """Install a single package.

        Args:
            package: Package specification (e.g., "numpy>=1.0").
        """
        self.install_requirements(packages=[package])

    def _run_pip(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run a pip command in the virtual environment."""
        cmd = [str(self.pip_path)] + args

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=self.model_dir,
            )
            logger.debug(f"pip output: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"pip error: {e.stderr}")
            raise DependencyInstallError(f"pip command failed: {e.stderr}") from e

    def get_python_version(self) -> str:
        """Get the Python version in the venv."""
        if not self.exists:
            raise VenvError("Virtual environment does not exist")

        result = subprocess.run(
            [str(self.python_path), "--version"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def run_script(
        self,
        script: Path | str,
        args: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen[str]:
        """Run a Python script in the virtual environment.

        Args:
            script: Path to the script to run.
            args: Additional arguments for the script.
            env: Environment variables to set.

        Returns:
            Popen object for the running process.
        """
        if not self.exists:
            raise VenvError("Virtual environment does not exist")

        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = self.model_dir / script_path

        cmd = [str(self.python_path), str(script_path)]
        if args:
            cmd.extend(args)

        # Build environment, inheriting from parent
        import os

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.model_dir,
            env=run_env,
        )

    def run_module(
        self,
        module: str,
        args: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.Popen[str]:
        """Run a Python module in the virtual environment.

        Args:
            module: Module name to run (e.g., "model").
            args: Additional arguments.
            env: Environment variables to set.

        Returns:
            Popen object for the running process.
        """
        if not self.exists:
            raise VenvError("Virtual environment does not exist")

        cmd = [str(self.python_path), "-m", module]
        if args:
            cmd.extend(args)

        import os

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.model_dir,
            env=run_env,
        )


def ensure_venv(
    model_dir: Path,
    requirements_file: str | Path = "requirements.txt",
    venv_name: str = VenvManager.DEFAULT_VENV_NAME,
) -> VenvManager:
    """Ensure a virtual environment exists with dependencies installed.

    This is a convenience function that creates a venv if needed and
    installs requirements.

    Args:
        model_dir: Directory containing the model.
        requirements_file: Path to requirements.txt (relative to model_dir).
        venv_name: Name for the venv directory.

    Returns:
        VenvManager instance.
    """
    manager = VenvManager(model_dir, venv_name=venv_name)

    if not manager.exists:
        manager.create()
        manager.install_requirements(requirements_file=requirements_file)
    else:
        logger.debug(f"Using existing venv at {manager.venv_dir}")

    return manager
