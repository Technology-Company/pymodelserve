"""Tests for virtual environment management."""

import subprocess
import pytest
from pathlib import Path

from pymodelserve.core.venv import (
    VenvManager,
    VenvError,
    VenvCreationError,
    DependencyInstallError,
    ensure_venv,
)


class TestVenvManagerUnit:
    """Unit tests for VenvManager (no actual venv creation)."""

    def test_venv_paths(self, tmp_path):
        """Test venv path properties."""
        manager = VenvManager(tmp_path)
        assert manager.venv_dir == tmp_path / "model_venv"
        assert manager.python_path == tmp_path / "model_venv" / "bin" / "python"
        assert manager.pip_path == tmp_path / "model_venv" / "bin" / "pip"

    def test_custom_venv_name(self, tmp_path):
        """Test custom venv name."""
        manager = VenvManager(tmp_path, venv_name="custom_venv")
        assert manager.venv_dir == tmp_path / "custom_venv"

    def test_exists_false_initially(self, tmp_path):
        """Test that venv doesn't exist initially."""
        manager = VenvManager(tmp_path)
        assert not manager.exists

    def test_get_python_version_no_venv(self, tmp_path):
        """Test error when getting version without venv."""
        manager = VenvManager(tmp_path)
        with pytest.raises(VenvError):
            manager.get_python_version()

    def test_install_requirements_no_venv(self, tmp_path):
        """Test error when installing without venv."""
        manager = VenvManager(tmp_path)
        with pytest.raises(VenvError):
            manager.install_requirements(packages=["six"])


@pytest.mark.slow
class TestVenvManagerIntegration:
    """Integration tests for VenvManager (creates actual venvs)."""

    def test_create_venv(self, tmp_path):
        """Test creating a virtual environment."""
        manager = VenvManager(tmp_path)
        manager.create()

        assert manager.exists
        assert manager.venv_dir.exists()
        assert manager.python_path.exists()
        assert manager.pip_path.exists()

    def test_create_venv_idempotent(self, tmp_path):
        """Test that creating venv twice is safe."""
        manager = VenvManager(tmp_path)
        manager.create()
        manager.create()  # Should not raise

        assert manager.exists

    def test_create_venv_force(self, tmp_path):
        """Test force recreation of venv."""
        manager = VenvManager(tmp_path)
        manager.create()

        # Add a marker file
        marker = manager.venv_dir / "marker.txt"
        marker.write_text("test")

        # Force recreate
        manager.create(force=True)

        # Marker should be gone
        assert not marker.exists()
        assert manager.exists

    def test_get_python_version(self, tmp_path):
        """Test getting Python version from venv."""
        manager = VenvManager(tmp_path)
        manager.create()

        version = manager.get_python_version()
        assert "Python" in version
        assert "3." in version

    def test_install_requirements(self, tmp_path):
        """Test installing from requirements file."""
        manager = VenvManager(tmp_path)
        manager.create()

        # Create a simple requirements file
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("six>=1.0\n")

        manager.install_requirements(requirements_file=req_file)

        # Verify installation
        result = subprocess.run(
            [str(manager.pip_path), "show", "six"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "six" in result.stdout

    def test_install_packages(self, tmp_path):
        """Test installing packages directly."""
        manager = VenvManager(tmp_path)
        manager.create()

        manager.install_requirements(packages=["six>=1.0"])

        result = subprocess.run(
            [str(manager.pip_path), "show", "six"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_install_package_single(self, tmp_path):
        """Test installing a single package."""
        manager = VenvManager(tmp_path)
        manager.create()

        manager.install_package("six")

        result = subprocess.run(
            [str(manager.pip_path), "show", "six"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_install_missing_requirements_file(self, tmp_path):
        """Test error when requirements file doesn't exist."""
        manager = VenvManager(tmp_path)
        manager.create()

        with pytest.raises(DependencyInstallError):
            manager.install_requirements(requirements_file="nonexistent.txt")

    def test_run_script(self, tmp_path):
        """Test running a script in the venv."""
        manager = VenvManager(tmp_path)
        manager.create()

        # Create a simple script
        script = tmp_path / "test_script.py"
        script.write_text('print("hello from script")')

        process = manager.run_script(script)
        stdout, stderr = process.communicate(timeout=30)

        assert process.returncode == 0
        assert "hello from script" in stdout

    def test_run_module(self, tmp_path):
        """Test running a module in the venv."""
        manager = VenvManager(tmp_path)
        manager.create()

        # Run the json.tool module as a test
        process = manager.run_module("json.tool", args=["--help"])
        stdout, stderr = process.communicate(timeout=30)

        # json.tool --help returns 0
        assert process.returncode == 0


@pytest.mark.slow
class TestEnsureVenv:
    """Tests for ensure_venv helper."""

    def test_ensure_venv_creates(self, tmp_path):
        """Test ensure_venv creates venv if missing."""
        # Create requirements file
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("# empty\n")

        manager = ensure_venv(tmp_path)

        assert manager.exists
        assert manager.venv_dir == tmp_path / "model_venv"

    def test_ensure_venv_reuses(self, tmp_path):
        """Test ensure_venv reuses existing venv."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("# empty\n")

        # Create first
        manager1 = ensure_venv(tmp_path)
        marker = manager1.venv_dir / "marker.txt"
        marker.write_text("test")

        # Should reuse
        manager2 = ensure_venv(tmp_path)

        assert marker.exists()  # Marker still there = venv was reused
