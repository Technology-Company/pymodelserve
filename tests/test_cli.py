"""Tests for CLI commands."""

import pytest
from pathlib import Path
from click.testing import CliRunner

from pymodelserve.cli.commands import main


@pytest.fixture
def runner():
    """Create CLI runner."""
    return CliRunner()


class TestCLIBasic:
    """Basic CLI tests."""

    def test_help(self, runner):
        """Test --help flag."""
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "pymodelserve" in result.output
        assert "serve" in result.output
        assert "list" in result.output
        assert "init" in result.output

    def test_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCLIList:
    """Tests for 'pml list' command."""

    def test_list_empty_directory(self, runner, tmp_path):
        """Test listing empty directory."""
        result = runner.invoke(main, ["list", str(tmp_path)])

        assert result.exit_code == 0
        assert "No models found" in result.output

    def test_list_with_models(self, runner, tmp_path):
        """Test listing directory with models."""
        # Create model
        model_dir = tmp_path / "my_model"
        model_dir.mkdir()
        (model_dir / "model.yaml").write_text("""
name: test_model
version: "2.0.0"
client:
  module: model
  class: Client
handlers:
  - name: predict
""")

        result = runner.invoke(main, ["list", str(tmp_path)])

        assert result.exit_code == 0
        assert "test_model" in result.output
        assert "2.0.0" in result.output
        assert "predict" in result.output

    def test_list_multiple_models(self, runner, tmp_path):
        """Test listing multiple models."""
        for name in ["model_a", "model_b"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            (model_dir / "model.yaml").write_text(f"""
name: {name}
client:
  module: model
  class: Client
""")

        result = runner.invoke(main, ["list", str(tmp_path)])

        assert result.exit_code == 0
        assert "model_a" in result.output
        assert "model_b" in result.output

    def test_list_nonexistent_path(self, runner):
        """Test listing nonexistent path."""
        result = runner.invoke(main, ["list", "/nonexistent/path"])

        assert result.exit_code != 0


class TestCLIInit:
    """Tests for 'pml init' command."""

    def test_init_generic(self, runner, tmp_path):
        """Test init with generic framework."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init", "my_model"])

            assert result.exit_code == 0
            assert "Created model scaffold" in result.output

            # Check files created
            assert Path("my_model/model.yaml").exists()
            assert Path("my_model/model.py").exists()
            assert Path("my_model/requirements.txt").exists()
            assert Path("my_model/weights").is_dir()

            # Check content
            config = Path("my_model/model.yaml").read_text()
            assert "name: my_model" in config
            assert "predict" in config

    def test_init_tensorflow(self, runner, tmp_path):
        """Test init with TensorFlow framework."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init", "tf_model", "--framework", "tensorflow"])

            assert result.exit_code == 0

            model_py = Path("tf_model/model.py").read_text()
            assert "tensorflow" in model_py
            assert "classify" in model_py

            requirements = Path("tf_model/requirements.txt").read_text()
            assert "tensorflow" in requirements

    def test_init_pytorch(self, runner, tmp_path):
        """Test init with PyTorch framework."""
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init", "pt_model", "--framework", "pytorch"])

            assert result.exit_code == 0

            model_py = Path("pt_model/model.py").read_text()
            assert "torch" in model_py

            requirements = Path("pt_model/requirements.txt").read_text()
            assert "torch" in requirements

    def test_init_custom_output(self, runner, tmp_path):
        """Test init with custom output directory."""
        output = tmp_path / "custom" / "path"

        result = runner.invoke(main, ["init", "custom_model", "-o", str(output)])

        assert result.exit_code == 0
        assert (output / "model.yaml").exists()

    def test_init_existing_directory_fails(self, runner, tmp_path):
        """Test init fails if directory exists."""
        existing = tmp_path / "existing"
        existing.mkdir()

        result = runner.invoke(main, ["init", "model", "-o", str(existing)])

        assert result.exit_code != 0
        assert "already exists" in result.output


class TestCLIServe:
    """Tests for 'pml serve' command (limited - no actual serving)."""

    def test_serve_no_config(self, runner, tmp_path):
        """Test serve with missing config."""
        result = runner.invoke(main, ["serve", str(tmp_path)])

        assert result.exit_code != 0
        assert "No model.yaml found" in result.output

    def test_serve_all_no_models(self, runner, tmp_path):
        """Test serve --all with no models."""
        result = runner.invoke(main, ["serve", str(tmp_path), "--all"])

        assert result.exit_code != 0
        assert "No models found" in result.output


class TestCLIHealth:
    """Tests for 'pml health' command."""

    def test_health_no_config(self, runner, tmp_path):
        """Test health with missing config."""
        result = runner.invoke(main, ["health", str(tmp_path)])

        # Should fail because no model.yaml
        assert result.exit_code != 0


class TestCLITest:
    """Tests for 'pml test' command."""

    def test_test_no_config(self, runner, tmp_path):
        """Test test command with missing config."""
        result = runner.invoke(main, ["test", str(tmp_path)])

        assert result.exit_code != 0
