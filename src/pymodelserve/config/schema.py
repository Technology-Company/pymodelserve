"""Pydantic models for configuration validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ClientConfig(BaseModel):
    """Configuration for the model client entry point."""

    module: str = Field(description="Python module containing the client class")
    class_name: str = Field(alias="class", description="Name of the ModelClient subclass")

    model_config = {"populate_by_name": True}


class HandlerInputSchema(BaseModel):
    """Schema for handler input parameters."""

    # Dynamic fields - validation happens at runtime
    model_config = {"extra": "allow"}


class HandlerOutputSchema(BaseModel):
    """Schema for handler output."""

    model_config = {"extra": "allow"}


class HandlerConfig(BaseModel):
    """Configuration for a single handler."""

    name: str = Field(description="Handler name")
    input: dict[str, Any] = Field(default_factory=dict, description="Input schema")
    output: dict[str, Any] = Field(default_factory=dict, description="Output schema")


class HealthConfig(BaseModel):
    """Health check configuration."""

    interval: int = Field(default=30, description="Health check interval in seconds")
    timeout: int = Field(default=5, description="Health check timeout in seconds")
    max_failures: int = Field(default=3, description="Max failures before restart")


class ResourceConfig(BaseModel):
    """Resource limits configuration."""

    memory_limit: str | None = Field(default=None, description="Memory limit (e.g., '4G')")
    cpu_limit: int | None = Field(default=None, description="CPU core limit")
    gpu_ids: list[int] | None = Field(default=None, description="GPU IDs to use")


class ModelConfig(BaseModel):
    """Complete model configuration schema."""

    name: str = Field(description="Model name")
    version: str = Field(default="1.0.0", description="Model version")
    python: str = Field(default=">=3.11", description="Python version requirement")

    client: ClientConfig = Field(description="Client entry point configuration")
    requirements: str = Field(default="requirements.txt", description="Requirements file path")

    handlers: list[HandlerConfig] = Field(
        default_factory=list, description="Available handlers"
    )

    health: HealthConfig = Field(default_factory=HealthConfig, description="Health check config")
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Resource limits"
    )

    # Internal field set after loading
    model_dir: Path | None = Field(default=None, exclude=True)

    model_config = {"extra": "allow"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate model name is a valid identifier."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Model name must be alphanumeric with underscores or hyphens")
        return v

    def get_handler_names(self) -> list[str]:
        """Get list of handler names."""
        return [h.name for h in self.handlers]

    def get_handler(self, name: str) -> HandlerConfig | None:
        """Get handler config by name."""
        for handler in self.handlers:
            if handler.name == name:
                return handler
        return None

    def get_requirements_path(self) -> Path | None:
        """Get the full path to the requirements file."""
        if self.model_dir is None:
            return None
        return self.model_dir / self.requirements

    def get_client_module_path(self) -> Path | None:
        """Get the path to the client module."""
        if self.model_dir is None:
            return None
        return self.model_dir / f"{self.client.module}.py"
