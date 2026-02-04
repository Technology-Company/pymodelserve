"""Inter-process communication using named pipes (FIFOs)."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


class IPCError(Exception):
    """Base exception for IPC errors."""


class PipeNotConnectedError(IPCError):
    """Raised when attempting to use a pipe that is not connected."""


class PipeTimeoutError(IPCError):
    """Raised when a pipe operation times out."""


@dataclass
class PipeConfig:
    """Configuration for named pipe communication."""

    pipe_dir: Path
    pipe_in_name: str = "pipe_in"
    pipe_out_name: str = "pipe_out"

    @property
    def pipe_in_path(self) -> Path:
        return self.pipe_dir / self.pipe_in_name

    @property
    def pipe_out_path(self) -> Path:
        return self.pipe_dir / self.pipe_out_name


class NamedPipeServer:
    """Server-side named pipe handler (runs in parent/manager process).

    The server writes to pipe_in (which the client reads) and reads from
    pipe_out (which the client writes to).
    """

    def __init__(self, config: PipeConfig | None = None):
        self._config = config
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._pipe_in: IO[str] | None = None
        self._pipe_out: IO[str] | None = None
        self._is_connected = False

    @property
    def config(self) -> PipeConfig:
        if self._config is None:
            raise PipeNotConnectedError("Pipe server not initialized")
        return self._config

    @property
    def pipe_dir(self) -> Path:
        return self.config.pipe_dir

    def setup(self) -> PipeConfig:
        """Create temporary directory and named pipes."""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="pymodelserve_")
        pipe_dir = Path(self._temp_dir.name)

        self._config = PipeConfig(pipe_dir=pipe_dir)

        # Create named pipes (FIFOs)
        os.mkfifo(self.config.pipe_in_path)
        os.mkfifo(self.config.pipe_out_path)

        return self._config

    def connect(self) -> None:
        """Open pipes for communication. Must be called after subprocess starts."""
        if self._config is None:
            raise PipeNotConnectedError("Call setup() before connect()")

        # Open pipe_in for writing (to send messages to client)
        # Open pipe_out for reading (to receive messages from client)
        # Note: These will block until the client opens the other ends
        self._pipe_in = open(self.config.pipe_in_path, "w")
        self._pipe_out = open(self.config.pipe_out_path, "r")
        self._is_connected = True

    def send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the client."""
        if not self._is_connected or self._pipe_in is None:
            raise PipeNotConnectedError("Pipe not connected")

        json_str = json.dumps(message)
        self._pipe_in.write(json_str + "\n")
        self._pipe_in.flush()

    def receive(self) -> dict[str, Any]:
        """Receive a JSON message from the client."""
        if not self._is_connected or self._pipe_out is None:
            raise PipeNotConnectedError("Pipe not connected")

        line = self._pipe_out.readline()
        if not line:
            raise IPCError("Connection closed by client")

        return json.loads(line)

    def request(self, message_type: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request and wait for response."""
        self.send({"message": message_type, "data": data or {}})
        return self.receive()

    def close(self) -> None:
        """Close pipes and cleanup."""
        self._is_connected = False

        if self._pipe_in is not None:
            try:
                self._pipe_in.close()
            except Exception:
                pass
            self._pipe_in = None

        if self._pipe_out is not None:
            try:
                self._pipe_out.close()
            except Exception:
                pass
            self._pipe_out = None

        if self._temp_dir is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

        self._config = None

    def __enter__(self) -> NamedPipeServer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class NamedPipeClient:
    """Client-side named pipe handler (runs in subprocess/model process).

    The client reads from pipe_in (which the server writes to) and writes
    to pipe_out (which the server reads from).
    """

    def __init__(self, pipe_dir: str | Path):
        self._config = PipeConfig(pipe_dir=Path(pipe_dir))
        self._pipe_in: IO[str] | None = None
        self._pipe_out: IO[str] | None = None
        self._is_connected = False

    @property
    def config(self) -> PipeConfig:
        return self._config

    def connect(self) -> None:
        """Open pipes for communication."""
        # Open pipe_in for reading (to receive messages from server)
        # Open pipe_out for writing (to send messages to server)
        self._pipe_in = open(self.config.pipe_in_path, "r")
        self._pipe_out = open(self.config.pipe_out_path, "w")
        self._is_connected = True

    def send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the server."""
        if not self._is_connected or self._pipe_out is None:
            raise PipeNotConnectedError("Pipe not connected")

        json_str = json.dumps(message)
        self._pipe_out.write(json_str + "\n")
        self._pipe_out.flush()

    def receive(self) -> dict[str, Any] | None:
        """Receive a JSON message from the server. Returns None on EOF."""
        if not self._is_connected or self._pipe_in is None:
            raise PipeNotConnectedError("Pipe not connected")

        line = self._pipe_in.readline()
        if not line:
            return None

        return json.loads(line)

    def close(self) -> None:
        """Close pipes."""
        self._is_connected = False

        if self._pipe_in is not None:
            try:
                self._pipe_in.close()
            except Exception:
                pass
            self._pipe_in = None

        if self._pipe_out is not None:
            try:
                self._pipe_out.close()
            except Exception:
                pass
            self._pipe_out = None

    def __enter__(self) -> NamedPipeClient:
        self.connect()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
