"""Tests for IPC module."""

import threading

import pytest

from pymodelserve.core.ipc import (
    NamedPipeClient,
    NamedPipeServer,
    PipeConfig,
    PipeNotConnectedError,
)


class TestPipeConfig:
    """Tests for PipeConfig."""

    def test_pipe_paths(self, tmp_path):
        """Test pipe path generation."""
        config = PipeConfig(pipe_dir=tmp_path)
        assert config.pipe_in_path == tmp_path / "pipe_in"
        assert config.pipe_out_path == tmp_path / "pipe_out"

    def test_custom_pipe_names(self, tmp_path):
        """Test custom pipe names."""
        config = PipeConfig(
            pipe_dir=tmp_path,
            pipe_in_name="input",
            pipe_out_name="output",
        )
        assert config.pipe_in_path == tmp_path / "input"
        assert config.pipe_out_path == tmp_path / "output"


class TestNamedPipeServer:
    """Tests for NamedPipeServer."""

    def test_setup_creates_pipes(self):
        """Test that setup creates FIFO pipes."""
        server = NamedPipeServer()
        try:
            config = server.setup()
            assert config.pipe_in_path.exists()
            assert config.pipe_out_path.exists()
        finally:
            server.close()

    def test_send_before_connect_raises(self):
        """Test that send before connect raises error."""
        server = NamedPipeServer()
        server.setup()
        try:
            with pytest.raises(PipeNotConnectedError):
                server.send({"test": "data"})
        finally:
            server.close()


class TestServerClientCommunication:
    """Integration tests for server-client communication."""

    def test_bidirectional_communication(self):
        """Test sending messages between server and client."""
        server = NamedPipeServer()
        config = server.setup()

        # Start client in thread
        client_received = []
        client_error = []

        def client_thread():
            try:
                client = NamedPipeClient(config.pipe_dir)
                client.connect()

                # Receive message from server
                msg = client.receive()
                client_received.append(msg)

                # Send response
                client.send({"response": "ok", "echo": msg.get("data")})
                client.close()
            except Exception as e:
                client_error.append(e)

        thread = threading.Thread(target=client_thread)
        thread.start()

        try:
            # Connect server (blocks until client connects)
            server.connect()

            # Send message
            server.send({"message": "test", "data": "hello"})

            # Receive response
            response = server.receive()

            thread.join(timeout=5)

            assert len(client_error) == 0
            assert len(client_received) == 1
            assert client_received[0]["data"] == "hello"
            assert response["response"] == "ok"
            assert response["echo"] == "hello"
        finally:
            server.close()

    def test_request_response(self):
        """Test request/response pattern."""
        server = NamedPipeServer()
        config = server.setup()

        def client_thread():
            client = NamedPipeClient(config.pipe_dir)
            client.connect()

            while True:
                msg = client.receive()
                if msg is None:
                    break

                message_type = msg.get("message")
                if message_type == "ping":
                    client.send({"status": "pong"})
                elif message_type == "add":
                    data = msg.get("data", {})
                    result = data.get("a", 0) + data.get("b", 0)
                    client.send({"result": result})
                elif message_type == "shutdown":
                    client.send({"status": "bye"})
                    break

            client.close()

        thread = threading.Thread(target=client_thread)
        thread.start()

        try:
            server.connect()

            # Test ping
            response = server.request("ping")
            assert response["status"] == "pong"

            # Test add
            response = server.request("add", {"a": 5, "b": 3})
            assert response["result"] == 8

            # Shutdown
            response = server.request("shutdown")
            assert response["status"] == "bye"

            thread.join(timeout=5)
        finally:
            server.close()
