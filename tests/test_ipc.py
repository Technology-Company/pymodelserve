"""Tests for IPC module."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


class TestThreadSafety:
    """Tests for thread-safe request/response on NamedPipeServer."""

    def _make_echo_client(self, config):
        """Create a client thread that echoes back requests with an id field."""

        def client_thread():
            client = NamedPipeClient(config.pipe_dir)
            client.connect()

            while True:
                msg = client.receive()
                if msg is None:
                    break

                message_type = msg.get("message")
                data = msg.get("data", {})

                if message_type == "shutdown":
                    client.send({"status": "bye"})
                    break
                elif message_type == "echo":
                    # Echo back the request id to prove ordering
                    client.send({"id": data.get("id"), "thread": data.get("thread")})
                elif message_type == "slow_echo":
                    time.sleep(0.01)  # Simulate work
                    client.send({"id": data.get("id"), "thread": data.get("thread")})

            client.close()

        return client_thread

    def test_concurrent_requests_are_serialized(self):
        """Multiple threads calling request() must not interleave pipe I/O.

        Each response must match the request that was sent (same id),
        proving the lock serializes send+receive pairs.
        """
        server = NamedPipeServer()
        config = server.setup()

        thread = threading.Thread(target=self._make_echo_client(config))
        thread.start()

        try:
            server.connect()

            num_threads = 8
            requests_per_thread = 10
            results = []  # list of (thread_id, request_id, response)
            errors = []

            def send_requests(thread_id):
                thread_results = []
                for i in range(requests_per_thread):
                    request_id = thread_id * 1000 + i
                    try:
                        resp = server.request("echo", {"id": request_id, "thread": thread_id})
                        thread_results.append((thread_id, request_id, resp))
                    except Exception as e:
                        errors.append((thread_id, request_id, e))
                return thread_results

            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = [pool.submit(send_requests, t) for t in range(num_threads)]
                for future in as_completed(futures):
                    results.extend(future.result())

            # Shutdown the echo client
            server.request("shutdown")
            thread.join(timeout=5)

            # Verify no errors
            assert len(errors) == 0, f"Got errors: {errors}"

            # Verify we got all responses
            assert len(results) == num_threads * requests_per_thread

            # Verify each response matches its request (lock prevented interleaving)
            for thread_id, request_id, resp in results:
                assert resp["id"] == request_id, (
                    f"Thread {thread_id} sent id={request_id} but got back id={resp['id']}. "
                    "This indicates pipe I/O interleaving (missing lock)."
                )
                assert resp["thread"] == thread_id

        finally:
            server.close()

    def test_concurrent_slow_requests(self):
        """Concurrent requests with simulated model latency still serialize correctly."""
        server = NamedPipeServer()
        config = server.setup()

        thread = threading.Thread(target=self._make_echo_client(config))
        thread.start()

        try:
            server.connect()

            num_threads = 4
            results = []
            errors = []

            def send_slow_request(thread_id):
                try:
                    resp = server.request("slow_echo", {"id": thread_id, "thread": thread_id})
                    return (thread_id, resp)
                except Exception as e:
                    errors.append((thread_id, e))
                    return None

            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                futures = [pool.submit(send_slow_request, t) for t in range(num_threads)]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)

            server.request("shutdown")
            thread.join(timeout=5)

            assert len(errors) == 0, f"Got errors: {errors}"
            assert len(results) == num_threads

            for thread_id, resp in results:
                assert resp["id"] == thread_id
                assert resp["thread"] == thread_id

        finally:
            server.close()

    def test_lock_attribute_exists(self):
        """NamedPipeServer must have a threading lock."""
        server = NamedPipeServer()
        assert hasattr(server, "_lock")
        assert isinstance(server._lock, type(threading.Lock()))
