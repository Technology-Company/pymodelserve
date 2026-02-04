"""Simple echo model for testing pymodelserve.

This example demonstrates the basic structure of a model client
without requiring any ML framework dependencies.
"""

from datetime import datetime

from pymodelserve import ModelClient, handler


class EchoClient(ModelClient):
    """A simple echo client for testing."""

    def setup(self) -> None:
        """Called on startup."""
        print("EchoClient setup complete")

    @handler("echo")
    def echo(self, message: str) -> dict:
        """Echo back the message with timestamp.

        Args:
            message: The message to echo.

        Returns:
            Echoed message with timestamp.
        """
        return {
            "echoed": message,
            "timestamp": datetime.now().isoformat(),
        }

    @handler("uppercase")
    def uppercase(self, text: str) -> dict:
        """Convert text to uppercase.

        Args:
            text: Text to convert.

        Returns:
            Uppercase text.
        """
        return {"result": text.upper()}


if __name__ == "__main__":
    EchoClient().run()
