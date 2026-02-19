"""
pymodelserve - Run ML models in isolated subprocess environments.

Example usage:
    from pymodelserve import ModelManager

    with ModelManager.from_yaml("./models/fruit_classifier/model.yaml") as model:
        result = model.request("classify", {"image_path": "/path/to/image.jpg"})
"""

from pymodelserve.core.client import ModelClient, handler, run_client
from pymodelserve.core.manager import ModelManager
from pymodelserve.discovery.finder import ModelRegistry, discover_models

__version__ = "0.1.1"
__all__ = [
    "ModelManager",
    "ModelClient",
    "ModelRegistry",
    "discover_models",
    "handler",
    "run_client",
]
