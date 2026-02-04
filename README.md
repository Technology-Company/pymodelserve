# pymodelserve

Run ML models in isolated subprocess environments with automatic dependency management.

Define your model once, and the library handles virtual environment creation, dependency installation, and inter-process communication via named pipes. Supports TensorFlow, PyTorch, or any Python ML framework with optional Django integration.

## Installation

```bash
pip install pymodelserve
```

For Django integration:
```bash
pip install pymodelserve[django]
```

## Quick Start

### 1. Create a Model

Create a directory structure for your model:

```
models/my_classifier/
├── model.yaml          # Configuration
├── model.py            # Client implementation
├── requirements.txt    # Dependencies
└── weights/            # Model files (optional)
```

**model.yaml:**
```yaml
name: my_classifier
version: "1.0.0"
python: ">=3.11"

client:
  module: model
  class: MyClassifierClient

requirements: requirements.txt

handlers:
  - name: classify
    input:
      image_path: string
    output:
      class: string
      confidence: float
```

**model.py:**
```python
from pymodelserve import ModelClient, handler

class MyClassifierClient(ModelClient):
    def setup(self):
        # Load your model here
        import tensorflow as tf
        self.model = tf.keras.models.load_model("weights/model.keras")

    @handler("classify")
    def classify(self, image_path: str) -> dict:
        # Your classification logic
        return {"class": "cat", "confidence": 0.95}

if __name__ == "__main__":
    MyClassifierClient().run()
```

### 2. Use the Model

```python
from pymodelserve import ModelManager

# Using context manager (recommended)
with ModelManager.from_yaml("./models/my_classifier/model.yaml") as model:
    result = model.request("classify", {"image_path": "/path/to/image.jpg"})
    print(result)  # {"class": "cat", "confidence": 0.95}

# Or manual lifecycle
manager = ModelManager.from_yaml("./models/my_classifier/model.yaml")
manager.start()
result = manager.request("classify", {"image_path": "/path/to/image.jpg"})
manager.stop()
```

### 3. Serve Models (CLI)

```bash
# Serve a single model
pml serve ./models/my_classifier/

# Serve all models in a directory
pml serve ./models/ --all

# List discovered models
pml list ./models/

# Create a new model scaffold
pml init my_new_model --framework tensorflow
```

## Features

- **Isolated Environments**: Each model runs in its own virtual environment with isolated dependencies
- **Named Pipe IPC**: Fast inter-process communication with no network overhead
- **Auto-Discovery**: Scan directories for models and register them automatically
- **Health Monitoring**: Periodic health checks with automatic restart on failure
- **Django Integration**: Optional Django app with views and management commands
- **CLI Tools**: Commands for serving, testing, and managing models

## Model Registry

Manage multiple models with the registry:

```python
from pymodelserve import ModelRegistry

registry = ModelRegistry()
registry.register("fruit", "./models/fruit_classifier/")
registry.register("sentiment", "./models/sentiment/")

# Start all models
registry.start_all()

# Use specific model
result = registry.get("fruit").request("classify", {"image_path": "..."})

# Get status
print(registry.status())

# Stop all
registry.stop_all()
```

Or use auto-discovery:

```python
from pymodelserve import discover_models, ModelRegistry

# Discover all models
configs = discover_models("./models/")

# Create registry from discovered models
registry = ModelRegistry()
registry.register_from_dir("./models/")
```

## Health Monitoring

```python
from pymodelserve import ModelRegistry
from pymodelserve.health import HealthChecker

registry = ModelRegistry()
registry.register_from_dir("./models/")
registry.start_all()

# Start health monitoring with auto-restart
checker = HealthChecker(
    registry=registry,
    interval=30,         # Check every 30 seconds
    max_failures=3,      # Restart after 3 failures
    auto_restart=True,
)
checker.start()

# ... your application runs ...

checker.stop()
registry.stop_all()
```

## Django Integration

**settings.py:**
```python
INSTALLED_APPS = [
    ...
    'pymodelserve.contrib.django',
]

MLSERVE = {
    "models_dir": BASE_DIR / "ml_models",
    "auto_start": True,
    "health_check_interval": 30,
}
```

**views.py:**
```python
from pymodelserve.contrib.django.views import ModelAPIView

class ClassifyImageView(ModelAPIView):
    model_name = "fruit_classifier"
    handler = "classify"

    def get_handler_input(self, request):
        image = request.FILES["image"]
        path = save_uploaded_file(image)
        return {"image_path": str(path)}
```

**urls.py:**
```python
from django.urls import path
from pymodelserve.contrib.django.views import GenericModelView, ModelStatusView

urlpatterns = [
    # Generic endpoint for any model/handler
    path("api/models/<str:model_name>/<str:handler>/", GenericModelView.as_view()),

    # Status endpoint
    path("api/models/status/", ModelStatusView.as_view()),
]
```

**Management command:**
```bash
python manage.py serve_models
python manage.py serve_models --model fruit_classifier --model sentiment
```

## Client Implementation

The `ModelClient` base class runs in the subprocess and handles IPC:

```python
from pymodelserve import ModelClient, handler

class MyModelClient(ModelClient):
    def setup(self):
        """Called once after IPC is established, before processing requests."""
        self.model = load_my_model()

    def teardown(self):
        """Called when shutting down."""
        cleanup_resources()

    @handler("predict")
    def predict(self, x: float, y: float) -> dict:
        """Handler methods receive kwargs and return dicts."""
        result = self.model.predict([[x, y]])
        return {"prediction": float(result[0])}

    # Alternative: use handle_* naming pattern
    def handle_info(self, **kwargs) -> dict:
        return {"version": "1.0", "status": "ready"}
```

## Configuration Reference

**model.yaml:**
```yaml
name: my_model              # Required: unique model name
version: "1.0.0"            # Model version
python: ">=3.11"            # Python version requirement

client:
  module: model             # Python module name (model.py)
  class: MyModelClient      # Class name in module

requirements: requirements.txt  # Dependencies file

handlers:                   # Optional: document handlers
  - name: predict
    input:
      x: float
    output:
      result: float

health:
  interval: 30              # Health check interval (seconds)
  timeout: 5                # Health check timeout
  max_failures: 3           # Failures before restart

resources:                  # Optional: resource limits
  memory_limit: 4G
  cpu_limit: 2
  gpu_ids: [0, 1]           # CUDA_VISIBLE_DEVICES
```

## License

MIT License - see [LICENSE](LICENSE) for details.
