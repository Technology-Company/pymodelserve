"""Minimal Django settings for testing pymodelserve.contrib.django."""

SECRET_KEY = "test-secret-key-not-for-production"

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "pymodelserve.contrib.django",
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": ":memory:",
    }
}

USE_TZ = True

# pymodelserve settings
MLSERVE = {
    "models_dir": None,  # Set in tests
    "auto_start": False,
    "health_check_interval": 30,
}
