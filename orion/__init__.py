# orion/__init__.py

from .core import (
    init_scheme, delete_scheme, encode, decode, encrypt, decrypt, fit, compile,
    generate_keys_from_manifest,
    validate_compiled_manifest,
)
from .core.logger import logger, set_log_level, TracerDashboard

__version__ = "1.0.2"

__all__ = [
    "init_scheme", "delete_scheme", "encode", "decode",
    "encrypt", "decrypt", "fit", "compile", "generate_keys_from_manifest",
    "validate_compiled_manifest",
    "logger", "set_log_level", "TracerDashboard",
]
