# orion/core/__init__.py
from .orion import scheme

init_scheme  = scheme.init_scheme
delete_scheme = scheme.delete_scheme
encode       = scheme.encode
decode       = scheme.decode
encrypt      = scheme.encrypt
decrypt      = scheme.decrypt
fit          = scheme.fit
compile      = scheme.compile
generate_keys_from_manifest = scheme.generate_keys_from_manifest
validate_compiled_manifest = scheme.validate_compiled_manifest

__all__ = [
    "init_scheme", "delete_scheme", "encode", "decode",
    "encrypt", "decrypt", "fit", "compile", "generate_keys_from_manifest",
    "validate_compiled_manifest",
]
