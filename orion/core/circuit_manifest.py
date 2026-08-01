"""Versioned, model-independent circuit requirements for client key generation."""

import json
from pathlib import Path


SCHEMA_VERSION = 1


def load_manifest(value):
    if isinstance(value, (str, Path)):
        with open(value, "r", encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError("Circuit manifest must be a mapping or JSON file path")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported circuit manifest schema_version: "
            f"{value.get('schema_version')!r}"
        )
    circuit = value.get("circuit")
    ckks = value.get("ckks")
    if not isinstance(circuit, dict) or not isinstance(ckks, dict):
        raise ValueError("Circuit manifest requires 'ckks' and 'circuit' mappings")
    rotations = circuit.get("rotation_galois_elements")
    bootstraps = circuit.get("bootstrap_slots")
    if not isinstance(rotations, list) or not all(
        isinstance(item, int) and item > 0 for item in rotations
    ):
        raise ValueError("rotation_galois_elements must be positive integers")
    if not isinstance(bootstraps, list) or not all(
        isinstance(item, int) and item > 0 for item in bootstraps
    ):
        raise ValueError("bootstrap_slots must be positive integers")
    return value


def validate_parameters(manifest, params):
    manifest = load_manifest(manifest)
    expected = {
        "logn": params.get_logn(),
        "logq": params.get_logq(),
        "logp": params.get_logp(),
        "logscale": params.get_logscale(),
        "ringtype": params.get_ringtype(),
    }
    actual = manifest["ckks"]
    mismatches = [
        name for name, value in expected.items()
        if actual.get(name) != value
    ]
    if mismatches:
        raise ValueError(
            "Circuit manifest CKKS parameters do not match initialized scheme: "
            + ", ".join(mismatches)
        )
    return manifest
