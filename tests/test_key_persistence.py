import h5py
import pytest
import torch

import orion
from orion.backend.python.tensors import CipherTensor


def _config(tmp_path, io_mode, load_secret_key=True):
    return {
        "ckks_params": {
            "LogN": 10,
            "LogQ": [40, 30],
            "LogP": [40],
            "LogScale": 30,
            "H": 64,
            "RingType": "Standard",
        },
        "orion": {
            "backend": "lattigo",
            "io_mode": io_mode,
            "diags_path": str(tmp_path / "diagonals.h5"),
            "keys_path": str(tmp_path / "evaluation_keys.h5"),
            "sk_path": str(tmp_path / "secret_key.h5"),
            "load_secret_key": load_secret_key,
        },
    }


def _manifest(config, **circuit):
    ckks = config["ckks_params"]
    return {
        "schema_version": 1,
        "ckks": {
            "logn": ckks["LogN"],
            "logq": ckks["LogQ"],
            "logp": ckks["LogP"],
            "logscale": ckks["LogScale"],
            "ringtype": ckks["RingType"].lower(),
        },
        "circuit": {
            "rotation_galois_elements": circuit.get(
                "rotation_galois_elements", []
            ),
            "bootstrap_slots": circuit.get("bootstrap_slots", []),
            "input_level": len(ckks["LogQ"]) - 1,
            "input_shapes": [[1]],
        },
    }


def test_public_key_only_initialization_can_encrypt(tmp_path):
    orion.init_scheme(_config(tmp_path, "save"))

    with h5py.File(tmp_path / "secret_key.h5", "r") as secret_keys:
        assert set(secret_keys) == {"sk"}
    with h5py.File(tmp_path / "evaluation_keys.h5", "r") as evaluation_keys:
        assert {"pk", "rk"}.issubset(evaluation_keys)
        assert "sk" not in evaluation_keys

    orion.init_scheme(
        _config(tmp_path, "load", load_secret_key=False)
    )
    plaintext = orion.encode(torch.tensor([1.0]), level=1)
    ciphertext = orion.encrypt(plaintext)

    rotated = ciphertext.roll(1)

    assert ciphertext is not None
    assert rotated is not None


def test_bootstrapper_keys_can_load_without_secret_key(tmp_path):
    config = _config(tmp_path, "save")
    config["ckks_params"].update({
        "LogN": 12,
        "LogQ": [50, 40, 40, 40, 40],
        "LogP": [50, 50],
        "LogScale": 40,
    })
    config["boot_params"] = {"LogP": [50, 50]}

    save_scheme = orion.init_scheme(config)
    save_scheme.bootstrapper.generate_bootstrapper(1 << 11)

    load_config = _config(tmp_path, "load", load_secret_key=False)
    load_config["ckks_params"] = config["ckks_params"]
    load_config["boot_params"] = config["boot_params"]
    load_scheme = orion.init_scheme(load_config)
    load_scheme.bootstrapper.generate_bootstrapper(1 << 11)
    plaintext = load_scheme.encode(torch.tensor([1.0]))
    ciphertext = load_scheme.encrypt(plaintext)
    bootstrapped_id = load_scheme.bootstrapper.bootstrap(
        ciphertext.ids[0], 1 << 11
    )

    with h5py.File(tmp_path / "evaluation_keys.h5", "r") as keys:
        assert "2048" in keys["bootstrappers"]
    assert bootstrapped_id is not None


def test_manifest_key_generation_does_not_create_model_data(tmp_path):
    config = _config(tmp_path, "none")
    config["orion"].update({
        "key_io_mode": "save",
        "diags_io_mode": "none",
    })
    scheme = orion.init_scheme(config)
    galois = int(scheme.backend.GetRotationGaloisElement(1))
    orion.generate_keys_from_manifest(
        _manifest(config, rotation_galois_elements=[galois])
    )

    assert (tmp_path / "evaluation_keys.h5").is_file()
    assert (tmp_path / "secret_key.h5").is_file()
    assert not (tmp_path / "diagonals.h5").exists()
    with h5py.File(tmp_path / "evaluation_keys.h5", "r") as keys:
        assert {"pk", "rk", str(galois)}.issubset(keys)
        assert "sk" not in keys


def test_independent_server_modes_do_not_load_secret_key(tmp_path):
    client_config = _config(tmp_path, "none")
    client_config["orion"].update({
        "key_io_mode": "save",
        "diags_io_mode": "none",
    })
    orion.init_scheme(client_config)

    server_config = _config(tmp_path, "none", load_secret_key=False)
    server_config["orion"].update({
        "key_io_mode": "load",
        "diags_io_mode": "save",
    })
    scheme = orion.init_scheme(server_config)
    plaintext = scheme.encode(torch.tensor([1.0]), level=1)
    assert scheme.encrypt(plaintext) is not None


def test_ciphertext_file_round_trip(tmp_path):
    scheme = orion.init_scheme(_config(tmp_path, "none"))
    original = orion.encrypt(orion.encode(torch.tensor([1.25, -2.5]), level=1))
    path = tmp_path / "ciphertext.h5"
    original.save(path)
    restored = CipherTensor.load(scheme, path)

    assert torch.allclose(
        restored.decrypt().decode()[:2],
        torch.tensor([1.25, -2.5]),
        atol=1e-3,
    )


def test_manifest_rejects_parameter_mismatch(tmp_path):
    config = _config(tmp_path, "none")
    config["orion"]["key_io_mode"] = "save"
    orion.init_scheme(config)
    manifest = _manifest(config)
    manifest["ckks"]["logn"] += 1

    with pytest.raises(ValueError, match="logn"):
        orion.generate_keys_from_manifest(manifest)


def test_compiled_manifest_rejects_missing_rotation(tmp_path):
    config = _config(tmp_path, "none")
    scheme = orion.init_scheme(config)
    scheme.compiled_circuit = {
        "input_level": 1,
        "bootstrap_slots": [],
        "rotation_galois_elements": [12345],
    }

    with pytest.raises(ValueError, match="missing 1 rotation"):
        orion.validate_compiled_manifest(_manifest(config))
