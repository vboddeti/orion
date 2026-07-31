import h5py
import torch

import orion


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
