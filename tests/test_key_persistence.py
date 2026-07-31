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
