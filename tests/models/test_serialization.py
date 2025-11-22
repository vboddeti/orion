"""
Test Ciphertext Serialization with Multiprocessing

Verifies that ciphertext serialization/deserialization works correctly
for passing encrypted data between processes.
"""
import torch
import orion
import orion.nn as on
from pathlib import Path
import numpy as np
import multiprocessing


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


class SimpleBranch(on.Module):
    """Simple 2-layer branch for testing."""
    def __init__(self):
        super(SimpleBranch, self).__init__()
        self.conv1 = on.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = on.Conv2d(8, 8, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Global model for fork to inherit
_global_model = None

def worker_with_serialization(worker_idx, serialized_input):
    """
    Worker that receives serialized ciphertext.

    With fork, the worker inherits the parent's compiled model via global variable!
    """
    global _global_model
    print(f"  Worker {worker_idx}: Starting (using inherited model via fork)...")

    # Deserialize input ciphertext (scheme is inherited via fork)
    from orion.core import scheme
    from orion.backend.python.tensors import CipherTensor
    input_ctxt = CipherTensor.deserialize(scheme, serialized_input)

    print(f"  Worker {worker_idx}: Deserialized input, shape={input_ctxt.shape}")

    # Process encrypted data using inherited model (already compiled!)
    output_ctxt = _global_model(input_ctxt)

    # Serialize output
    serialized_output = output_ctxt.serialize()

    print(f"  Worker {worker_idx}: Completed")

    return worker_idx, serialized_output


def test_serialization():
    """Test that serialization works for multiprocessing."""
    global _global_model

    print("\n" + "="*70)
    print("CIPHERTEXT SERIALIZATION TEST")
    print("="*70)

    torch.manual_seed(42)
    config_path = get_config_path("resnet.yml")

    # Setup in main process
    print("\nStep 1: Setting up model in main process...")
    orion.init_scheme(config_path)

    model = SimpleBranch()
    _global_model = model  # Set global for fork to inherit
    inp = torch.randn(1, 3, 16, 16)

    # Get cleartext output
    model.eval()
    with torch.no_grad():
        cleartext_out = model(inp).numpy()

    print("Step 2: Compiling model...")
    orion.fit(model, inp)
    input_level = orion.compile(model)
    model.he()

    # Encrypt input
    print("Step 3: Encrypting input...")
    input_ptxt = orion.encode(inp, input_level)
    input_ctxt = orion.encrypt(input_ptxt)

    # Serialize input
    print("Step 4: Serializing ciphertext...")
    serialized_input = input_ctxt.serialize()
    print(f"  Serialized data size: {sum(len(d) for d in serialized_input['data'])} bytes")

    # Test serialization in subprocess with fork
    print("\nStep 5: Testing serialization with multiprocessing (fork)...")
    print("  Worker will inherit compiled model via fork!")

    # Process in worker (model and scheme are inherited via fork!)
    with multiprocessing.Pool(processes=1) as pool:
        results = pool.starmap(
            worker_with_serialization,
            [(0, serialized_input)]
        )

    worker_idx, serialized_output = results[0]

    # Deserialize result in main process (scheme still active)
    print("\nStep 6: Deserializing result in main process...")
    from orion.core import scheme
    from orion.backend.python.tensors import CipherTensor

    output_ctxt = CipherTensor.deserialize(scheme, serialized_output)

    # Decrypt and verify
    print("Step 7: Verifying result...")
    output = output_ctxt.decrypt().decode().numpy()

    error = np.max(np.abs(output - cleartext_out))
    print(f"  Max error: {error:.6f}")

    assert error < 1.0, f"Error {error:.6f} exceeds tolerance"

    print("\n" + "="*70)
    print("✓ SERIALIZATION TEST PASSED!")
    print("="*70)
    print("Ciphertext serialization works correctly with multiprocessing")

    return True


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    test_serialization()
