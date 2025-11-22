"""
Simple Multiprocessing FHE Test - Option 1

Tests parallel execution using Python's multiprocessing to verify that
we can process multiple encrypted inputs in parallel using separate processes.
"""
import torch
import orion
import orion.nn as on
from pathlib import Path
import numpy as np
import time
import multiprocessing


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


class SimpleModel(on.Module):
    """Simple 2-layer model for testing parallel execution."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = on.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = on.Conv2d(16, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def worker_process(args):
    """
    Worker function that processes a single encrypted input.

    Each worker:
    1. Initializes the FHE scheme
    2. Recreates and compiles the model
    3. Processes the encrypted input
    4. Returns the decrypted result
    """
    idx, config_path, test_inp, seed = args

    print(f"  Worker {idx}: Starting (PID={multiprocessing.current_process().pid})")

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Initialize scheme in this process
    orion.init_scheme(config_path)

    # Recreate model
    model = SimpleModel()
    inp = torch.randn(1, 3, 32, 32)

    # Compile model
    orion.fit(model, inp)
    input_level = orion.compile(model)

    # Switch to FHE mode
    model.he()

    # Encrypt and process
    ptxt = orion.encode(test_inp, input_level)
    ctxt = orion.encrypt(ptxt)

    # Run FHE inference
    fhe_out_ctxt = model(ctxt)

    # Decrypt and decode
    fhe_out = fhe_out_ctxt.decrypt().decode()

    print(f"  Worker {idx}: Completed")

    # Return as numpy array for easier handling
    return idx, fhe_out.numpy()


def test_multiprocessing_simple():
    """
    Test parallel execution using multiprocessing with a simple 2-layer model.

    Creates 4 independent inputs and processes them in parallel using
    separate processes.
    """
    print("\n" + "="*70)
    print("SIMPLE MULTIPROCESSING FHE TEST - Option 1")
    print("="*70)

    torch.manual_seed(42)
    config_path = get_config_path("resnet.yml")

    # Note: We don't init_scheme here - each worker will do it
    # But we need to create the model once to get cleartext outputs

    print("\nStep 1: Creating simple 2-layer model...")
    model = SimpleModel()

    # Create 4 test inputs
    print("\nStep 2: Creating 4 test inputs...")
    test_inputs = []
    cleartext_outputs = []

    model.eval()
    for i in range(4):
        test_inp = torch.randn(1, 3, 32, 32)
        test_inputs.append(test_inp)

        # Get cleartext output for comparison
        with torch.no_grad():
            clear_out = model(test_inp)
            cleartext_outputs.append(clear_out.numpy())

    print(f"  Created {len(test_inputs)} test inputs")

    # Prepare arguments for workers
    print("\nStep 3: Preparing worker arguments...")
    worker_args = [
        (i, config_path, test_inputs[i], 42)  # idx, config, input, seed
        for i in range(len(test_inputs))
    ]

    # Process in parallel using multiprocessing
    print(f"\nStep 4: Processing {len(test_inputs)} inputs IN PARALLEL (multiprocessing)...")
    start_time = time.time()

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(worker_process, worker_args)

    end_time = time.time()
    parallel_time = end_time - start_time
    print(f"\n✓ Parallel processing completed in {parallel_time:.2f} seconds")

    # Sort results by index and extract outputs
    results.sort(key=lambda x: x[0])
    fhe_outputs = [r[1] for r in results]

    # Verify results
    print("\nStep 5: Verifying results...")
    max_error = 0.0
    for i in range(len(fhe_outputs)):
        fhe_out = fhe_outputs[i]
        clear_out = cleartext_outputs[i]

        error = np.max(np.abs(fhe_out - clear_out))
        max_error = max(max_error, error)
        print(f"  Input {i}: MAE = {error:.6f}")

    print(f"\nMaximum error across all inputs: {max_error:.6f}")
    assert max_error < 1.0, f"Error {max_error:.6f} exceeds tolerance 1.0"

    print("\n" + "="*70)
    print("✓ MULTIPROCESSING TEST PASSED!")
    print("="*70)
    print(f"Successfully processed {len(test_inputs)} inputs in parallel")
    print(f"Total time: {parallel_time:.2f} seconds")
    print(f"Average time per input: {parallel_time/len(test_inputs):.2f} seconds")
    print(f"\nNote: Each process compiled its own model")
    print(f"Total compilation overhead: ~{parallel_time:.2f}s across all workers")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    multiprocessing.set_start_method('spawn', force=True)
    test_multiprocessing_simple()
