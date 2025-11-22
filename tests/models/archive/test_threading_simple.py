"""
Simple Threading Test - Option B2

Tests threading with a simple 2-layer model to verify Go backend can handle
concurrent operations without modifications.
"""
import torch
import orion
import orion.nn as on
from pathlib import Path
import numpy as np
import time
import threading


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


def thread_worker(thread_idx, branch, input_data, input_level, results, errors):
    """
    Thread function to process one input through a branch.

    No backend modifications - just use threading directly.
    """
    try:
        print(f"  Thread {thread_idx}: Starting...")

        # Encrypt input
        input_ptxt = orion.encode(input_data, input_level)
        input_ctxt = orion.encrypt(input_ptxt)

        # Process through branch (encrypted)
        output_ctxt = branch(input_ctxt)

        # Store encrypted result (NOT decrypted!)
        results[thread_idx] = output_ctxt

        print(f"  Thread {thread_idx}: Completed")

    except Exception as e:
        print(f"  Thread {thread_idx}: ERROR - {e}")
        errors[thread_idx] = str(e)


def test_simple_threading():
    """
    Test parallel processing with 3 threads using simple threading.

    All threads share the SAME compiled model and scheme.
    No backend modifications required.
    """
    print("\n" + "="*70)
    print("SIMPLE THREADING TEST - Option B2")
    print("="*70)

    torch.manual_seed(42)
    config_path = get_config_path("resnet.yml")

    num_threads = 3

    # Create a single reference branch
    print(f"\nStep 1: Creating reference branch model...")
    branch = SimpleBranch()

    # Create different inputs for each thread
    print(f"Step 2: Creating {num_threads} test inputs...")
    inputs = [torch.randn(1, 3, 16, 16) for _ in range(num_threads)]

    # Get cleartext outputs using the reference branch
    print("Step 3: Computing cleartext outputs...")
    branch.eval()
    cleartext_outputs = []
    with torch.no_grad():
        for i, inp in enumerate(inputs):
            out = branch(inp)
            cleartext_outputs.append(out.numpy())

    cleartext_sum = sum(cleartext_outputs)
    print(f"  Cleartext sum range: [{cleartext_sum.min():.4f}, {cleartext_sum.max():.4f}]")

    # Compile the branch ONCE
    print("\nStep 4: Compiling branch (ONCE, before threading)...")
    orion.init_scheme(config_path)
    sample_input = torch.randn(1, 3, 16, 16)
    orion.fit(branch, sample_input)
    input_level = orion.compile(branch)
    print(f"  Input level: {input_level}")

    # Switch to HE mode
    branch.he()

    # Process inputs using threads
    print(f"\nStep 5: Processing {num_threads} inputs with THREADING...")
    results = [None] * num_threads
    errors = [None] * num_threads
    threads = []

    start_time = time.time()

    # Launch threads
    for i in range(num_threads):
        t = threading.Thread(
            target=thread_worker,
            args=(i, branch, inputs[i], input_level, results, errors)
        )
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join()

    threading_time = time.time() - start_time

    # Check for errors
    if any(errors):
        print("\n❌ THREADING FAILED - Errors occurred:")
        for i, err in enumerate(errors):
            if err:
                print(f"  Thread {i}: {err}")
        return False

    print(f"\n✓ Threading completed in {threading_time:.2f} seconds")

    # Decrypt results and verify
    print("\nStep 6: Decrypting and verifying results...")
    fhe_outputs = []
    for i in range(num_threads):
        output = results[i].decrypt().decode()
        fhe_outputs.append(output.numpy())

    fhe_sum = sum(fhe_outputs)
    print(f"  FHE sum range: [{fhe_sum.min():.4f}, {fhe_sum.max():.4f}]")

    # Verify each output
    print("\nStep 7: Verifying individual outputs...")
    max_error = 0.0
    for i in range(num_threads):
        error = np.max(np.abs(fhe_outputs[i] - cleartext_outputs[i]))
        max_error = max(max_error, error)
        print(f"  Thread {i}: MAE = {error:.6f}")

    # Verify aggregated result
    print("\nStep 8: Verifying aggregated output...")
    agg_error = np.max(np.abs(fhe_sum - cleartext_sum))
    print(f"  Aggregation MAE = {agg_error:.6f}")
    max_error = max(max_error, agg_error)

    print(f"\nMaximum error: {max_error:.6f}")
    assert max_error < 1.0, f"Error {max_error:.6f} exceeds tolerance"

    print("\n" + "="*70)
    print("✓ SIMPLE THREADING TEST PASSED!")
    print("="*70)
    print(f"Successfully processed {num_threads} inputs with threading")
    print(f"Total time: {threading_time:.2f} seconds")
    print("\nGo backend handles concurrent operations correctly!")
    print("Ready to apply threading to PCNN!")

    return True


if __name__ == "__main__":
    success = test_simple_threading()
    if not success:
        print("\n⚠️  Threading test failed - backend may need thread-safety modifications")
        exit(1)
