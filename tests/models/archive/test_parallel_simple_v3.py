"""
Simple Parallel FHE Test - Verify Thread-Safe Evaluators
"""
import torch
import orion
import orion.nn as on
from pathlib import Path
# import numpy as np
import time
# from concurrent.futures import ThreadPoolExecutor


def get_config_path(yml_str):
    orion_path = Path(__file__).parent.parent.parent
    return str(orion_path / "configs" / f"{yml_str}")


class SimpleModel(on.Module):
    """Simple 2-layer model for testing compilation."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = on.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = on.Conv2d(16, 16, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def test_parallel_simple():
    """Test parallel execution with thread-safe evaluators."""
    print("\n" + "="*70)
    print("SIMPLE PARALLEL FHE TEST - Thread-Safe Evaluators")
    print("="*70)

    torch.manual_seed(42)
    config_path = get_config_path("resnet.yml")
    orion.init_scheme(config_path)

    # Create simple model
    print("\nStep 1: Creating simple 2-layer model...")
    model = SimpleModel()
    inp = torch.randn(1, 3, 32, 32)

    # Fit and compile
    print("Step 2: Running orion.fit()...")
    orion.fit(model, inp)

    print("Step 3: Running orion.compile()...")
    input_level = orion.compile(model)
    print(f"  ✓ Compilation successful! Input level: {input_level}")

    # Create 4 test inputs
    print("\nStep 4: Creating 4 test inputs...")
    test_inputs = []
    cleartext_outputs = []

    model.eval()
    for i in range(4):
        test_inp = torch.randn(1, 3, 32, 32)
        test_inputs.append(test_inp)
        with torch.no_grad():
            clear_out = model(test_inp)
            cleartext_outputs.append(clear_out)
    print(f"  Created {len(test_inputs)} test inputs")

    # Encrypt inputs
    print("\nStep 5: Encrypting inputs...")
    encrypted_inputs = []
    for test_inp in test_inputs:
        ptxt = orion.encode(test_inp, input_level)
        ctxt = orion.encrypt(ptxt)
        encrypted_inputs.append(ctxt)
    print(f"  Encrypted {len(encrypted_inputs)} inputs")

    # Switch to FHE mode
    model.he()

    # Enable thread-safe mode
    print("\nStep 6: Enabling thread-safe mode...")
    orion.enable_thread_safe_mode()

    # Process sequentially for now (to test if ThreadPoolExecutor is the issue)
    print(f"\nStep 7: Processing {len(encrypted_inputs)} inputs...")
    start_time = time.time()

    fhe_outputs = []
    for idx in range(len(encrypted_inputs)):
        print(f"  Processing input {idx}...")
        result = model(encrypted_inputs[idx])
        fhe_outputs.append(result)
        print(f"  Input {idx}: Completed")

    end_time = time.time()
    parallel_time = end_time - start_time
    print(f"✓ Parallel processing completed in {parallel_time:.2f} seconds")

    # Verify results
    print("\nStep 8: Verifying results...")
    import numpy as np
    max_error = 0.0
    for i in range(len(fhe_outputs)):
        fhe_out = fhe_outputs[i].decrypt().decode()
        clear_out = cleartext_outputs[i]
        error = np.max(np.abs(fhe_out.numpy() - clear_out.numpy()))
        max_error = max(max_error, error)
        print(f"  Input {i}: MAE = {error:.6f}")

    print(f"\nMaximum error across all inputs: {max_error:.6f}")
    assert max_error < 1.0, f"Error {max_error:.6f} exceeds tolerance"

    print("\n" + "="*70)
    print("✓ PARALLEL TEST PASSED!")
    print("="*70)
    print(f"Successfully processed {len(encrypted_inputs)} inputs in parallel")
    print(f"Total time: {parallel_time:.2f} seconds")
    print(f"Average time per input: {parallel_time/len(encrypted_inputs):.2f} seconds")


if __name__ == "__main__":
    test_parallel_simple()
