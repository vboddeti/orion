"""
Sequential Branches Test - For Performance Comparison

Processes the same 3 branches SEQUENTIALLY (one after another) to compare
with parallel execution timing.
"""
import torch
import orion
import orion.nn as on
from pathlib import Path
import numpy as np
import time


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


def process_branch_sequential(branch_idx, config_path, input_data, input_level, seed):
    """
    Process a single branch sequentially (no multiprocessing).

    This is the same logic as the worker function but runs in the main process.
    """
    print(f"  Branch {branch_idx}: Starting...")

    # Initialize scheme
    orion.init_scheme(config_path)

    # Create branch model with same seed for reproducible weights
    torch.manual_seed(seed)
    branch = SimpleBranch()

    # Fit and compile
    sample_input = torch.randn(1, 3, 16, 16)
    orion.fit(branch, sample_input)
    orion.compile(branch)

    # Encrypt input
    input_ptxt = orion.encode(input_data, input_level)
    input_ctxt = orion.encrypt(input_ptxt)

    # Switch to HE mode and process
    branch.he()
    output_ctxt = branch(input_ctxt)

    # Decrypt and return
    output = output_ctxt.decrypt().decode()

    # Clean up scheme for next iteration
    orion.delete_scheme()

    print(f"  Branch {branch_idx}: Completed")

    return output.numpy()


def test_sequential_branches():
    """
    Test SEQUENTIAL processing with 3 branches.

    All branches use the SAME model weights (seed=42) but process DIFFERENT inputs.
    Processes branches ONE AT A TIME to measure sequential performance.
    """
    print("\n" + "="*70)
    print("SEQUENTIAL BRANCHES TEST - Performance Comparison")
    print("="*70)

    torch.manual_seed(42)
    config_path = get_config_path("resnet.yml")

    num_branches = 3

    # Create a single reference branch (all iterations will use same weights via seed=42)
    print(f"\nStep 1: Creating reference branch model...")
    reference_branch = SimpleBranch()

    # Create different inputs for each branch
    print(f"Step 2: Creating {num_branches} test inputs...")
    inputs = [torch.randn(1, 3, 16, 16) for _ in range(num_branches)]

    # Get cleartext outputs using the SAME reference branch for all inputs
    print("Step 3: Computing cleartext outputs...")
    reference_branch.eval()
    cleartext_outputs = []
    with torch.no_grad():
        for i, inp in enumerate(inputs):
            out = reference_branch(inp)
            cleartext_outputs.append(out.numpy())

    # Compute cleartext aggregation (sum of all branches)
    cleartext_sum = sum(cleartext_outputs)
    print(f"  Cleartext sum range: [{cleartext_sum.min():.4f}, {cleartext_sum.max():.4f}]")

    # Compile the reference branch to get input_level (same for all)
    print("\nStep 4: Compiling reference branch...")
    orion.init_scheme(config_path)
    sample_input = torch.randn(1, 3, 16, 16)
    orion.fit(reference_branch, sample_input)
    input_level = orion.compile(reference_branch)
    print(f"  Input level: {input_level}")

    # Clean up - will be recreated for each iteration
    orion.delete_scheme()

    # Process branches SEQUENTIALLY (one after another)
    print(f"\nStep 5: Processing {num_branches} branches SEQUENTIALLY...")
    start_time = time.time()

    fhe_outputs = []
    for i in range(num_branches):
        output = process_branch_sequential(i, config_path, inputs[i], input_level, 42)
        fhe_outputs.append(output)

    sequential_time = time.time() - start_time
    print(f"\n✓ Sequential processing completed in {sequential_time:.2f} seconds")

    # Aggregate FHE results (sum)
    fhe_sum = sum(fhe_outputs)
    print(f"  FHE sum range: [{fhe_sum.min():.4f}, {fhe_sum.max():.4f}]")

    # Verify each branch
    print("\nStep 6: Verifying individual branch outputs...")
    max_error = 0.0
    for i in range(num_branches):
        error = np.max(np.abs(fhe_outputs[i] - cleartext_outputs[i]))
        max_error = max(max_error, error)
        print(f"  Branch {i}: MAE = {error:.6f}")

    # Verify aggregated result
    print("\nStep 7: Verifying aggregated output...")
    agg_error = np.max(np.abs(fhe_sum - cleartext_sum))
    print(f"  Aggregation MAE = {agg_error:.6f}")
    max_error = max(max_error, agg_error)

    print(f"\nMaximum error: {max_error:.6f}")
    assert max_error < 1.0, f"Error {max_error:.6f} exceeds tolerance"

    print("\n" + "="*70)
    print("✓ SEQUENTIAL BRANCHES TEST PASSED!")
    print("="*70)
    print(f"Successfully processed {num_branches} branches sequentially")
    print(f"Total time: {sequential_time:.2f} seconds")
    print(f"Average time per branch: {sequential_time/num_branches:.2f} seconds")


if __name__ == "__main__":
    test_sequential_branches()
