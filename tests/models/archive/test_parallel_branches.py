"""
Quick Parallel Branches Test

Tests multiprocessing with 3 simple branches processing different inputs in parallel.
This is a fast sanity check before running the full PCNN test.
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


def worker_process_branch(args):
    """
    Worker function to process one input through a branch.

    Args:
        branch_idx: Index of this branch
        config_path: Path to config file
        input_data: Input tensor (plaintext)
        input_level: FHE input level
        seed: Random seed for reproducibility
    """
    branch_idx, config_path, input_data, input_level, seed = args

    print(f"  Worker {branch_idx}: Starting (PID={multiprocessing.current_process().pid})")

    # Initialize scheme in this worker process
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

    print(f"  Worker {branch_idx}: Completed")

    return branch_idx, output.numpy()


def test_parallel_branches():
    """
    Test parallel processing with 3 branches processing different inputs.

    All branches use the SAME model weights (seed=42) but process DIFFERENT inputs.
    Each branch:
    - Processes a different input through the same model architecture
    - Runs in a separate process
    - Returns its result

    Results are aggregated (summed) to verify correctness against cleartext.
    """
    print("\n" + "="*70)
    print("PARALLEL BRANCHES TEST - Quick Sanity Check")
    print("="*70)

    torch.manual_seed(42)
    config_path = get_config_path("resnet.yml")

    num_branches = 3

    # Create a single reference branch (all workers will use same weights via seed=42)
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

    # Compile the reference branch to get input_level (same for all workers)
    print("\nStep 4: Compiling reference branch...")
    orion.init_scheme(config_path)
    sample_input = torch.randn(1, 3, 16, 16)
    orion.fit(reference_branch, sample_input)
    input_level = orion.compile(reference_branch)
    print(f"  Input level: {input_level}")

    # Clean up - will be recreated in workers
    orion.delete_scheme()

    # Prepare worker arguments
    print(f"\nStep 5: Preparing worker arguments for {num_branches} branches...")
    worker_args = [
        (i, config_path, inputs[i], input_level, 42)
        for i in range(num_branches)
    ]

    # Process in parallel using multiprocessing
    print(f"\nStep 6: Processing {num_branches} branches IN PARALLEL...")
    start_time = time.time()

    with multiprocessing.Pool(processes=num_branches) as pool:
        results = pool.map(worker_process_branch, worker_args)

    parallel_time = time.time() - start_time
    print(f"\n✓ Parallel processing completed in {parallel_time:.2f} seconds")

    # Sort results by branch index
    results.sort(key=lambda x: x[0])
    fhe_outputs = [r[1] for r in results]

    # Aggregate FHE results (sum)
    fhe_sum = sum(fhe_outputs)
    print(f"  FHE sum range: [{fhe_sum.min():.4f}, {fhe_sum.max():.4f}]")

    # Verify each branch
    print("\nStep 7: Verifying individual branch outputs...")
    max_error = 0.0
    for i in range(num_branches):
        error = np.max(np.abs(fhe_outputs[i] - cleartext_outputs[i]))
        max_error = max(max_error, error)
        print(f"  Branch {i}: MAE = {error:.6f}")

    # Verify aggregated result
    print("\nStep 8: Verifying aggregated output...")
    agg_error = np.max(np.abs(fhe_sum - cleartext_sum))
    print(f"  Aggregation MAE = {agg_error:.6f}")
    max_error = max(max_error, agg_error)

    print(f"\nMaximum error: {max_error:.6f}")
    assert max_error < 1.0, f"Error {max_error:.6f} exceeds tolerance"

    print("\n" + "="*70)
    print("✓ PARALLEL BRANCHES TEST PASSED!")
    print("="*70)
    print(f"Successfully processed {num_branches} branches in parallel")
    print(f"Total time: {parallel_time:.2f} seconds")
    print(f"Average time per branch: {parallel_time/num_branches:.2f} seconds")
    print("\nReady to run full PCNN test!")


if __name__ == "__main__":
    # Required for multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    test_parallel_branches()
