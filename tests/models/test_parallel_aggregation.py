"""
Parallel Branches with Aggregation - Option D Test

Tests the full parallel workflow:
- 4 parallel workers (fork)
- 4 simple branches processing different inputs
- Serialization for encrypted data transfer
- Aggregation of encrypted results
- End-to-end encryption maintained
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


# Global variables for fork to inherit
_global_branches = None


def worker_process_branch(branch_idx, serialized_input):
    """
    Worker function using fork + serialization.

    Processes one input through one branch and returns serialized encrypted result.
    """
    global _global_branches
    print(f"  Worker {branch_idx}: Starting (using inherited model via fork)...")

    # Deserialize input ciphertext (scheme is inherited via fork)
    from orion.core import scheme
    from orion.backend.python.tensors import CipherTensor
    input_ctxt = CipherTensor.deserialize(scheme, serialized_input)

    # Process through branch (encrypted! using inherited compiled model!)
    output_ctxt = _global_branches[branch_idx](input_ctxt)

    # Serialize encrypted result
    serialized_result = output_ctxt.serialize()

    print(f"  Worker {branch_idx}: Completed")

    return branch_idx, serialized_result


def test_parallel_aggregation():
    """
    Test 4 parallel branches with aggregation.

    Workflow:
    1. Compile 4 branches once
    2. Fork 4 workers (inherit compiled models)
    3. Each worker processes different encrypted input
    4. Aggregate 4 encrypted results (sum)
    5. Verify accuracy
    """
    global _global_branches

    print("\n" + "="*70)
    print("PARALLEL BRANCHES WITH AGGREGATION TEST")
    print("="*70)

    torch.manual_seed(42)
    config_path = get_config_path("resnet.yml")
    num_branches = 4

    # Setup in main process
    print(f"\nStep 1: Creating {num_branches} branch models...")
    orion.init_scheme(config_path)

    # Create branches with same weights (seed=42)
    branches = [SimpleBranch() for _ in range(num_branches)]
    _global_branches = branches  # Set global for fork to inherit

    # Create different inputs for each branch
    print(f"Step 2: Creating {num_branches} test inputs...")
    inputs = [torch.randn(1, 3, 16, 16) for _ in range(num_branches)]

    # Get cleartext outputs
    print("Step 3: Computing cleartext outputs...")
    cleartext_outputs = []
    for i, (branch, inp) in enumerate(zip(branches, inputs)):
        branch.eval()
        with torch.no_grad():
            out = branch(inp)
            cleartext_outputs.append(out.numpy())

    # Compute cleartext aggregation (sum)
    cleartext_sum = sum(cleartext_outputs)
    print(f"  Cleartext sum range: [{cleartext_sum.min():.4f}, {cleartext_sum.max():.4f}]")

    # Compile branches (ONCE!)
    print(f"\nStep 4: Compiling {num_branches} branches (ONCE, before forking)...")
    start_compile = time.time()

    sample_input = torch.randn(1, 3, 16, 16)
    for i, branch in enumerate(branches):
        orion.fit(branch, sample_input)
        input_level = orion.compile(branch)
        branch.he()

    compile_time = time.time() - start_compile
    print(f"  ✓ Compilation successful! Input level: {input_level}")
    print(f"  Compilation time: {compile_time:.2f} seconds")

    # Encrypt inputs
    print(f"\nStep 5: Encrypting {num_branches} inputs...")
    encrypted_inputs = []
    for i, inp in enumerate(inputs):
        input_ptxt = orion.encode(inp, input_level)
        input_ctxt = orion.encrypt(input_ptxt)
        encrypted_inputs.append(input_ctxt)

    # Serialize encrypted inputs
    print("Step 6: Serializing encrypted inputs...")
    serialized_inputs = [inp_ctxt.serialize() for inp_ctxt in encrypted_inputs]
    total_size = sum(sum(len(d) for d in si['data']) for si in serialized_inputs)
    print(f"  Total serialized size: {total_size / 1024 / 1024:.2f} MB")

    # Process branches in parallel using fork
    print(f"\nStep 7: Processing {num_branches} branches IN PARALLEL (fork + serialization)...")
    start_time = time.time()

    with multiprocessing.Pool(processes=num_branches) as pool:
        results = pool.starmap(
            worker_process_branch,
            [(i, serialized_inputs[i]) for i in range(num_branches)]
        )

    parallel_time = time.time() - start_time
    print(f"\n✓ Parallel processing completed in {parallel_time:.2f} seconds")

    # Deserialize results
    print("\nStep 8: Deserializing encrypted outputs...")
    from orion.core import scheme
    from orion.backend.python.tensors import CipherTensor

    results.sort(key=lambda x: x[0])
    encrypted_outputs = []
    for idx, serialized_result in results:
        output_ctxt = CipherTensor.deserialize(scheme, serialized_result)
        encrypted_outputs.append(output_ctxt)

    print(f"  Deserialized {num_branches} encrypted outputs")

    # Aggregate encrypted results (sum)
    print("\nStep 9: Aggregating encrypted results (sum)...")
    # Start with first output
    aggregated_ctxt = encrypted_outputs[0]
    # Add remaining outputs
    for i in range(1, num_branches):
        aggregated_ctxt = aggregated_ctxt + encrypted_outputs[i]

    print("  ✓ Aggregation complete (all encrypted!)")

    # Decrypt only at the end!
    print("\nStep 10: Decrypting final aggregated result...")
    aggregated_fhe = aggregated_ctxt.decrypt().decode().numpy()

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n{'='*70}")
    print("TIMING RESULTS")
    print(f"{'='*70}")
    print(f"Compilation time (once):          {compile_time:.2f} seconds")
    print(f"Parallel processing:              {parallel_time:.2f} seconds")
    print(f"Aggregation + decrypt:            {total_time - parallel_time:.2f} seconds")
    print(f"Total time:                       {total_time:.2f} seconds")
    print(f"{'='*70}")

    # Verify results (only decrypt the aggregated result, like PCNN!)
    print("\nStep 11: Verifying results...")
    print(f"FHE aggregated sum range: [{aggregated_fhe.min():.4f}, {aggregated_fhe.max():.4f}]")
    print(f"Cleartext sum range: [{cleartext_sum.min():.4f}, {cleartext_sum.max():.4f}]")

    # Compare aggregated result (NO individual decryptions!)
    print("\nVerifying aggregated output (PCNN workflow - no intermediate decryption)...")
    agg_error = np.max(np.abs(aggregated_fhe - cleartext_sum))
    print(f"  Aggregation MAE = {agg_error:.6f}")

    print(f"\nMaximum error: {agg_error:.6f}")
    assert agg_error < 1.0, f"Error {agg_error:.6f} exceeds tolerance"

    print("\n" + "="*70)
    print("✓ PARALLEL AGGREGATION TEST PASSED!")
    print("="*70)
    print(f"End-to-end encryption maintained!")
    print(f"Successfully processed {num_branches} branches in parallel")
    print(f"Aggregated {num_branches} encrypted results")
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    test_parallel_aggregation()
