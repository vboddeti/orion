"""
Minimal test to check if basic compilation works without parallel execution.
"""
import torch
import orion
import orion.nn as on
from pathlib import Path


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


def test_simple_compile():
    """Test that basic compilation works."""
    print("\n" + "="*70)
    print("SIMPLE COMPILATION TEST")
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

    # Test a single encrypted inference (no parallelism)
    print("\nStep 4: Testing single encrypted inference...")
    test_inp = torch.randn(1, 3, 32, 32)

    # Get cleartext output
    model.eval()
    with torch.no_grad():
        clear_out = model(test_inp)

    # Encrypt and run FHE
    ptxt = orion.encode(test_inp, input_level)
    ctxt = orion.encrypt(ptxt)

    model.he()
    fhe_out_ctxt = model(ctxt)
    fhe_out = fhe_out_ctxt.decrypt().decode()

    # Check error
    import numpy as np
    error = np.max(np.abs(fhe_out.numpy() - clear_out.numpy()))
    print(f"  MAE: {error:.6f}")
    assert error < 1.0, f"Error {error:.6f} exceeds tolerance"

    print("\n" + "="*70)
    print("✓ SIMPLE COMPILATION TEST PASSED!")
    print("="*70)


if __name__ == "__main__":
    test_simple_compile()
