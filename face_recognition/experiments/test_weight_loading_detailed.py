"""
Test if BatchNorm statistics are being loaded correctly.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_batchnorm_loading():
    """Check if BatchNorm statistics are loaded."""
    print("="*80)
    print("TEST: BatchNorm Statistics Loading")
    print("="*80)

    # Create model
    model = CryptoFaceNet4()

    # Check initial BatchNorm statistics (should be random)
    print("\nBEFORE loading weights:")
    print("-"*80)
    bn0_1_mean_before = model.nets[0].layer1.bn0_1.running_mean.clone()
    bn0_1_var_before = model.nets[0].layer1.bn0_1.running_var.clone()
    print(f"nets.0.layer1.bn0_1.running_mean[:5] = {bn0_1_mean_before[:5]}")
    print(f"nets.0.layer1.bn0_1.running_var[:5]  = {bn0_1_var_before[:5]}")

    # Load weights with verbose mode
    print("\n" + "="*80)
    print("LOADING WEIGHTS:")
    print("="*80)
    load_checkpoint_for_config(model, input_size=64, verbose=True)

    # Check after loading
    print("\nAFTER loading weights:")
    print("-"*80)
    bn0_1_mean_after = model.nets[0].layer1.bn0_1.running_mean
    bn0_1_var_after = model.nets[0].layer1.bn0_1.running_var
    print(f"nets.0.layer1.bn0_1.running_mean[:5] = {bn0_1_mean_after[:5]}")
    print(f"nets.0.layer1.bn0_1.running_var[:5]  = {bn0_1_var_after[:5]}")

    # Check if they changed
    mean_changed = not torch.allclose(bn0_1_mean_before, bn0_1_mean_after)
    var_changed = not torch.allclose(bn0_1_var_before, bn0_1_var_after)

    print(f"\nBatchNorm statistics changed:")
    print(f"  running_mean: {mean_changed}")
    print(f"  running_var:  {var_changed}")

    if mean_changed and var_changed:
        print("\n✓ BatchNorm statistics were loaded successfully!")
    else:
        print("\n✗ BatchNorm statistics were NOT loaded!")

    # Also check final bn
    print("\n" + "="*80)
    print("Final BatchNorm1d:")
    print("="*80)
    print(f"bn.running_mean[:5] = {model.nets[0].bn.running_mean[:5]}")
    print(f"bn.running_var[:5]  = {model.nets[0].bn.running_var[:5]}")


if __name__ == "__main__":
    test_batchnorm_loading()
