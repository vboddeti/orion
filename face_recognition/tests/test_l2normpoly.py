#!/usr/bin/env python3
"""
Isolated test for L2NormPoly polynomial evaluation in FHE.

This test bypasses the full model and directly tests:
1. Orion scheme initialization
2. L2NormPoly compilation with polynomial evaluator
3. FHE inference (x² -> sum -> polynomial -> normalize)

Expected: FHE output should match cleartext output with low MAE.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import orion

from models.pcnn import L2NormPoly


def test_l2normpoly():
    print("=" * 70)
    print("L2NormPoly Isolated Test")
    print("=" * 70)
    
    # Polynomial coefficients for 1/sqrt(y) approximation
    # These are from the CryptoFace LFW defaults
    a = 2.41e-07
    b = -2.44e-04
    c = 1.09e-01
    num_features = 256
    
    print(f"\nPolynomial coefficients:")
    print(f"  a = {a:.2e}")
    print(f"  b = {b:.2e}")
    print(f"  c = {c:.2e}")
    print(f"  num_features = {num_features}")
    
    # Create L2NormPoly module
    norm = L2NormPoly(a, b, c, num_features)
    
    # Create random input similar to what the linear layer outputs
    # Based on logs: linear output range was [-1.353, 1.674]
    torch.manual_seed(42)
    x = torch.randn(1, num_features) * 1.5  # Scale to match expected range
    print(f"\nInput x: shape={x.shape}, min={x.min():.3f}, max={x.max():.3f}")
    
    # Compute cleartext output
    norm.he_mode = False
    out_clear = norm(x)
    print(f"Cleartext output: min={out_clear.min():.3f}, max={out_clear.max():.3f}")
    
    # Also compute intermediate values for debugging
    y_clear = torch.sum(x ** 2, dim=1, keepdim=True)
    norm_inv_clear = a * y_clear ** 2 + b * y_clear + c
    print(f"  y (sum of squares) = {y_clear.item():.3f}")
    print(f"  norm_inv = {norm_inv_clear.item():.6f}")
    
    # ========== FHE Setup ==========
    print("\n" + "=" * 70)
    print("Setting up FHE scheme...")
    print("=" * 70)
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_dir, "configs/cryptoface_net4.yml")
    from orion.core.orion import Scheme
    scheme = Scheme()
    scheme.init_scheme(config_path)
    
    # Fit the module
    print("\nFitting L2NormPoly...")
    scheme.fit(norm, x)
    
    # Set level (normally done by level assignment, but we do it manually here)
    norm.level = 10  # Start at level 10, matches what we see in full model
    norm.fhe_input_shape = torch.Size([1, 256])
    
    # Compile
    print("Compiling L2NormPoly...")
    scheme.compile(norm)
    
    # ========== FHE Inference ==========
    print("\n" + "=" * 70)
    print("Running FHE inference...")
    print("=" * 70)
    
    # Encrypt input
    x_flat = x.flatten()
    x_ptxt = scheme.encoder.encode(x_flat, level=10)
    x_ctxt = scheme.encryptor.encrypt(x_ptxt)
    print(f"Encrypted input: level={x_ctxt.level()}")
    
    # Run FHE
    norm.he_mode = True
    out_ctxt = norm(x_ctxt)
    
    # Decrypt
    out_ptxt = scheme.encryptor.decrypt(out_ctxt)
    out_fhe = scheme.encoder.decode(out_ptxt)[:num_features]
    out_fhe = out_fhe.reshape(1, num_features)
    
    print(f"\nFHE output: min={out_fhe.min():.3f}, max={out_fhe.max():.3f}")
    
    # Compare
    mae = torch.mean(torch.abs(out_clear - out_fhe)).item()
    print(f"\n" + "=" * 70)
    print(f"RESULTS")
    print(f"=" * 70)
    print(f"Cleartext: [{out_clear.min():.6f}, {out_clear.max():.6f}]")
    print(f"FHE:       [{out_fhe.min():.6f}, {out_fhe.max():.6f}]")
    print(f"MAE: {mae:.6f}")
    
    if mae < 0.01:
        print("\n✓ SUCCESS")
    else:
        print("\n✗ FAILURE - Large error")
        
        # Debug: check if polynomial evaluation is the issue
        print("\n[DEBUG] Checking intermediate values...")


if __name__ == "__main__":
    test_l2normpoly()
