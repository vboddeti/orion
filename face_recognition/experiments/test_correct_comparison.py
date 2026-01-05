"""
Correctly compare CryptoFace and Orion models following CryptoFace's pattern.

CryptoFace workflow (from ckks.py):
1. Load model and checkpoint
2. Evaluate unfused model using forward() → (out_global, pred, target)
3. Call fuse() to fuse BatchNorm
4. Evaluate fused model using forward_fuse() → out_global

Our Orion model:
- Fuses BatchNorm during weight loading (in load_checkpoint_for_config)
- Uses forward() → out (already fused)

Comparison strategy:
- CryptoFace FUSED (forward_fuse) vs Orion (forward with fused weights)
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

from CryptoFace.models import build_model
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_correct_comparison():
    print("="*80)
    print("TEST: CryptoFace (Fused) vs Orion (Fused Weights)")
    print("="*80)

    checkpoint_path = "face_recognition/checkpoints/backbone-64x64.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    device = torch.device('cpu')

    # ========================================================================
    # 1. Load CryptoFace model (following ckks.py:37-41)
    # ========================================================================
    print("\n" + "="*80)
    print("1. Loading CryptoFace Model")
    print("="*80)

    class Args:
        input_size = 64
        patch_size = 32

    backbone = build_model(Args())
    backbone.to(device)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()
    print("✓ CryptoFace model loaded")

    # Test unfused model first (like ckks.py:50)
    print("\nTesting UNFUSED CryptoFace model...")
    x_test = torch.randn(2, 3, 64, 64).to(device)

    with torch.no_grad():
        result = backbone(x_test)  # Should return (out_global, pred, target)

    if isinstance(result, tuple):
        out_global, pred, target = result
        print(f"  ✓ Unfused forward() returns tuple: ({out_global.shape}, {pred.shape}, {target.shape})")
        print(f"  Embedding range: [{out_global.min():.2f}, {out_global.max():.2f}]")

        # Check for NaN
        if torch.isnan(out_global).any():
            print(f"  ✗ WARNING: Unfused model produces NaN!")
        else:
            print(f"  ✓ Unfused model produces valid outputs")
    else:
        print(f"  ✗ ERROR: Expected tuple, got {type(result)}")
        return

    # Fuse BatchNorm (like ckks.py:61)
    print("\nFusing BatchNorm...")
    backbone.fuse()
    print("✓ BatchNorm fused")

    # Test fused model (like ckks.py:88)
    print("\nTesting FUSED CryptoFace model...")
    with torch.no_grad():
        emb_fused = backbone.forward_fuse(x_test)

    print(f"  Fused forward_fuse() returns: {emb_fused.shape}")
    print(f"  Embedding range: [{emb_fused.min():.2f}, {emb_fused.max():.2f}]")

    # Check for NaN/Inf
    if torch.isnan(emb_fused).any():
        print(f"  ✗ WARNING: Fused model produces NaN!")
    elif torch.isinf(emb_fused).any():
        print(f"  ✗ WARNING: Fused model produces Inf!")
    elif emb_fused.abs().max() > 1e6:
        print(f"  ✗ WARNING: Fused model produces very large values!")
    else:
        print(f"  ✓ Fused model produces reasonable outputs")

    # ========================================================================
    # 2. Load Orion model (with fused weights)
    # ========================================================================
    print("\n" + "="*80)
    print("2. Loading Orion Model (Fused Weights)")
    print("="*80)

    orion_model = CryptoFaceNet4()
    load_checkpoint_for_config(orion_model, input_size=64, verbose=False)
    orion_model.eval()
    orion_model.to(device)
    print("✓ Orion model loaded (BatchNorm fused during weight loading)")

    # Test Orion model
    print("\nTesting Orion model...")
    with torch.no_grad():
        orion_emb = orion_model(x_test)

    print(f"  Orion forward() returns: {orion_emb.shape}")
    print(f"  Embedding range: [{orion_emb.min():.2f}, {orion_emb.max():.2f}]")

    if torch.isnan(orion_emb).any():
        print(f"  ✗ WARNING: Orion model produces NaN!")
    else:
        print(f"  ✓ Orion model produces valid outputs")

    # ========================================================================
    # 3. Compare CryptoFace (fused) vs Orion (fused weights)
    # ========================================================================
    print("\n" + "="*80)
    print("3. Comparing Outputs")
    print("="*80)

    print(f"\nCryptoFace (fused): range=[{emb_fused.min():.4f}, {emb_fused.max():.4f}], " +
          f"mean={emb_fused.mean():.4f}, std={emb_fused.std():.4f}")
    print(f"Orion (fused):      range=[{orion_emb.min():.4f}, {orion_emb.max():.4f}], " +
          f"mean={orion_emb.mean():.4f}, std={orion_emb.std():.4f}")

    # Compute differences
    diff = (emb_fused - orion_emb).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nAbsolute differences:")
    print(f"  Max:  {max_diff:.6f}")
    print(f"  Mean: {mean_diff:.6f}")
    print(f"  Per-sample max: ", end="")
    for i in range(x_test.shape[0]):
        print(f"{diff[i].max():.4f}", end=" ")
    print()

    # Relative difference
    rel_diff = diff / (emb_fused.abs() + 1e-8)
    print(f"\nRelative differences:")
    print(f"  Max:  {rel_diff.max():.6f}")
    print(f"  Mean: {rel_diff.mean():.6f}")

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(emb_fused, orion_emb, dim=1)
    print(f"\nCosine similarities:")
    print(f"  Mean: {cos_sim.mean():.6f}")
    print(f"  Min:  {cos_sim.min():.6f}")
    print(f"  Values: {cos_sim}")

    # ========================================================================
    # 4. Verdict
    # ========================================================================
    print("\n" + "="*80)
    print("4. Verdict")
    print("="*80)

    # Check if both models produce valid outputs
    cryptoface_valid = not (torch.isnan(emb_fused).any() or torch.isinf(emb_fused).any() or emb_fused.abs().max() > 1e6)
    orion_valid = not (torch.isnan(orion_emb).any() or torch.isinf(orion_emb).any())

    if not cryptoface_valid:
        print("\n❌ CryptoFace fused model produces invalid outputs!")
        print("   This suggests an issue with the reference implementation.")
    elif not orion_valid:
        print("\n❌ Orion model produces invalid outputs!")
        print("   There may be an issue with weight loading.")
    elif max_diff < 0.01:
        print("\n✅ EXCELLENT: Models match within 0.01!")
        print("   Weight loading is correct.")
    elif max_diff < 1.0 and cos_sim.min() > 0.99:
        print("\n✅ GOOD: Models are very similar (max diff < 1.0, cos sim > 0.99)")
        print("   Minor differences likely due to numerical precision.")
    elif cos_sim.min() > 0.95:
        print("\n⚠️  ACCEPTABLE: Models are similar (cos sim > 0.95)")
        print(f"   Max difference: {max_diff:.4f}")
        print("   Differences may be due to architecture or implementation details.")
    else:
        print("\n❌ FAIL: Models differ significantly!")
        print(f"   Max difference: {max_diff:.4f}")
        print(f"   Min cosine similarity: {cos_sim.min():.4f}")
        print("   Weight loading may be incorrect.")

    print("="*80)


if __name__ == "__main__":
    test_correct_comparison()
