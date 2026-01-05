"""
Compare UN-FUSED CryptoFace vs Orion PCNN.

Our Orion model fuses BatchNorm during weight loading but uses normal forward().
CryptoFace fuses BatchNorm via fuse() method and uses forward_fuse().

This test compares:
- CryptoFace (unfused) using forward() → returns (embedding, pred, target)
- Orion (fused weights) using forward() → returns embedding

Both should produce similar embeddings.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

from CryptoFace.models import build_model
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def test_unfused_comparison():
    print("="*80)
    print("TEST: CryptoFace (Unfused) vs Orion (Fused Weights) Comparison")
    print("="*80)

    checkpoint_path = "face_recognition/checkpoints/backbone-64x64.ckpt"
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Load CryptoFace model (UNFUSED)
    print("\n1. Loading CryptoFace model (unfused)...")
    class Args:
        input_size = 64
        patch_size = 32

    cryptoface_model = build_model(Args())
    cryptoface_model.load_state_dict(ckpt['backbone'])
    cryptoface_model.eval()
    print("   ✓ CryptoFace loaded (using normal forward with BatchNorm)")

    # Load Orion model (fused weights)
    print("\n2. Loading Orion model (fused weights)...")
    orion_model = CryptoFaceNet4()
    load_checkpoint_for_config(orion_model, input_size=64, verbose=False)
    orion_model.eval()
    print("   ✓ Orion loaded (BatchNorm fused into weights)")

    # Create test input
    print("\n3. Creating test input...")
    x = torch.randn(4, 3, 64, 64)
    print(f"   Input shape: {x.shape}")

    # Forward pass
    print("\n4. Running forward pass...")
    with torch.no_grad():
        # CryptoFace (unfused) - returns tuple
        cryptoface_result = cryptoface_model(x)
        cryptoface_emb = cryptoface_result[0]  # Get embedding from tuple

        # Orion (fused weights) - returns embedding directly
        orion_emb = orion_model(x)

    print(f"\n   CryptoFace embedding: shape={cryptoface_emb.shape}, " +
          f"range=[{cryptoface_emb.min():.2f}, {cryptoface_emb.max():.2f}]")
    print(f"   Orion embedding:      shape={orion_emb.shape}, " +
          f"range=[{orion_emb.min():.2f}, {orion_emb.max():.2f}]")

    # Compare
    print("\n5. Comparison:")
    diff = (cryptoface_emb - orion_emb).abs()
    print(f"   Max abs diff:  {diff.max():.6f}")
    print(f"   Mean abs diff: {diff.mean():.6f}")

    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(cryptoface_emb, orion_emb, dim=1)
    print(f"   Cosine similarity: mean={cos_sim.mean():.6f}, min={cos_sim.min():.6f}")

    # Check for match
    if diff.max() < 1e-4 and cos_sim.min() > 0.9999:
        print("\n✅ PERFECT MATCH: Models produce identical outputs!")
    elif diff.max() < 1.0 and cos_sim.min() > 0.99:
        print("\n✅ CLOSE MATCH: Models produce very similar outputs!")
    else:
        print("\n❌ MISMATCH: Outputs differ significantly")

    print("="*80)


if __name__ == "__main__":
    test_unfused_comparison()
