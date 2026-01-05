"""
Test CryptoFace's own fusion implementation to verify it works correctly.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

from CryptoFace.models import build_model
import argparse


def test_cryptoface_fusion():
    """Test CryptoFace's fusion with their code."""
    print(f"\n{'='*80}")
    print("Testing CryptoFace's Own Fusion Implementation")
    print(f"{'='*80}\n")

    # Create args for CryptoFace model
    args = argparse.Namespace()
    args.arch = 'cryptoface'
    args.input_size = 64
    args.patch_size = 32  # For 64x64 input, 2x2 grid of 32x32 patches
    args.num_classes = 93431  # Default from their code
    args.device = torch.device('cpu')

    # Build model
    print("Building CryptoFace model...")
    backbone = build_model(args)

    # Load checkpoint
    ckpt_path = "/research/hal-vishnu/code/orion-fhe/face_recognition/checkpoints/backbone-64x64.ckpt"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()

    print("✓ Model loaded\n")

    # Create test input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 64, 64)
    x = (x - x.min()) / (x.max() - x.min()) * 2 - 1  # Normalize to [-1, 1]

    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.6f}, {x.max():.6f}]\n")

    # Test UNFUSED
    print("UNFUSED forward:")
    with torch.no_grad():
        result = backbone(x)
        if isinstance(result, tuple):
            out_unfused = result[0]  # Get embeddings
        else:
            out_unfused = result

    print(f"  Output shape: {out_unfused.shape}")
    print(f"  Output range: [{out_unfused.min():.6f}, {out_unfused.max():.6f}]")
    print(f"  Output mean: {out_unfused.mean():.6f}\n")

    # Test FUSED
    print("FUSED forward:")
    backbone.fuse()  # CryptoFace's fusion

    with torch.no_grad():
        result = backbone.forward_fuse(x)
        if isinstance(result, tuple):
            out_fused = result[0]
        else:
            out_fused = result

    print(f"  Output shape: {out_fused.shape}")
    print(f"  Output range: [{out_fused.min():.6f}, {out_fused.max():.6f}]")
    print(f"  Output mean: {out_fused.mean():.6f}\n")

    # Compare
    print(f"{'='*80}")
    print("Comparison")
    print(f"{'='*80}\n")

    diff = (out_unfused - out_fused).abs()
    mae = diff.mean().item()
    max_err = diff.max().item()

    print(f"MAE: {mae:.6f}")
    print(f"Max error: {max_err:.6f}")

    if mae < 0.01:
        print(f"\n✓ CryptoFace fusion is CORRECT (MAE < 0.01)")
        return True
    else:
        print(f"\n✗ CryptoFace fusion has issues (MAE = {mae:.6f})")
        return False


if __name__ == "__main__":
    success = test_cryptoface_fusion()
    sys.exit(0 if success else 1)
