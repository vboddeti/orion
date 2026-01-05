"""
Quick test: Single backbone with pretrained weights.
"""
import torch
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

# Create model and load checkpoint
print("Loading CryptoFaceNet4...")
model = CryptoFaceNet4()
load_checkpoint_for_config(model, input_size=64, verbose=False)

# Extract single backbone
backbone = model.nets[0]
backbone.eval()
print("✓ Extracted backbone\n")

# Fuse BatchNorm
print("Fusing BatchNorm into HerPN...")
backbone.init_orion_params()
print("✓ BatchNorm fused\n")

# Test forward pass
print("Testing cleartext forward pass...")
inp = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    out = backbone(inp)

print(f"Output shape: {out.shape}")
print(f"Output range: [{out.min():.6f}, {out.max():.6f}]")
print(f"Output mean: {out.mean():.6f}")

if torch.isnan(out).any():
    print("\n✗ NaN detected in output!")
else:
    print("\n✓ No NaN - backbone works correctly!")
