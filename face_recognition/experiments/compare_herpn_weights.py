"""
Compare HerPN weights between CryptoFace and Orion.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')
sys.path.append('/research/hal-vishnu/code/orion-fhe/CryptoFace')

from CryptoFace.models import PatchCNN as CryptoFacePatchCNN
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

print("="*80)
print("COMPARING HERPN WEIGHTS: CryptoFace vs Orion")
print("="*80)

# Load CryptoFace model
cryptoface_model = CryptoFacePatchCNN(input_size=64, patch_size=32)
ckpt = torch.load("face_recognition/checkpoints/backbone-64x64.ckpt", map_location='cpu', weights_only=False)
cryptoface_model.load_state_dict(ckpt['backbone'], strict=False)
cryptoface_model.eval()
cryptoface_model.fuse()

# Load Orion model
orion_model = CryptoFaceNet4()
load_checkpoint_for_config(orion_model, input_size=64, verbose=False)
orion_model.init_orion_params()
orion_model.eval()

# Compare first patch, first layer (layer1/layers[0])
print("\n" + "="*80)
print("Patch 0, Layer 1 (layers[0])")
print("="*80)

# CryptoFace: nets[0].layers[0] (after fuse, has a0, a1, a2, b0, b1)
cf_layer = cryptoface_model.nets[0].layers[0]

print("\nCryptoFace layers[0] (after fuse):")
print(f"  Type: {type(cf_layer).__name__}")
print(f"  a0 (first HerPN, x^2): min={cf_layer.a0.min():.6f}, max={cf_layer.a0.max():.6f}, mean={cf_layer.a0.mean():.6f}")
print(f"  a1 (first HerPN, x):   min={cf_layer.a1.min():.6f}, max={cf_layer.a1.max():.6f}, mean={cf_layer.a1.mean():.6f}")
print(f"  a2 (first HerPN, 1):   min={cf_layer.a2.min():.6f}, max={cf_layer.a2.max():.6f}, mean={cf_layer.a2.mean():.6f}")
print(f"  b0 (second HerPN, x^2): min={cf_layer.b0.min():.6f}, max={cf_layer.b0.max():.6f}, mean={cf_layer.b0.mean():.6f}")
print(f"  b1 (second HerPN, x):   min={cf_layer.b1.min():.6f}, max={cf_layer.b1.max():.6f}, mean={cf_layer.b1.mean():.6f}")
print(f"  a0 shape: {cf_layer.a0.shape}")
print(f"  a0[:3]: {cf_layer.a0.squeeze()[:3]}")

# Orion: nets[0].layer1.herpn1
or_herpn = orion_model.nets[0].layer1.herpn1

print("\nOrion herpn1:")
print(f"  Type: {type(or_herpn).__name__}")
if hasattr(or_herpn, 'w2'):
    print(f"  w2: min={or_herpn.w2.min():.6f}, max={or_herpn.w2.max():.6f}, mean={or_herpn.w2.mean():.6f}")
    print(f"  w1: min={or_herpn.w1.min():.6f}, max={or_herpn.w1.max():.6f}, mean={or_herpn.w1.mean():.6f}")
    print(f"  w0: min={or_herpn.w0.min():.6f}, max={or_herpn.w0.max():.6f}, mean={or_herpn.w0.mean():.6f}")
    print(f"  w2 shape: {or_herpn.w2.shape}")
    print(f"  w2[0,0,:3]: {or_herpn.w2[0,0,:3]}")

# Compare coefficients
if hasattr(or_herpn, 'w2'):
    print("\n" + "="*80)
    print("COMPARISON (First HerPN only):")
    print("="*80)

    # CryptoFace uses: a0*x^2 + a1*x + a2
    # Orion uses: w2*x^2 + w1*x + w0
    # So: cf.a0 should match or.w2, cf.a1 should match or.w1, cf.a2 should match or.w0

    # Reshape for comparison
    cf_a0 = cf_layer.a0.squeeze()  # [16]
    cf_a1 = cf_layer.a1.squeeze()  # [16]
    cf_a2 = cf_layer.a2.squeeze()  # [16]

    or_w2 = or_herpn.w2.squeeze()  # [16, H, W]
    or_w1 = or_herpn.w1.squeeze()  # [16, H, W]
    or_w0 = or_herpn.w0.squeeze()  # [16, H, W]

    # For spatial HerPN, take mean over spatial dimensions if needed
    if len(or_w2.shape) > 1:
        print(f"\nOrion HerPN has spatial dimensions: {or_w2.shape}")
        print(f"Taking values at position (0,0) for comparison")
        or_w2_comp = or_w2[:, 0, 0]
        or_w1_comp = or_w1[:, 0, 0]
        or_w0_comp = or_w0[:, 0, 0]
    else:
        or_w2_comp = or_w2
        or_w1_comp = or_w1
        or_w0_comp = or_w0

    print(f"\nQuadratic coefficient (x^2):")
    print(f"  CryptoFace a0[:5]: {cf_a0[:5]}")
    print(f"  Orion w2[:5]:      {or_w2_comp[:5]}")
    print(f"  Match: {torch.allclose(cf_a0, or_w2_comp, rtol=1e-5, atol=1e-6)}")

    print(f"\nLinear coefficient (x):")
    print(f"  CryptoFace a1[:5]: {cf_a1[:5]}")
    print(f"  Orion w1[:5]:      {or_w1_comp[:5]}")
    print(f"  Match: {torch.allclose(cf_a1, or_w1_comp, rtol=1e-5, atol=1e-6)}")

    print(f"\nConstant coefficient:")
    print(f"  CryptoFace a2[:5]: {cf_a2[:5]}")
    print(f"  Orion w0[:5]:      {or_w0_comp[:5]}")
    print(f"  Match: {torch.allclose(cf_a2, or_w0_comp, rtol=1e-5, atol=1e-6)}")
