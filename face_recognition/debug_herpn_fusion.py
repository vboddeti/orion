"""Debug HerPN fusion to find source of NaN."""
import sys
import torch
import math

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

print("Loading model and checkpoint...")
model = CryptoFaceNet4()
load_checkpoint_for_config(model, input_size=64, verbose=False)

print("\nChecking BatchNorm statistics BEFORE fusion...")
net = model.nets[0]
layer1 = net.layer1

# Check bn statistics
for bn_name in ['bn0_1', 'bn1_1', 'bn2_1']:
    bn = getattr(layer1, bn_name)
    mean = bn.running_mean
    var = bn.running_var
    print(f"\n{bn_name}:")
    print(f"  mean: min={mean.min():.6f}, max={mean.max():.6f}, has_nan={torch.isnan(mean).any()}")
    print(f"  var:  min={var.min():.6f}, max={var.max():.6f}, has_nan={torch.isnan(var).any()}")
    print(f"  First 5 mean values: {mean[:5]}")
    print(f"  First 5 var values: {var[:5]}")

# Check if herpn1_weight and herpn1_bias exist
if hasattr(layer1, 'herpn1_weight'):
    weight = layer1.herpn1_weight
    bias = layer1.herpn1_bias
    print(f"\nherpn1_weight: shape={weight.shape}, has_nan={torch.isnan(weight).any()}")
    print(f"  range: [{weight.min():.6f}, {weight.max():.6f}]")
    print(f"herpn1_bias: shape={bias.shape}, has_nan={torch.isnan(bias).any()}")
    print(f"  range: [{bias.min():.6f}, {bias.max():.6f}]")
else:
    print("\n⚠ herpn1_weight/bias not found!")

print("\n" + "="*80)
print("Manually computing HerPN fusion (like in init_orion_params)...")
print("="*80)

# Manually replicate HerPN fusion math
m0, v0 = layer1.bn0_1.running_mean, layer1.bn0_1.running_var
m1, v1 = layer1.bn1_1.running_mean, layer1.bn1_1.running_var
m2, v2 = layer1.bn2_1.running_mean, layer1.bn2_1.running_var

if hasattr(layer1, 'herpn1_weight'):
    g = layer1.herpn1_weight.squeeze()
    b = layer1.herpn1_bias.squeeze()
else:
    g = torch.ones(layer1.in_planes)
    b = torch.zeros(layer1.in_planes)

eps = layer1.bn0_1.eps

print(f"\nInput parameters:")
print(f"  g (herpn weight): has_nan={torch.isnan(g).any()}, range=[{g.min():.6f}, {g.max():.6f}]")
print(f"  b (herpn bias): has_nan={torch.isnan(b).any()}, range=[{b.min():.6f}, {b.max():.6f}]")
print(f"  eps: {eps}")

# Compute w2, w1, w0
print(f"\nComputing w2...")
denominator_w2 = torch.sqrt(8 * math.pi * (v2 + eps))
print(f"  denominator (sqrt(8π(v2+eps))): min={denominator_w2.min():.6f}, max={denominator_w2.max():.6f}")
w2 = torch.divide(g, denominator_w2)
print(f"  w2: has_nan={torch.isnan(w2).any()}, min={w2.min():.6f}, max={w2.max():.6f}")

print(f"\nComputing w1...")
denominator_w1 = 2 * torch.sqrt(v1 + eps)
print(f"  denominator (2*sqrt(v1+eps)): min={denominator_w1.min():.6f}, max={denominator_w1.max():.6f}")
w1 = torch.divide(g, denominator_w1)
print(f"  w1: has_nan={torch.isnan(w1).any()}, min={w1.min():.6f}, max={w1.max():.6f}")

print(f"\nComputing w0...")
term1 = torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + eps)))
term2 = torch.divide(m1, 2 * torch.sqrt(v1 + eps))
term3 = torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + eps)))
print(f"  term1: has_nan={torch.isnan(term1).any()}, min={term1.min():.6f}, max={term1.max():.6f}")
print(f"  term2: has_nan={torch.isnan(term2).any()}, min={term2.min():.6f}, max={term2.max():.6f}")
print(f"  term3: has_nan={torch.isnan(term3).any()}, min={term3.min():.6f}, max={term3.max():.6f}")

w0 = b + g * (term1 - term2 - term3)
print(f"  w0: has_nan={torch.isnan(w0).any()}", end="")
if not torch.isnan(w0).any():
    print(f", min={w0.min():.6f}, max={w0.max():.6f}")
else:
    print()
    # Find where NaN occurs
    nan_mask = torch.isnan(w0)
    print(f"  NaN indices: {torch.where(nan_mask)[0].tolist()[:10]}")

print(f"\nComputing factored coefficients (a1 = w1/w2, a0 = w0/w2)...")
a1 = w1 / w2
a0 = w0 / w2
print(f"  a1: has_nan={torch.isnan(a1).any()}", end="")
if not torch.isnan(a1).any():
    print(f", min={a1.min():.6f}, max={a1.max():.6f}")
else:
    print()
    nan_mask = torch.isnan(a1)
    print(f"    NaN at indices: {torch.where(nan_mask)[0].tolist()[:10]}")
    print(f"    w1 at NaN indices: {w1[nan_mask][:5]}")
    print(f"    w2 at NaN indices: {w2[nan_mask][:5]}")

print(f"  a0: has_nan={torch.isnan(a0).any()}", end="")
if not torch.isnan(a0).any():
    print(f", min={a0.min():.6f}, max={a0.max():.6f}")
else:
    print()
    nan_mask = torch.isnan(a0)
    print(f"    NaN at indices: {torch.where(nan_mask)[0].tolist()[:10]}")
    print(f"    w0 at NaN indices: {w0[nan_mask][:5]}")
    print(f"    w2 at NaN indices: {w2[nan_mask][:5]}")

print("\n" + "="*80)
print("Now calling init_orion_params() and checking result...")
print("="*80)

model.init_orion_params()

print("\nChecking fused HerPN...")
herpn1 = layer1.herpn1
if herpn1 is not None:
    print(f"herpn1.weight0_raw: has_nan={torch.isnan(herpn1.weight0_raw).any()}")
    print(f"herpn1.weight1_raw: has_nan={torch.isnan(herpn1.weight1_raw).any()}")
    print(f"herpn1.scale_factor: has_nan={torch.isnan(herpn1.scale_factor).any()}")
else:
    print("⚠ herpn1 is None!")

print("\nTrying a forward pass...")
x = torch.randn(1, 3, 64, 64)
try:
    out = model(x)
    print(f"Output: has_nan={torch.isnan(out).any()}")
    if torch.isnan(out).any():
        print("  Forward pass produced NaN!")
    else:
        print(f"  Output range: [{out.min():.6f}, {out.max():.6f}]")
except Exception as e:
    print(f"Forward pass failed with error: {e}")
