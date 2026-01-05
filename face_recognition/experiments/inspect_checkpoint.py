"""
Inspect the checkpoint to understand its structure and check for issues.
"""
import torch
import sys

sys.path.append('/research/hal-vishnu/code/orion-fhe')

checkpoint_path = "face_recognition/checkpoints/backbone-64x64.ckpt"
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*80)
print("Checkpoint Inspection")
print("="*80)

print("\nTop-level keys:")
for key in ckpt.keys():
    print(f"  {key}")

print("\n" + "="*80)
print("Backbone State Dict")
print("="*80)

state_dict = ckpt['backbone']
print(f"\nTotal parameters: {len(state_dict)}")

# Check for NaN or Inf in parameters
print("\nChecking for NaN/Inf in parameters...")
nan_params = []
inf_params = []
zero_var_bn = []

for name, param in state_dict.items():
    if torch.isnan(param).any():
        nan_params.append(name)
    if torch.isinf(param).any():
        inf_params.append(name)
    # Check for zero variance in BatchNorm
    if 'running_var' in name:
        if (param == 0).any():
            zero_var_bn.append((name, (param == 0).sum().item()))

if nan_params:
    print(f"\n❌ Found {len(nan_params)} parameters with NaN:")
    for name in nan_params[:10]:
        print(f"  {name}")
else:
    print("  ✓ No NaN parameters")

if inf_params:
    print(f"\n❌ Found {len(inf_params)} parameters with Inf:")
    for name in inf_params[:10]:
        print(f"  {name}")
else:
    print("  ✓ No Inf parameters")

if zero_var_bn:
    print(f"\n⚠️  Found {len(zero_var_bn)} BatchNorm layers with zero variance:")
    for name, count in zero_var_bn[:10]:
        print(f"  {name}: {count} zeros")
else:
    print("  ✓ No zero variance in BatchNorm")

# Print sample BatchNorm statistics
print("\n" + "="*80)
print("Sample BatchNorm Statistics (first 3 layers)")
print("="*80)

bn_params = [(k, v) for k, v in state_dict.items() if 'running_mean' in k or 'running_var' in k]
print(f"\nFound {len(bn_params)} BatchNorm parameters")

for name, param in bn_params[:6]:
    print(f"\n{name}:")
    print(f"  Shape: {param.shape}")
    print(f"  Range: [{param.min():.6f}, {param.max():.6f}]")
    print(f"  Mean: {param.mean():.6f}")
    if 'running_var' in name and (param <= 0).any():
        print(f"  ⚠️  WARNING: {(param <= 0).sum()} non-positive values!")

print("\n" + "="*80)
