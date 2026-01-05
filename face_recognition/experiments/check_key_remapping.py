"""
Check what keys are being remapped.
"""
import torch

checkpoint_path = "face_recognition/checkpoints/backbone-64x64.ckpt"
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = ckpt['backbone']

# Get keys for patch 0, layer 0 (which maps to layer1 in Orion)
patch_idx = 0
prefix = f'nets.{patch_idx}.'

# Filter for layer 0 herpn1 keys
layer0_herpn1_keys = [k for k in state_dict.keys() if k.startswith(f'{prefix}layers.0.herpn1')]

print("="*80)
print("CHECKPOINT KEYS for nets.0.layers.0.herpn1:")
print("="*80)
for key in sorted(layer0_herpn1_keys):
    value = state_dict[key]
    shape_str = str(tuple(value.shape)) if hasattr(value, 'shape') else str(value)
    print(f"{key:60s} {shape_str:20s} {value[:3] if len(value.shape) == 1 else 'tensor'}")

# Now simulate the remapping logic
print("\n" + "="*80)
print("REMAPPING:")
print("="*80)

for key in sorted(layer0_herpn1_keys):
    key_no_prefix = key[len(prefix):]
    parts = key_no_prefix.split('.')
    layer_num = int(parts[1])  # 0
    rest = '.'.join(parts[2:])  # e.g., "herpn1.bn0.running_mean"

    orion_layer_name = f'layer{layer_num + 1}'  # layer1

    # Handle HerPN BatchNorm remapping
    if 'herpn1' in rest:
        rest_new = rest.replace('herpn1.bn0', 'bn0_1')
        rest_new = rest_new.replace('herpn1.bn1', 'bn1_1')
        rest_new = rest_new.replace('herpn1.bn2', 'bn2_1')
    else:
        rest_new = rest

    new_key = f'{orion_layer_name}.{rest_new}'

    print(f"{key_no_prefix:50s} → {new_key}")
