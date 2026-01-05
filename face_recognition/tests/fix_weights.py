# Insert this code after line 88 (after orion.compile())

# STEP 5: Copy FHE weights from traced modules to original modules
# CRITICAL: orion.compile() creates FHE weights on the traced model (a copy).
# We must copy them to the original model before FHE inference.
print("Copying FHE weights from traced to original modules...")
from orion.core import scheme
traced_model = scheme.trace

def copy_herpn_weights(original_herpn, traced_path):
    """Copy FHE weights from traced HerPN/ChannelSquare to original."""
    try:
        traced_herpn = traced_model.get_submodule(traced_path)
        if hasattr(traced_herpn, 'w1_fhe'):
            print(f"  Copying w1_fhe and w0_fhe from {traced_path}")
            original_herpn.w1_fhe = traced_herpn.w1_fhe
            original_herpn.w0_fhe = traced_herpn.w0_fhe
        else:
            print(f"  Warning: {traced_path} has no w1_fhe")
    except Exception as e:
        print(f"  Warning: Could not copy weights for {traced_path}: {e}")

# Copy weights for layer1 (only layer in PartialBackbone)
if hasattr(model.layer1, 'herpn1') and model.layer1.herpn1 is not None:
    copy_herpn_weights(model.layer1.herpn1, 'layer1.herpn1')
if hasattr(model.layer1, 'herpn2') and model.layer1.herpn2 is not None:
    copy_herpn_weights(model.layer1.herpn2, 'layer1.herpn2')

# Copy ScaleModule weights (shortcut_scale)
if hasattr(model.layer1, 'shortcut_scale') and model.layer1.shortcut_scale is not None:
    try:
        traced_scale = traced_model.get_submodule('layer1.shortcut_scale')
        if hasattr(traced_scale, 'scale_fhe'):
            print(f"  Copying scale_fhe from layer1.shortcut_scale")
            model.layer1.shortcut_scale.scale_fhe = traced_scale.scale_fhe
        else:
            print(f"  Warning: layer1.shortcut_scale has no scale_fhe")
    except Exception as e:
        print(f"  Warning: Could not copy scale for layer1.shortcut_scale: {e}")

print("✓ FHE weights copied\n")
