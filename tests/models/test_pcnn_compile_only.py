"""
PCNN Backbone Compile-Only Test
Just runs fit() and compile() to check level assignments.
Skips FHE inference to save time.
"""

import torch
import orion
from models import PatchCNN_Backbone


def main():
    print("="*80)
    print("PCNN Backbone Compile-Only Test")
    print("="*80)
    
    orion.init_scheme('configs/pcnn_optionC.yml')
    
    # Create backbone with 5 stages
    model = PatchCNN_Backbone(num_classes=10, num_stages=5)
    model.eval()
    
    # STEP 1: Collect BN statistics
    print("\nStep 1: Collecting BatchNorm statistics (10 iterations)...")
    with torch.no_grad():
        for i in range(10):
            _ = model(torch.randn(4, 3, 8, 8))
            if (i+1) % 5 == 0:
                print(f"  Completed {i+1}/10 iterations")
    print("  ✓ BatchNorm statistics collected")
    
    # STEP 2: Fuse HerPN BEFORE orion.fit()
    print("\nStep 2: Fusing HerPN activations...")
    model.init_orion_params()
    print("  ✓ HerPN modules created and fused")
    
    # STEP 3: Orion fit
    print("\nStep 3: Running orion.fit()...")
    inp = torch.randn(1, 3, 8, 8)
    orion.fit(model, inp)
    print("  ✓ Orion fit completed")
    
    # STEP 4: Compile (this shows DEBUG output)
    print("\nStep 4: Compiling network...")
    input_level = orion.compile(model)
    print(f"\n  ✓ Compilation successful! Input level: {input_level}")
    
    # Print summary of shortcut levels
    print("\n" + "="*80)
    print("SUMMARY: Shortcut Level Assignments")
    print("="*80)
    
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5']:
        layer = getattr(model, layer_name, None)
        if layer and hasattr(layer, 'has_shortcut') and layer.has_shortcut:
            print(f"\n{layer_name.upper()} (has shortcut):")
            
            # Main path
            if hasattr(layer, 'herpn1'):
                print(f"  Main herpn1:    level={getattr(layer.herpn1, 'level', 'N/A'):>3}, depth={getattr(layer.herpn1, 'depth', 'N/A')}")
            if hasattr(layer, 'conv1'):
                print(f"  Main conv1:     level={getattr(layer.conv1, 'level', 'N/A'):>3}, depth={getattr(layer.conv1, 'depth', 'N/A')}")
            if hasattr(layer, 'herpn2'):
                print(f"  Main herpn2:    level={getattr(layer.herpn2, 'level', 'N/A'):>3}, depth={getattr(layer.herpn2, 'depth', 'N/A')}")
            if hasattr(layer, 'conv2'):
                print(f"  Main conv2:     level={getattr(layer.conv2, 'level', 'N/A'):>3}, depth={getattr(layer.conv2, 'depth', 'N/A')}")
            
            # Shortcut path
            if hasattr(layer, 'shortcut_herpn'):
                print(f"  Shortcut herpn: level={getattr(layer.shortcut_herpn, 'level', 'N/A'):>3}, depth={getattr(layer.shortcut_herpn, 'depth', 'N/A')}")
            if hasattr(layer, 'shortcut_conv'):
                print(f"  Shortcut conv:  level={getattr(layer.shortcut_conv, 'level', 'N/A'):>3}, depth={getattr(layer.shortcut_conv, 'depth', 'N/A')}")
            
            # Check alignment
            main_end = getattr(layer.conv2, 'level', None)
            shortcut_end = getattr(layer.shortcut_conv, 'level', None)
            if main_end is not None and shortcut_end is not None:
                if main_end == shortcut_end:
                    print(f"  ✓ Paths align at addition: both @ level={main_end}")
                else:
                    print(f"  ✗ MISMATCH: main@{main_end} vs shortcut@{shortcut_end}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
