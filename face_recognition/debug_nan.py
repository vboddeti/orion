"""Debug NaN issue in cleartext inference."""
import sys
import torch
sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

print("Creating model...")
model = CryptoFaceNet4()

print("\nLoading checkpoint...")
load_checkpoint_for_config(model, input_size=64, verbose=False)

print("\nWarmup...")
model.eval()
with torch.no_grad():
    for i in range(5):
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        print(f"  Iteration {i+1}: output contains NaN = {torch.isnan(out).any()}, "
              f"range = [{out.min():.4f}, {out.max():.4f}]" if not torch.isnan(out).any()
              else f"  Iteration {i+1}: output contains NaN = True")

print("\nFusing HerPN...")
model.init_orion_params()

print("\nAfter fusion:")
with torch.no_grad():
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    print(f"  Output contains NaN = {torch.isnan(out).any()}")
    if not torch.isnan(out).any():
        print(f"  Range = [{out.min():.4f}, {out.max():.4f}]")
    else:
        # Debug layer by layer
        print("\n  Debugging layer by layer...")
        patches = []
        for h in range(2):
            for w in range(2):
                patch = x[:, :, h*32:(h+1)*32, w*32:(w+1)*32]
                patches.append(patch)

        # Test first patch
        print("\n  Testing first patch backbone...")
        net = model.nets[0]
        p = patches[0]

        print(f"    Input: NaN={torch.isnan(p).any()}, range=[{p.min():.4f}, {p.max():.4f}]")

        out = net.conv(p)
        print(f"    After conv: NaN={torch.isnan(out).any()}, range=[{out.min():.4f}, {out.max():.4f}]")

        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4', 'layer5'], 1):
            layer = getattr(net, layer_name)
            out = layer(out)
            has_nan = torch.isnan(out).any()
            print(f"    After {layer_name}: NaN={has_nan}", end="")
            if not has_nan:
                print(f", range=[{out.min():.4f}, {out.max():.4f}]")
            else:
                print()
                # Check components
                print(f"      Debugging {layer_name}...")
                # Re-run with intermediate checks
                break
