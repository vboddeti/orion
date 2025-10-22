"""
Quick test for PCNN HerPNConv shortcut level assignment.
Lightweight: single HerPNConv block with shortcut; fewer BN collection passes.

Workflow:
1. Create a minimal TestBlock (or reuse HerPNConv with in_channels=4, out_channels=8)
2. Collect BN stats (5 iterations)
3. Fuse HerPN via init_orion_params()
4. Run orion.fit() and orion.compile()
5. Print level assignments and shortcut module details
"""

import torch
import orion
from models import HerPNConv


def main():
    orion.init_scheme('configs/pcnn_optionC.yml')

    # Small test network: conv -> HerPNConv (with shortcut)
    class SmallNet(orion.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = orion.nn.Conv2d(3, 4, kernel_size=3, padding=1)
            # HerPNConv expects channels; make block increase channels and stride=2
            # so that it creates a shortcut path automatically
            self.block = HerPNConv(4, 8, stride=2)

        def forward(self, x):
            x = self.conv(x)
            x = self.block(x)
            return x

        def init_orion_params(self):
            if hasattr(self.block, 'init_orion_params'):
                self.block.init_orion_params()

    model = SmallNet()
    model.eval()

    # STEP 1: Collect BN statistics (few passes)
    print("Collecting BN stats (5 iterations)...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(torch.randn(2, 3, 8, 8))
    print("Done BN stats")

    # STEP 2: Fuse HerPN
    print("Fusing HerPN (init_orion_params)")
    model.init_orion_params()

    # STEP 3: Orion fit
    print("Running orion.fit()...")
    inp = torch.randn(1, 3, 8, 8)
    orion.fit(model, inp)
    print("orion.fit() done")

    # STEP 4: Compile
    print("Running orion.compile()...")
    input_level = orion.compile(model)
    print(f"Compilation input level: {input_level}")

    # DEBUG: Print traced modules and levels
    from orion.core import scheme
    traced = scheme.trace

    print("\nTraced modules and levels:")
    for name, module in traced.named_modules():
        if name and hasattr(module, 'level'):
            print(f"  {name:30s} @ level={module.level}, depth={getattr(module, 'depth', 'N/A')}")

    # Print shortcut details
    block = model.block
    # Copy traced levels back to original model for inspection
    try:
        traced_block = traced.get_submodule('block')
        for name, mod in traced_block.named_children():
            orig_mod = getattr(block, name, None)
            if orig_mod is not None and hasattr(mod, 'level'):
                orig_mod.level = mod.level

    except Exception:
        pass

    print("\nShortcut details on original model (after copying levels):")
    if hasattr(block, 'shortcut_herpn'):
        print(f"  shortcut_herpn.level = {getattr(block.shortcut_herpn, 'level', None)}")
        print(f"  shortcut_herpn.depth = {getattr(block.shortcut_herpn, 'depth', None)}")
    if hasattr(block, 'shortcut_conv'):
        print(f"  shortcut_conv.level = {getattr(block.shortcut_conv, 'level', None)}")
        print(f"  shortcut_conv.depth = {getattr(block.shortcut_conv, 'depth', None)}")


if __name__ == '__main__':
    main()
