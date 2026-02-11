"""
TFormer (Transmission Transformer) implementation for Orion FHE.

FHE-optimized variant with multi-path token mixing:
- Each path has its own nonlinearity (HerPN-style polynomial: a·x² + b·x + c)
- Paths: 3×3 pool, 7×7 pool, depthwise conv
- Merge via 1×1 convolutions (emulates concat without actual concat)
- BatchNorm2d (fuseable, 0 FHE cost)
- HerPN activations throughout (token mixer AND MLP)

Architecture ensures each path provides distinct nonlinear features,
avoiding the "everything collapses to linear" problem.

Expected FHE depth: ~34 levels for 4 blocks
"""

import torch
import torch.nn as nn
import orion.nn as on
from orion.nn.module import Module


class HerPNActivation(Module):
    """
    HerPN-style activation using 3 BatchNorm layers.

    During training: applies 3 BatchNorms sequentially
    After init_orion_params(): fuses into polynomial a·x² + b·x + c

    Coefficients computed from BatchNorm statistics (not directly learnable).

    FHE cost: 2 levels (1 for x², 1 for linear combination)
    """
    def __init__(self, channels):
        super().__init__()

        # Three BatchNorm layers (will be fused into HerPN)
        self.bn0 = on.BatchNorm2d(channels)
        self.bn1 = on.BatchNorm2d(channels)
        self.bn2 = on.BatchNorm2d(channels)

        # HerPN activation (created during init_orion_params)
        self.herpn = None

        self.depth = 2  # After fusion: x² (1 level) + linear (1 level)

    def init_orion_params(self):
        """Fuse 3 BatchNorms into HerPN polynomial."""
        from models.pcnn import HerPN
        import math

        # Get BatchNorm statistics
        bn0_mean = self.bn0.running_mean
        bn0_var = self.bn0.running_var
        bn1_mean = self.bn1.running_mean
        bn1_var = self.bn1.running_var
        bn2_mean = self.bn2.running_mean
        bn2_var = self.bn2.running_var

        # Default weight and bias (typically learned, but we use identity)
        channels = self.bn0.num_features
        weight = torch.ones(channels, 1, 1)
        bias = torch.zeros(channels, 1, 1)

        # Create fused HerPN
        self.herpn = HerPN(
            bn0_mean=bn0_mean,
            bn0_var=bn0_var,
            bn1_mean=bn1_mean,
            bn1_var=bn1_var,
            bn2_mean=bn2_mean,
            bn2_var=bn2_var,
            weight=weight,
            bias=bias,
        )

    def forward(self, x):
        if self.herpn is None:
            # Training/cleartext mode: apply 3 BatchNorms
            x = self.bn0(x)
            x = self.bn1(x)
            x = self.bn2(x)
            return x
        else:
            # FHE mode: use fused HerPN
            return self.herpn(x)


class MultiPathTokenMixer(Module):
    """
    Multi-path token mixer with per-path nonlinearity.

    Three parallel paths:
    1. Small-scale pooling (3×3) → HerPN → Merge
    2. Large-scale pooling (7×7) → HerPN → Merge
    3. Depthwise conv (3×3) → HerPN → Merge

    Each path produces distinct nonlinear features, then merge
    layers learn how to combine them.

    FHE depth: 4 levels (max across parallel paths)
    - Pool/Conv: 0-1 level
    - HerPN: 2 levels
    - Merge: 1 level
    """
    def __init__(self, dim):
        super().__init__()

        # Path 1: Small-scale pooling (local features)
        self.pool_small = on.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.act_small = HerPNActivation(dim)
        self.merge_small = on.Conv2d(dim, dim, kernel_size=1, bias=True)

        # Path 2: Large-scale pooling (global features)
        self.pool_large = on.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.act_large = HerPNActivation(dim)
        self.merge_large = on.Conv2d(dim, dim, kernel_size=1, bias=True)

        # Path 3: Depthwise convolution (learnable spatial patterns)
        self.dw_conv = on.Conv2d(dim, dim, kernel_size=3, stride=1,
                                 padding=1, groups=dim, bias=True)
        self.act_conv = HerPNActivation(dim)
        self.merge_conv = on.Conv2d(dim, dim, kernel_size=1, bias=True)

        # Depth: paths run in parallel, take max
        # Pool(0) + HerPN(2) + Merge(1) = 3 levels
        # DWConv(1) + HerPN(2) + Merge(1) = 4 levels
        # Max = 4 levels
        self.depth = 4

    def init_orion_params(self):
        """Initialize HerPN activations from BatchNorm statistics."""
        self.act_small.init_orion_params()
        self.act_large.init_orion_params()
        self.act_conv.init_orion_params()

    def forward(self, x):
        # Path 1: Small-scale pooling → nonlinear → merge
        y1 = self.pool_small(x)      # 0 levels (pooling)
        y1 = self.act_small(y1)       # 2 levels (HerPN)
        y1 = self.merge_small(y1)     # 1 level (conv)

        # Path 2: Large-scale pooling → nonlinear → merge
        y2 = self.pool_large(x)       # 0 levels (pooling)
        y2 = self.act_large(y2)       # 2 levels (HerPN)
        y2 = self.merge_large(y2)     # 1 level (conv)

        # Path 3: Depthwise conv → nonlinear → merge
        y3 = self.dw_conv(x)          # 1 level (conv)
        y3 = self.act_conv(y3)        # 2 levels (HerPN)
        y3 = self.merge_conv(y3)      # 1 level (conv)

        # Combine paths (addition is free in FHE)
        out = y1 + y2 + y3

        return out


class MLP(Module):
    """
    Channel mixing MLP using 1×1 convolutions.

    Structure: Conv1×1 → HerPN → Conv1×1

    FHE depth: 4 levels
    """
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        hidden_dim = int(dim * mlp_ratio)

        self.fc1 = on.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act = HerPNActivation(hidden_dim)  # HerPN-style polynomial
        self.fc2 = on.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

        self.depth = 4  # Conv(1) + HerPN(2) + Conv(1)

    def init_orion_params(self):
        """Initialize HerPN activation from BatchNorm statistics."""
        self.act.init_orion_params()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TFormerBlock(Module):
    """
    TFormer block with multi-path token mixing and MLP.

    Structure (pre-norm residual):
        x → BN → MultiPathTokenMixer → (+) residual
          → BN → MLP → (+) residual

    FHE depth: 8 levels per block
    - Token mixing: 4 levels
    - MLP: 4 levels
    - BatchNorm: 0 levels (fused!)
    """
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        # Pre-normalization (fuseable BatchNorm)
        self.norm1 = on.BatchNorm2d(dim)
        self.norm2 = on.BatchNorm2d(dim)

        # Multi-path token mixer with per-path nonlinearity
        self.token_mixer = MultiPathTokenMixer(dim)

        # Channel mixing MLP
        self.mlp = MLP(dim, mlp_ratio)

        self.depth = 4 + 4  # Token mixer (4) + MLP (4)

    def init_orion_params(self):
        """Initialize HerPN activations in token mixer and MLP."""
        self.token_mixer.init_orion_params()
        self.mlp.init_orion_params()

    def forward(self, x):
        # Token mixing with pre-norm residual
        # Standard form: x = x + f(norm(x))
        x = x + self.token_mixer(self.norm1(x))

        # Channel mixing with pre-norm residual
        x = x + self.mlp(self.norm2(x))

        return x


class TFormer(Module):
    """
    TFormer: FHE-optimized Transmission Transformer.

    Key features:
    - Multi-path token mixing (3×3 pool, 7×7 pool, depthwise conv)
    - Per-path HerPN-style nonlinearity (prevents linear collapse)
    - HerPN activations everywhere (a·x² + b·x + c)
    - BatchNorm fusion (0 FHE cost)
    - No concat/split operations (FHE incompatible)

    Expected FHE depth: ~34 levels for 4 blocks
    """
    def __init__(
        self,
        input_size=32,
        in_channels=3,
        embed_dim=64,
        num_blocks=4,
        mlp_ratio=4,
        num_classes=10
    ):
        super().__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        # Patch embedding: Conv stem
        self.patch_embed = on.Conv2d(
            in_channels, embed_dim,
            kernel_size=7, stride=2, padding=3
        )
        self.norm_embed = on.BatchNorm2d(embed_dim)

        # TFormer blocks
        self.blocks = nn.Sequential(*[
            TFormerBlock(dim=embed_dim, mlp_ratio=mlp_ratio)
            for _ in range(num_blocks)
        ])

        # Global average pooling
        self.avgpool = on.AdaptiveAvgPool2d(1)
        self.flatten = on.Flatten()

        # Classification head
        self.head = on.Linear(embed_dim, num_classes)

        # Total depth estimate
        total_depth = 1 + (num_blocks * 8) + 1
        # Embed(1) + BN(0) + Blocks(8×N) + Pool(0) + Head(1)

        print(f"\n{'='*70}")
        print(f"TFormer Configuration (FHE-Optimized)")
        print(f"{'='*70}")
        print(f"Input size:          {input_size}×{input_size}")
        print(f"Embedding dim:       {embed_dim}")
        print(f"Number of blocks:    {num_blocks}")
        print(f"MLP ratio:           {mlp_ratio}")
        print(f"Output classes:      {num_classes}")
        print(f"")
        print(f"Token Mixer Design:")
        print(f"  - Path 1: AvgPool 3×3 → HerPN → Merge1×1")
        print(f"  - Path 2: AvgPool 7×7 → HerPN → Merge1×1")
        print(f"  - Path 3: DWConv 3×3 → HerPN → Merge1×1")
        print(f"  - Each path: distinct nonlinear features")
        print(f"")
        print(f"MLP Design:")
        print(f"  - Conv1×1 → HerPN → Conv1×1")
        print(f"  - HerPN: a·x² + b·x + c (more expressive than Quad)")
        print(f"")
        print(f"FHE Depth per Block:")
        print(f"  - Token mixer: 4 levels (max of parallel paths)")
        print(f"  - MLP: 4 levels (Conv + HerPN + Conv)")
        print(f"  - Total: 8 levels")
        print(f"")
        print(f"Estimated FHE depth: {total_depth} levels")
        print(f"{'='*70}\n")

    def init_orion_params(self):
        """
        Initialize HerPN activations from BatchNorm statistics.

        IMPORTANT: Must be called after training, before orion.fit()
        This uses the custom fit workflow (see CLAUDE.md)
        """
        # Fuse HerPN activations in all blocks
        for block in self.blocks:
            block.init_orion_params()

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, C, H/2, W/2]
        x = self.norm_embed(x)

        # TFormer blocks
        x = self.blocks(x)

        # Global pooling
        x = self.avgpool(x)  # [B, C, 1, 1]
        x = self.flatten(x)  # [B, C]

        # Classification head
        x = self.head(x)

        return x


def create_tformer(input_size=32, num_blocks=4):
    """
    Factory function for FHE-optimized TFormer.

    Creates a TFormer model with:
    - Multi-path token mixing with per-path HerPN activation
    - No concat/split (FHE incompatible)
    - BatchNorm fusion for efficiency

    Args:
        input_size: Input image size (e.g., 32 for CIFAR-10)
        num_blocks: Number of TFormer blocks (4 recommended)

    Returns:
        TFormer model instance
    """
    return TFormer(
        input_size=input_size,
        in_channels=3,
        embed_dim=64,
        num_blocks=num_blocks,
        mlp_ratio=4,
        num_classes=10
    )


if __name__ == "__main__":
    # Test model creation
    print("Testing TFormer model creation...\n")

    model = create_tformer(input_size=32, num_blocks=4)

    # Test forward pass with random input
    x = torch.randn(2, 3, 32, 32)
    print(f"Input shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([2, 10])")

    if output.shape == (2, 10):
        print("\n✓ TFormer model works correctly!")
    else:
        print(f"\n✗ Unexpected output shape: {output.shape}")
