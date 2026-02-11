"""
PoolFormer implementation for Orion FHE framework.

PoolFormer replaces attention with simple pooling operations,
making it significantly more FHE-friendly than standard transformers.

Reference: "MetaFormer is Actually What You Need for Vision" (CVPR 2022)
"""

import torch
import torch.nn as nn
import orion.nn as on
from orion.nn.module import Module


class Pooling(Module):
    """
    Pooling layer for token mixing (replaces self-attention).

    Uses average pooling to aggregate spatial information.
    FHE cost: 0 levels (pooling is just averaging, can be fused)
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = on.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2
        )
        self.depth = 0  # Pooling doesn't consume levels in FHE

    def forward(self, x):
        # Return residual: pool(x) - x
        # This captures context while preserving identity
        return self.pool(x) - x


class MLP(Module):
    """
    Channel MLP using 1x1 convolutions (FHE-compatible, no reshaping!).

    Following Orion's pattern: use spatial operations throughout.
    Structure: Conv1x1 -> Quad -> Conv1x1
    FHE cost: 3 levels (Conv: 1, Quad: 1, Conv: 1)
    """
    def __init__(self, dim, hidden_dim=None, activation='quad'):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        # Use 1x1 convolutions instead of Linear to stay in spatial format
        # This avoids reshaping and gap issues!
        self.fc1 = on.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)

        # Use simple quadratic activation
        self.activation = on.Quad() if activation == 'quad' else None

        self.fc2 = on.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

        # Depth: Conv (1) + Quad (1) + Conv (1) = 3
        self.depth = 3

    def forward(self, x):
        # x: [B, C, H, W] - stays in spatial format!
        x = self.fc1(x)       # [B, hidden_dim, H, W]
        if self.activation is not None:
            x = self.activation(x)
        x = self.fc2(x)       # [B, dim, H, W]
        return x


class PoolFormerBlock(Module):
    """
    PoolFormer block: Pooling + MLP with residual connections.

    Architecture:
        x -> Pooling ─> (+) -> MLP ─> (+) -> output
         └───────────────┘      └─────────┘

    FHE Depth: ~3-4 levels per block
    - Token mixing (Pooling): 0 levels
    - MLP: 3 levels
    - Total: 3 levels
    """
    def __init__(self, dim, pool_size=3, mlp_ratio=4, activation='quad'):
        super().__init__()

        # Token mixing: Pooling (replaces self-attention)
        self.token_mixer = Pooling(pool_size=pool_size)

        # Channel mixing: MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, activation=activation)

        # Note: We skip normalization for now to reduce complexity
        # In full implementation, would use BatchNorm2d (fuseable)

        self.depth = 3  # Pooling (0) + MLP (3)

    def forward(self, x):
        # x shape: [B, C, H, W]

        # Token mixing with residual
        x = x + self.token_mixer(x)

        # Channel mixing with residual
        # MLP now uses 1x1 convs, so no reshaping needed!
        x = x + self.mlp(x)

        return x


class BasicPoolFormer(Module):
    """
    Basic PoolFormer model for FHE proof-of-concept.

    Simple architecture:
    - Patch embedding (Conv stem)
    - 4 PoolFormer blocks
    - Global average pooling
    - Classification head

    Expected FHE depth: ~15-18 levels
    """
    def __init__(
        self,
        input_size=32,
        in_channels=3,
        embed_dim=64,
        num_blocks=4,
        pool_size=3,
        mlp_ratio=4,
        num_classes=10
    ):
        super().__init__()

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        # Patch embedding: Simple conv stem
        # Depth: 1 level
        self.patch_embed = nn.Sequential(
            on.Conv2d(in_channels, embed_dim, kernel_size=7, stride=2, padding=3),
            # Skip BatchNorm for simplicity in basic version
        )
        # After conv: input_size/2 x input_size/2

        # PoolFormer blocks
        # Depth: num_blocks * 3 levels
        self.blocks = nn.Sequential(*[
            PoolFormerBlock(
                dim=embed_dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                activation='quad'
            )
            for _ in range(num_blocks)
        ])

        # Global average pooling
        # Depth: 0 levels (just averaging)
        self.avgpool = on.AdaptiveAvgPool2d(1)

        # Flatten (needed for FHE compatibility)
        self.flatten = on.Flatten()

        # Classification head
        # Depth: 1 level
        self.head = on.Linear(embed_dim, num_classes)

        # Total depth estimate
        total_depth = 1 + (num_blocks * 3) + 1

        print(f"\n{'='*70}")
        print(f"BasicPoolFormer Configuration")
        print(f"{'='*70}")
        print(f"Input size:        {input_size}×{input_size}")
        print(f"Embedding dim:     {embed_dim}")
        print(f"Number of blocks:  {num_blocks}")
        print(f"Pool size:         {pool_size}")
        print(f"MLP ratio:         {mlp_ratio}")
        print(f"Output classes:    {num_classes}")
        print(f"Estimated FHE depth: {total_depth} levels")
        print(f"{'='*70}\n")

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # [B, C, H/2, W/2]

        # PoolFormer blocks
        x = self.blocks(x)

        # Global pooling
        x = self.avgpool(x)  # [B, C, 1, 1]
        x = self.flatten(x)  # [B, C] - use on.Flatten() for FHE

        # Classification head
        x = self.head(x)     # [B, num_classes]

        return x


def create_basic_poolformer(input_size=32, num_blocks=4):
    """
    Factory function for basic PoolFormer.

    Args:
        input_size: Input image size (e.g., 32 for CIFAR-10)
        num_blocks: Number of PoolFormer blocks (4 recommended for POC)

    Returns:
        BasicPoolFormer model instance
    """
    return BasicPoolFormer(
        input_size=input_size,
        in_channels=3,
        embed_dim=64,
        num_blocks=num_blocks,
        pool_size=3,
        mlp_ratio=4,
        num_classes=10
    )


if __name__ == "__main__":
    # Test model creation
    print("Testing BasicPoolFormer model creation...\n")

    model = create_basic_poolformer(input_size=32, num_blocks=4)

    # Test forward pass with random input
    x = torch.randn(2, 3, 32, 32)
    print(f"Input shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected: torch.Size([2, 10])")

    if output.shape == (2, 10):
        print("\n✓ BasicPoolFormer model works correctly!")
    else:
        print(f"\n✗ Unexpected output shape: {output.shape}")
