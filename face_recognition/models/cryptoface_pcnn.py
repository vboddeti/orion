"""
CryptoFace-compatible PCNN for Orion FHE.

This module implements a Patch-based CNN that matches CryptoFace's architecture
for encrypted face recognition inference.

Architecture:
- Configurable number of patches (4, 9, or 16)
- N backbones (one per patch) with HerPN activations
- N linear layers (256→256 each)
- Aggregation via summation
- Final normalization (ChannelSquare replacing BatchNorm1d)

Reference: CryptoFace CVPR 2025
"""
import torch
import torch.nn as nn
import orion.nn as on
from models.pcnn import Backbone, ChannelSquare, L2NormPoly


class CryptoFacePCNN(on.Module):
    """
    CryptoFace-compatible Patch-based CNN for encrypted face recognition.

    Matches the CryptoFace inference architecture:
    - Per-patch processing through identical backbones
    - Per-patch linear transformations (256→256)
    - Feature aggregation via summation
    - Final normalization

    Args:
        input_size (int): Input image size (e.g., 64, 96, 128)
        patch_size (int): Patch size (e.g., 32)
        embedding_dim (int): Output embedding dimension (default: 256)
        sqrt_weights (tuple): Normalization weights (w0, w1) for ChannelSquare

    Examples:
        >>> # CryptoFaceNet4: 64×64 input, 2×2 patches
        >>> model = CryptoFacePCNN(input_size=64, patch_size=32)
        >>>
        >>> # CryptoFaceNet9: 96×96 input, 3×3 patches
        >>> model = CryptoFacePCNN(input_size=96, patch_size=32)
        >>>
        >>> # CryptoFaceNet16: 128×128 input, 4×4 patches
        >>> model = CryptoFacePCNN(input_size=128, patch_size=32)
    """

    def __init__(
        self,
        input_size=64,
        patch_size=32,
        embedding_dim=256,
        sqrt_weights=(1.0, 1.0),
        l2_norm_coeffs=None,
    ):
        super().__init__()

        # Patch configuration
        self.input_size = input_size
        self.patch_size = patch_size
        self.H = self.W = input_size // patch_size  # Grid size
        self.N = self.H * self.W  # Number of patches

        # Backbone output dimension (always 64*2*2 = 256)
        self.backbone_dim = 64 * 2 * 2

        # Embedding dimension
        self.embedding_dim = embedding_dim

        print(f"\n{'='*70}")
        print(f"CryptoFace PCNN Configuration")
        print(f"{'='*70}")
        print(f"Input size:        {input_size}×{input_size}")
        print(f"Patch size:        {patch_size}×{patch_size}")
        print(f"Patch grid:        {self.H}×{self.W} = {self.N} patches")
        print(f"Backbone output:   {self.backbone_dim} per patch")
        print(f"Linear layers:     {self.N} × ({self.backbone_dim}→{embedding_dim})")
        print(f"Embedding dim:     {embedding_dim}")
        print(f"{'='*70}\n")

        # Create N backbone networks (one per patch)
        # Each backbone outputs 64*2*2 = 256 features
        self.nets = nn.ModuleList([
            Backbone(
                output_size=(2, 2),  # 64 channels × 2×2 = 256 features
                input_size=patch_size
            )
            for _ in range(self.N)
        ])

        # Create N linear layers (256→embedding_dim each)
        # Matches CryptoFace's per-patch linear transformations
        self.linear = nn.ModuleList([
            on.Linear(self.backbone_dim, embedding_dim)
            for _ in range(self.N)
        ])

        # Final normalization using L2 norm with polynomial approximation
        # Coefficients (a, b, c) will be loaded from checkpoint or provided
        if l2_norm_coeffs is not None:
            a, b, c = l2_norm_coeffs
            self.normalization = L2NormPoly(a, b, c, embedding_dim)
            print(f"L2 normalization: a={a:.2e}, b={b:.2e}, c={c:.2e}")
        else:
            # Default coefficients (will be overridden when loading checkpoint)
            # Using LFW default from estimate_l2_norm.py
            a, b, c = 2.41e-07, -2.44e-04, 1.09e-01
            self.normalization = L2NormPoly(a, b, c, embedding_dim)
            print(f"L2 normalization: Using default coefficients (will be loaded from checkpoint)")

        self.he_mode = False

    def forward(self, x):
        """
        Forward pass for CryptoFace PCNN.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            out: Embedding tensor of shape (B, embedding_dim)

        Processing flow:
            1. Extract N patches from input
            2. Process each patch through its backbone → 256 features
            3. Apply per-patch linear transformation → embedding_dim features
            4. Sum features across patches → aggregated embedding
            5. Apply normalization → final embedding
        """
        B = x.shape[0]
        H, W = self.H, self.W
        N = self.N
        P = self.patch_size

        # Extract patches: (B, C, H_img, W_img) → N × (B, C, P, P)
        patches = []
        for h in range(H):
            for w in range(W):
                patch = x[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
                patches.append(patch)

        # Process each patch through backbone + linear
        if not self.he_mode:
            # Cleartext mode: sequential processing
            features = []
            for i in range(N):
                feat = self.nets[i](patches[i])  # (B, 256)
                feat = self.linear[i](feat)      # (B, embedding_dim)
                features.append(feat)

            # Aggregate via summation
            out = torch.stack(features, dim=0).sum(dim=0)  # (B, embedding_dim)

        else:
            # FHE mode: patches processed independently (potentially in parallel)
            # Note: Actual parallelization happens at test/inference level
            # Here we just ensure independence for tracing
            features = []
            for i in range(N):
                feat = self.nets[i](patches[i])  # Encrypted (B, 256)
                feat = self.linear[i](feat)      # Encrypted (B, embedding_dim)
                features.append(feat)

            # Aggregate using tree reduction (encrypted)
            out = self._tree_reduce_add(features)

        # Apply final normalization
        out = self.normalization(out)

        return out

    def _tree_reduce_add(self, tensors):
        """
        Binary tree reduction for parallel addition.

        Reduces sequential addition depth from N to log2(N).
        Example with 4 patches:
            [a, b, c, d] -> [(a+b), (c+d)] -> [(a+b+c+d)]
            Depth: 2 instead of 4

        Args:
            tensors: List of tensors to sum

        Returns:
            Single tensor with all values summed
        """
        while len(tensors) > 1:
            new_tensors = []
            for i in range(0, len(tensors), 2):
                if i + 1 < len(tensors):
                    # Pair-wise addition (can be done in parallel)
                    new_tensors.append(tensors[i] + tensors[i + 1])
                else:
                    # Odd one out, carry forward
                    new_tensors.append(tensors[i])
            tensors = new_tensors
        return tensors[0]

    def init_orion_params(self):
        """
        Initialize HerPN parameters for all backbone networks.

        This must be called BEFORE orion.fit() to ensure correct fusion.
        See: docs/CUSTOM_FIT_WORKFLOW.md
        """
        for net in self.nets:
            net.init_orion_params()

    def he(self):
        """Switch model to homomorphic encryption mode."""
        self.he_mode = True
        for module in self.modules():
            if hasattr(module, 'he_mode'):
                module.he_mode = True

    def pt(self):
        """Switch model to plaintext mode."""
        self.he_mode = False
        for module in self.modules():
            if hasattr(module, 'he_mode'):
                module.he_mode = False


def create_cryptoface_pcnn(config):
    """
    Factory function to create CryptoFace PCNN from config dict.

    Args:
        config (dict): Configuration with keys:
            - input_size: Input image size
            - patch_size: Patch size
            - embedding_dim: Output embedding dimension (default: 256)
            - sqrt_weights: Normalization weights (default: (1.0, 1.0))

    Returns:
        CryptoFacePCNN model instance

    Example:
        >>> config = {
        ...     'input_size': 64,
        ...     'patch_size': 32,
        ...     'embedding_dim': 256
        ... }
        >>> model = create_cryptoface_pcnn(config)
    """
    return CryptoFacePCNN(
        input_size=config.get('input_size', 64),
        patch_size=config.get('patch_size', 32),
        embedding_dim=config.get('embedding_dim', 256),
        sqrt_weights=config.get('sqrt_weights', (1.0, 1.0)),
        l2_norm_coeffs=config.get('l2_norm_coeffs', None)
    )


# Model variants for convenience
def CryptoFaceNet4(embedding_dim=256, sqrt_weights=(1.0, 1.0), l2_norm_coeffs=None):
    """
    CryptoFaceNet4: 64×64 input, 2×2 patches (4 total)

    Architecture:
    - Input: 64×64 RGB image
    - Patches: 4 patches of 32×32 each (2×2 grid)
    - Backbones: 4 identical backbones
    - Output: 256-dim embedding

    Args:
        embedding_dim (int): Output embedding dimension (default: 256)
        sqrt_weights (tuple): Normalization weights (deprecated)
        l2_norm_coeffs (tuple): L2 norm coefficients (a, b, c)

    Returns:
        CryptoFacePCNN configured for 4 patches
    """
    return CryptoFacePCNN(
        input_size=64,
        patch_size=32,
        embedding_dim=embedding_dim,
        sqrt_weights=sqrt_weights,
        l2_norm_coeffs=l2_norm_coeffs
    )


def CryptoFaceNet9(embedding_dim=256, sqrt_weights=(1.0, 1.0), l2_norm_coeffs=None):
    """
    CryptoFaceNet9: 96×96 input, 3×3 patches (9 total)

    Architecture:
    - Input: 96×96 RGB image
    - Patches: 9 patches of 32×32 each (3×3 grid)
    - Backbones: 9 identical backbones
    - Output: 256-dim embedding

    Args:
        embedding_dim (int): Output embedding dimension (default: 256)
        sqrt_weights (tuple): Normalization weights (deprecated)
        l2_norm_coeffs (tuple): L2 norm coefficients (a, b, c)

    Returns:
        CryptoFacePCNN configured for 9 patches
    """
    return CryptoFacePCNN(
        input_size=96,
        patch_size=32,
        embedding_dim=embedding_dim,
        sqrt_weights=sqrt_weights,
        l2_norm_coeffs=l2_norm_coeffs
    )


def CryptoFaceNet16(embedding_dim=256, sqrt_weights=(1.0, 1.0), l2_norm_coeffs=None):
    """
    CryptoFaceNet16: 128×128 input, 4×4 patches (16 total)

    Architecture:
    - Input: 128×128 RGB image
    - Patches: 16 patches of 32×32 each (4×4 grid)
    - Backbones: 16 identical backbones
    - Output: 256-dim embedding

    Args:
        embedding_dim (int): Output embedding dimension (default: 256)
        sqrt_weights (tuple): Normalization weights (deprecated)
        l2_norm_coeffs (tuple): L2 norm coefficients (a, b, c)

    Returns:
        CryptoFacePCNN configured for 16 patches
    """
    return CryptoFacePCNN(
        input_size=128,
        patch_size=32,
        embedding_dim=embedding_dim,
        sqrt_weights=sqrt_weights,
        l2_norm_coeffs=l2_norm_coeffs
    )


if __name__ == "__main__":
    # Test model creation
    print("\nTesting CryptoFace PCNN Models\n")

    # Test CryptoFaceNet4
    print("="*70)
    print("Testing CryptoFaceNet4 (64×64, 4 patches)")
    print("="*70)
    model4 = CryptoFaceNet4()
    x4 = torch.randn(2, 3, 64, 64)
    out4 = model4(x4)
    print(f"Input shape:  {x4.shape}")
    print(f"Output shape: {out4.shape}")
    print(f"Expected:     torch.Size([2, 256])")
    assert out4.shape == (2, 256), f"Expected (2, 256), got {out4.shape}"
    print("✓ CryptoFaceNet4 test passed!\n")

    # Test CryptoFaceNet9
    print("="*70)
    print("Testing CryptoFaceNet9 (96×96, 9 patches)")
    print("="*70)
    model9 = CryptoFaceNet9()
    x9 = torch.randn(2, 3, 96, 96)
    out9 = model9(x9)
    print(f"Input shape:  {x9.shape}")
    print(f"Output shape: {out9.shape}")
    print(f"Expected:     torch.Size([2, 256])")
    assert out9.shape == (2, 256), f"Expected (2, 256), got {out9.shape}"
    print("✓ CryptoFaceNet9 test passed!\n")

    # Test CryptoFaceNet16
    print("="*70)
    print("Testing CryptoFaceNet16 (128×128, 16 patches)")
    print("="*70)
    model16 = CryptoFaceNet16()
    x16 = torch.randn(2, 3, 128, 128)
    out16 = model16(x16)
    print(f"Input shape:  {x16.shape}")
    print(f"Output shape: {out16.shape}")
    print(f"Expected:     torch.Size([2, 256])")
    assert out16.shape == (2, 256), f"Expected (2, 256), got {out16.shape}"
    print("✓ CryptoFaceNet16 test passed!\n")

    print("="*70)
    print("All CryptoFace PCNN models work correctly!")
    print("="*70)
