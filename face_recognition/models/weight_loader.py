"""
Weight Loader for CryptoFace Pretrained Checkpoints

This module loads CryptoFace PyTorch checkpoints and maps them to Orion's
CryptoFacePCNN model structure.

Key transformations:
1. Load backbone weights (conv, HerPN, pooling) for each patch network
2. Chunk concatenated linear.weight [256, 1024] → 4×[256, 256] per-patch weights
3. Fuse final BatchNorm statistics into per-patch linear layers
4. Handle normalization (ChannelSquare replacement for BatchNorm1d)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, Dict, Optional


def load_cryptoface_checkpoint(
    model,
    checkpoint_path: Union[str, Path],
    device: str = 'cpu',
    verbose: bool = True
) -> None:
    """
    Load CryptoFace pretrained weights into Orion CryptoFacePCNN model.

    Args:
        model: Orion CryptoFacePCNN model instance
        checkpoint_path: Path to .ckpt file
        device: Device to load checkpoint to ('cpu' or 'cuda')
        verbose: Print loading progress

    Example:
        >>> from face_recognition.models import CryptoFaceNet4
        >>> model = CryptoFaceNet4()
        >>> load_cryptoface_checkpoint(model, 'checkpoints/backbone-64x64.ckpt')
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading CryptoFace checkpoint: {checkpoint_path}")
        print(f"{'='*70}")

    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'backbone' not in ckpt:
        raise ValueError("Checkpoint must contain 'backbone' key")

    state_dict = ckpt['backbone']

    if verbose:
        print(f"✓ Loaded checkpoint with {len(state_dict)} parameters")

    # Get model info
    N = model.N  # Number of patches
    embedding_dim = model.embedding_dim  # Usually 256
    backbone_dim = model.backbone_dim  # Usually 256 (64*2*2)

    if verbose:
        print(f"\nModel configuration:")
        print(f"  Patches: {N}")
        print(f"  Backbone dim: {backbone_dim}")
        print(f"  Embedding dim: {embedding_dim}")

    # Step 1: Load backbone weights for each patch network
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 1: Loading {N} backbone networks...")
        print(f"{'='*70}")

    for i in range(N):
        _load_backbone_weights(model.nets[i], state_dict, i, verbose=verbose)

    # Step 2: Fuse BatchNorm into Linear layer, THEN chunk
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 2: Fusing BatchNorm into Linear layer...")
        print(f"{'='*70}")

    # Get all required parameters
    if 'linear.weight' not in state_dict or 'linear.bias' not in state_dict:
        raise ValueError("Checkpoint missing 'linear.weight' or 'linear.bias'")
    if 'bn.running_mean' not in state_dict or 'bn.running_var' not in state_dict:
        raise ValueError("Checkpoint missing BatchNorm statistics")

    full_weight = state_dict['linear.weight']  # [embedding_dim, N*backbone_dim]
    full_bias = state_dict['linear.bias']      # [embedding_dim]
    bn_mean = state_dict['bn.running_mean']    # [embedding_dim]
    bn_var = state_dict['bn.running_var']      # [embedding_dim]
    bn_eps = model.normalization.eps

    if verbose:
        print(f"  Linear weight shape: {full_weight.shape}")
        print(f"  Linear bias shape: {full_bias.shape}")
        print(f"  Expected: [{embedding_dim}, {N * backbone_dim}]")
        print(f"  BatchNorm mean range: [{bn_mean.min():.4f}, {bn_mean.max():.4f}]")
        print(f"  BatchNorm var range:  [{bn_var.min():.4f}, {bn_var.max():.4f}]")

    if full_weight.shape != (embedding_dim, N * backbone_dim):
        raise ValueError(
            f"Linear weight shape mismatch: expected [{embedding_dim}, {N*backbone_dim}], "
            f"got {full_weight.shape}"
        )

    # Fuse BatchNorm into Linear layer (following CryptoFace pcnn.py:78-83)
    # This normalizes the linear layer's output by the BatchNorm statistics
    if verbose:
        print(f"\n  Fusing BatchNorm statistics into linear layer...")

    # Normalize weight: weight_fused = weight / sqrt(var + eps)
    # Note: We normalize along output dimension (columns)
    weight_fused = torch.divide(full_weight.T, torch.sqrt(bn_var + bn_eps))
    weight_fused = weight_fused.T  # [embedding_dim, N*backbone_dim]

    # Normalize bias: bias_fused = (bias - mean) / sqrt(var + eps)
    bias_fused = torch.divide(full_bias - bn_mean, torch.sqrt(bn_var + bn_eps))

    if verbose:
        print(f"    ✓ Applied BatchNorm normalization to weights and bias")

    # Step 3: Chunk the FUSED weights and divide bias by N
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 3: Chunking fused weights into {N} per-patch layers...")
        print(f"{'='*70}")

    # Divide bias by number of patches (following CryptoFace pcnn.py:85)
    bias_fused = bias_fused / N

    if verbose:
        print(f"  Divided bias by {N} patches")
        print(f"  Fused bias range: [{bias_fused.min():.4f}, {bias_fused.max():.4f}]")

    # Chunk fused weight into N pieces along dim=1
    chunked_weights = torch.chunk(weight_fused, N, dim=1)  # List of N × [embedding_dim, backbone_dim]

    if verbose:
        print(f"  Chunked fused weights into {len(chunked_weights)} pieces:")
        for i, w in enumerate(chunked_weights):
            print(f"    Patch {i}: {w.shape}")

    # Load fused weights and bias into per-patch linear layers
    for i in range(N):
        model.linear[i].weight.data = chunked_weights[i].clone()
        model.linear[i].bias.data = bias_fused.clone()  # All patches get same fused bias

    if verbose:
        print(f"  ✓ Loaded {N} per-patch linear layers with fused BatchNorm")

    # Step 4: Set BatchNorm to identity (since it's already fused into linear layers)
    if verbose:
        print(f"\n{'='*70}")
        print(f"Step 4: Setting final BatchNorm to identity...")
        print(f"{'='*70}")

    # Set BatchNorm to identity transformation (mean=0, var=1)
    # This ensures the fused weights work correctly without double-normalization
    model.normalization.running_mean.data = torch.zeros_like(bn_mean)
    model.normalization.running_var.data = torch.ones_like(bn_var)

    if verbose:
        print(f"  Set BatchNorm running_mean = 0 (identity)")
        print(f"  Set BatchNorm running_var = 1 (identity)")
        print(f"  ✓ BatchNorm is now identity (fusion complete)")

    # Set model to eval mode to avoid BatchNorm training issues
    model.eval()

    if verbose:
        print(f"\n{'='*70}")
        print(f"✓ Successfully loaded CryptoFace checkpoint!")
        print(f"  Model set to eval() mode")
        print(f"{'='*70}\n")


def _load_backbone_weights(
    backbone_module,
    state_dict: Dict[str, torch.Tensor],
    patch_idx: int,
    verbose: bool = True
) -> None:
    """
    Load weights for a single Backbone network from CryptoFace checkpoint.

    Remaps CryptoFace naming convention to Orion naming:
    - CryptoFace: nets.{i}.layers.{j}.herpn{k}.bn{n}.{param}
    - Orion:      nets.{i}.layer{j+1}.bn{n}_{k}.{param}

    Example mappings:
    - nets.0.layers.0.herpn1.bn0.running_mean → nets.0.layer1.bn0_1.running_mean
    - nets.0.layers.0.herpn2.bn1.running_var → nets.0.layer1.bn1_2.running_var
    - nets.0.herpnpool.herpn.bn0.running_mean → nets.0.herpnpool.bn0.running_mean

    Also extracts herpn.weight and herpn.bias and stores them as attributes on
    the layer modules for use in init_orion_params().

    Args:
        backbone_module: Orion Backbone module
        state_dict: CryptoFace state_dict
        patch_idx: Index of the patch (0, 1, 2, ...)
        verbose: Print loading progress
    """
    prefix = f'nets.{patch_idx}.'

    # Filter keys for this patch
    patch_keys = {k: v for k, v in state_dict.items() if k.startswith(prefix)}

    if verbose:
        print(f"  Patch {patch_idx}: {len(patch_keys)} parameters")

    # Create a state dict with keys stripped of prefix and remapped
    # CryptoFace naming: nets.{i}.layers.{j}.herpn{k}.bn{n}
    # Orion naming:      nets.{i}.layer{j+1}.bn{n}_{k}
    backbone_state = {}

    # Store herpn.weight and herpn.bias separately for later use
    herpn_params = {}

    for key, value in patch_keys.items():
        key_no_prefix = key[len(prefix):]

        # Remap CryptoFace keys to Orion keys
        # CryptoFace: layers.0.herpn1.bn0 → Orion: layer1.bn0_1
        # CryptoFace: layers.0.herpn2.bn0 → Orion: layer1.bn0_2
        # etc.

        # Handle layer keys (layers.{j} → layer{j+1})
        if key_no_prefix.startswith('layers.'):
            # Extract layer number and rest of key
            parts = key_no_prefix.split('.')
            layer_num = int(parts[1])  # e.g., 0, 1, 2, 3, 4
            rest = '.'.join(parts[2:])  # e.g., "herpn1.bn0.running_mean"

            # CryptoFace uses layers.0-4 → Orion uses layer1-5
            orion_layer_name = f'layer{layer_num + 1}'

            # Handle HerPN weight and bias (store separately)
            if rest == 'herpn1.weight':
                herpn_params[f'{orion_layer_name}.herpn1_weight'] = value
                continue  # Don't add to backbone_state
            elif rest == 'herpn1.bias':
                herpn_params[f'{orion_layer_name}.herpn1_bias'] = value
                continue
            elif rest == 'herpn2.weight':
                herpn_params[f'{orion_layer_name}.herpn2_weight'] = value
                continue
            elif rest == 'herpn2.bias':
                herpn_params[f'{orion_layer_name}.herpn2_bias'] = value
                continue
            # Handle shortcut remapping (CryptoFace: shortcut.0/shortcut.1 → Orion: shortcut_conv/shortcut_bn)
            elif 'shortcut.0' in rest:
                # shortcut.0.weight → shortcut_conv.weight
                rest = rest.replace('shortcut.0', 'shortcut_conv')
            elif 'shortcut.1' in rest:
                # shortcut.1.running_mean → shortcut_bn.running_mean
                rest = rest.replace('shortcut.1', 'shortcut_bn')
            # Handle HerPN BatchNorm remapping
            elif 'herpn1' in rest:
                # herpn1.bn0 → bn0_1, herpn1.bn1 → bn1_1, herpn1.bn2 → bn2_1
                rest = rest.replace('herpn1.bn0', 'bn0_1')
                rest = rest.replace('herpn1.bn1', 'bn1_1')
                rest = rest.replace('herpn1.bn2', 'bn2_1')
            elif 'herpn2' in rest:
                # herpn2.bn0 → bn0_2, herpn2.bn1 → bn1_2, herpn2.bn2 → bn2_2
                rest = rest.replace('herpn2.bn0', 'bn0_2')
                rest = rest.replace('herpn2.bn1', 'bn1_2')
                rest = rest.replace('herpn2.bn2', 'bn2_2')

            new_key = f'{orion_layer_name}.{rest}'
        else:
            # Handle herpnpool weight and bias
            if key_no_prefix == 'herpnpool.herpn.weight':
                herpn_params['herpnpool.herpn_weight'] = value
                continue
            elif key_no_prefix == 'herpnpool.herpn.bias':
                herpn_params['herpnpool.herpn_bias'] = value
                continue
            # Handle herpnpool remapping
            elif 'herpnpool.herpn.bn0' in key_no_prefix:
                new_key = key_no_prefix.replace('herpnpool.herpn.bn0', 'herpnpool.bn0')
            elif 'herpnpool.herpn.bn1' in key_no_prefix:
                new_key = key_no_prefix.replace('herpnpool.herpn.bn1', 'herpnpool.bn1')
            elif 'herpnpool.herpn.bn2' in key_no_prefix:
                new_key = key_no_prefix.replace('herpnpool.herpn.bn2', 'herpnpool.bn2')
            else:
                # Keep original key
                new_key = key_no_prefix

        backbone_state[new_key] = value

    # Load into backbone (strict=False to allow missing keys like jigsaw)
    missing_keys, unexpected_keys = backbone_module.load_state_dict(
        backbone_state, strict=False
    )

    # Store herpn.weight and herpn.bias as attributes on the layer modules
    # These will be used by init_orion_params() instead of ones/zeros
    for key, value in herpn_params.items():
        parts = key.split('.')
        if len(parts) == 2:
            # layer{j}.herpn{k}_{param} or herpnpool.herpn_{param}
            layer_name, param_name = parts
            layer_module = getattr(backbone_module, layer_name)

            # Move value to same device as the module's parameters
            # This ensures HerPN weights are on the correct device (CPU or CUDA)
            try:
                device = next(layer_module.parameters()).device
                value = value.to(device)
            except StopIteration:
                # Module has no parameters, keep value on its current device
                pass

            setattr(layer_module, param_name, value)
            if verbose:
                print(f"    Stored {key}: {value.shape}")

    if verbose and (missing_keys or unexpected_keys):
        if missing_keys:
            # Filter out jigsaw keys (expected to be missing)
            missing_no_jigsaw = [k for k in missing_keys if 'jigsaw' not in k]
            if missing_no_jigsaw:
                print(f"    Missing keys: {missing_no_jigsaw[:3]}..." if len(missing_no_jigsaw) > 3 else f"    Missing keys: {missing_no_jigsaw}")
        if unexpected_keys:
            print(f"    Unexpected keys: {unexpected_keys[:3]}..." if len(unexpected_keys) > 3 else f"    Unexpected keys: {unexpected_keys}")


def load_checkpoint_for_config(
    model,
    input_size: int,
    checkpoint_dir: Union[str, Path] = "face_recognition/checkpoints",
    device: str = 'cpu',
    verbose: bool = True
) -> None:
    """
    Load checkpoint based on input size.

    Args:
        model: Orion CryptoFacePCNN model
        input_size: Input image size (64, 96, or 128)
        checkpoint_dir: Directory containing checkpoints
        device: Device to load to
        verbose: Print progress

    Example:
        >>> model = CryptoFaceNet4()
        >>> load_checkpoint_for_config(model, input_size=64)
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_map = {
        64: "backbone-64x64.ckpt",
        96: "backbone-96x96.ckpt",
        128: "backbone-128x128.ckpt",
    }

    if input_size not in checkpoint_map:
        raise ValueError(f"Invalid input_size: {input_size}. Must be 64, 96, or 128.")

    checkpoint_path = checkpoint_dir / checkpoint_map[input_size]
    load_cryptoface_checkpoint(model, checkpoint_path, device=device, verbose=verbose)


if __name__ == "__main__":
    # Test weight loading
    import sys
    sys.path.append('/research/hal-vishnu/code/orion-fhe')

    from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4

    print("\n" + "="*70)
    print("Testing CryptoFace Weight Loading")
    print("="*70)

    # Create model
    model = CryptoFaceNet4()

    # Load weights
    load_checkpoint_for_config(model, input_size=64)

    # Test forward pass
    print("\nTesting forward pass...")
    import torch
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")

    print("\n" + "="*70)
    print("✓ Weight loading test passed!")
    print("="*70)
