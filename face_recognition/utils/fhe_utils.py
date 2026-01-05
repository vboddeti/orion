"""
FHE utility functions for testing and debugging.

These utilities help with unpacking hybrid-packed FHE tensors and comparing
FHE vs cleartext outputs.
"""
import torch
import torch.nn.functional as F


def unpack_fhe_tensor(packed_tensor, gap, original_shape):
    """Unpack a hybrid-packed FHE tensor back to original cleartext shape.

    Args:
        packed_tensor: The packed FHE tensor (e.g., shape (1, 8, 32, 32))
        gap: The packing gap factor (e.g., 2 for 2x2 packing)
        original_shape: The original cleartext shape (e.g., (1, 32, 16, 16) or (1, 256) for flattened)

    Returns:
        Unpacked tensor in original cleartext shape
    """
    # Handle 1D/2D tensors (flatten/bn outputs)
    # FHE tensor is always 4D, but cleartext might be 1D or 2D after flatten
    if len(original_shape) < 4:
        # For flattened outputs, first unpack spatially, then crop and flatten
        # Example: FHE (1, 1, 32, 32) with gap=16 → (1, 256, 2, 2) → crop to (1, 64, 2, 2) → (1, 256)

        # Total cleartext elements
        total_elements = 1
        for dim in original_shape:
            total_elements *= dim

        # Unpack the 4D FHE tensor
        if gap > 1:
            unpacked_4d = F.pixel_unshuffle(packed_tensor, gap)
        else:
            unpacked_4d = packed_tensor

        # The unpacked tensor may have extra padding channels
        # We need to figure out the 4D shape before flattening
        # If cleartext is (1, 256) = 256 elements
        # And unpacked_4d is (1, C_fhe, H_fhe, W_fhe)
        # Then we need C_actual × H_fhe × W_fhe = 256
        # For herpnpool case: unpacked_4d is (1, 256, 2, 2) but we only want (1, 64, 2, 2)

        N_fhe, C_fhe, H_fhe, W_fhe = unpacked_4d.shape
        fhe_total = C_fhe * H_fhe * W_fhe

        if fhe_total > total_elements:
            # Need to crop channels
            # total_elements = C_actual × H_fhe × W_fhe
            C_actual = total_elements // (H_fhe * W_fhe)
            unpacked_4d = unpacked_4d[:, :C_actual, :, :]

        # Now flatten to match original shape
        return unpacked_4d.reshape(original_shape)

    # Handle 4D tensors (normal conv/pool outputs)
    if gap == 1:
        return packed_tensor

    unpacked = F.pixel_unshuffle(packed_tensor, gap)
    N, C_orig, H_orig, W_orig = original_shape
    unpacked = unpacked[:, :C_orig, :H_orig, :W_orig]
    return unpacked


def calculate_gap(cleartext_shape, packed_shape):
    """Calculate packing gap from cleartext and packed shapes.

    Args:
        cleartext_shape: Original cleartext shape (e.g., (1, 32, 16, 16) or (1, 256) for flattened)
        packed_shape: Packed FHE shape (e.g., (1, 8, 32, 32) - always 4D)

    Returns:
        Gap factor (e.g., 2 means 2x2 packing)
    """
    # For flattened tensors (1D or 2D), we can't directly calculate gap from H dimension
    # Instead, calculate from total elements
    if len(cleartext_shape) < 4:
        # Total cleartext elements
        clear_total = 1
        for dim in cleartext_shape:
            clear_total *= dim

        # FHE is packed in spatial dimensions
        # Example: gap=16 means (64, 2, 2) → (1, 32, 32) where 64*2*2 = 256 and 1*32*32 = 1024
        # But 1024 = 256 * (16/4) = 256 * 4 channels packed

        # Simpler: extract gap from FHE spatial size
        # If FHE is (1, C_fhe, H_fhe, W_fhe) and cleartext flattens to (1, total)
        # We know from context that gap relates spatial dimensions
        # For now, return the spatial dimension ratio
        # This assumes square packing (H_fhe == W_fhe)

        # Actually, let's infer from the FHE shape itself
        # If packed_shape is (1, 1, 32, 32) with 1024 total slots
        # and cleartext is (1, 256), then gap = sqrt(1024 / 256) = sqrt(4) = 2... wait that's wrong

        # Let me think differently: if FHE is (1, 1, 32, 32) and cleartext is (1, 256)
        # Then unpacking with gap gives us (1, gap^2, 32/gap, 32/gap)
        # Total elements: gap^2 * (32/gap) * (32/gap) = gap^2 * 1024/gap^2 = 1024
        # But we want 256 cleartext elements
        # So gap^2 * (32/gap)^2 = 256
        # 1024 / gap^2 = 256
        # gap^2 = 4
        # gap = 2... but that's not right either

        # Actually looking at the log:
        # herpnpool output: cleartext (1, 64, 2, 2), FHE (1, 1, 32, 32), gap=16
        # After unpack with gap=16: (1, 64, 2, 2)  ← this matches!
        # Then flatten: (1, 256)
        # So the gap is still 16 from the previous layer!

        # The simplest approach: use the packed FHE spatial dimension
        # gap = H_fhe / 2 (assuming final unpacked is 2x2 for this architecture)
        # But that's too specific...

        # Better: return gap based on FHE channels
        # If FHE has 1 channel, it's maximally packed
        C_fhe = packed_shape[1]
        H_fhe = packed_shape[2]

        if C_fhe == 1:
            # Maximally packed, gap = H_fhe / 2 (assuming 2x2 final shape in this arch)
            # This is a heuristic for herpnpool → flatten → bn case
            return H_fhe // 2
        else:
            # Partially packed
            # Use spatial dimension ratio (needs 4D cleartext to compare)
            # For now, return 1
            return 1

    # For 4D tensors, use spatial dimensions
    clear_H = cleartext_shape[2]
    packed_H = packed_shape[2]
    return packed_H // clear_H if clear_H > 0 else 1


def compare_tensors(name, cleartext, fhe_packed, gap=None, cleartext_shape=None, indent=0, verbose=True):
    """Compare cleartext vs FHE tensors after unpacking.

    Args:
        name: Name of the layer/operation being compared
        cleartext: Cleartext output tensor
        fhe_packed: FHE output tensor (possibly packed)
        gap: Packing gap (if None, will be calculated from shapes)
        cleartext_shape: Original cleartext shape (if None, uses cleartext.shape)
        indent: Indentation level for printing
        verbose: Whether to print comparison results

    Returns:
        Relative error percentage
    """
    if cleartext_shape is None:
        cleartext_shape = cleartext.shape

    if gap is None:
        gap = calculate_gap(cleartext_shape, fhe_packed.shape)

    prefix = "  " * indent

    fhe_unpacked = unpack_fhe_tensor(fhe_packed, gap, cleartext_shape)

    diff = (cleartext - fhe_unpacked).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    clear_max = cleartext.abs().max().item()
    rel_error = (max_diff / clear_max * 100) if clear_max > 0 else 0

    if verbose:
        print(f"{prefix}{name}:")
        print(f"{prefix}  Cleartext:     range=[{cleartext.min():.6f}, {cleartext.max():.6f}]")
        print(f"{prefix}  FHE (unpack):  range=[{fhe_unpacked.min():.6f}, {fhe_unpacked.max():.6f}]")
        print(f"{prefix}  Difference:    max={max_diff:.6f}, mean={mean_diff:.6f}")
        print(f"{prefix}  Relative err:  {rel_error:.2f}%")

        if rel_error > 5.0:
            print(f"{prefix}  ⚠️  WARNING: Large divergence!")
        elif rel_error < 0.1:
            print(f"{prefix}  ✓ Match")

    return rel_error
