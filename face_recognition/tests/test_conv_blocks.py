"""
Test to diagnose the number of blocks in layer4_conv1 and level consumption.
"""
import sys
import torch
import math

sys.path.append('/research/hal-vishnu/code/orion-fhe')

import orion
import orion.nn as on
from orion.core import scheme
from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config

def test_conv_blocks():
    """Diagnose layer4_conv1 block structure."""
    print(f"\n{'='*80}")
    print("Conv Blocks Diagnostic Test")
    print(f"{'='*80}\n")

    # Initialize scheme
    print("Initializing CKKS scheme...")
    orion.init_scheme("configs/pcnn-backbone.yml")
    slots = scheme.params.get_slots()
    print(f"Slots: {slots}\n")

    # Load model
    print("Loading model...")
    full_model = CryptoFaceNet4()
    load_checkpoint_for_config(full_model, input_size=64, verbose=False)
    backbone = full_model.nets[0]
    backbone.eval()
    print("✓ Model loaded\n")

    # Fuse BatchNorm
    print("Fusing BatchNorm...")
    backbone.init_orion_params()
    print("✓ BatchNorm fused\n")

    # Trace
    print("Tracing model...")
    inp = torch.randn(1, 3, 32, 32)
    orion.fit(backbone, inp)
    print("✓ Model traced\n")

    # Compile (this generates the transforms)
    print("Compiling model...")
    input_level = orion.compile(backbone)
    print(f"✓ Model compiled, input_level={input_level}\n")

    # Now check layer4_conv1 specifically
    traced_model = scheme.trace
    layer4_conv1 = traced_model.layer4.conv1

    print("="*80)
    print("LAYER4_CONV1 Analysis")
    print("="*80)
    print(f"Input shape (clear): {layer4_conv1.input_shape}")
    print(f"Input shape (FHE):   {layer4_conv1.fhe_input_shape}")
    print(f"Input gap:           {layer4_conv1.input_gap}")
    print(f"Output shape (clear): {layer4_conv1.output_shape}")
    print(f"Output shape (FHE):  {layer4_conv1.fhe_output_shape}")
    print(f"Output gap:          {layer4_conv1.output_gap}")
    print(f"Stride:              {layer4_conv1.stride}")
    print(f"Level:               {layer4_conv1.level}")
    print(f"Depth:               {layer4_conv1.depth}")
    print()

    # Compute expected matrix dimensions
    N, on_Co, on_Ho, on_Wo = layer4_conv1.fhe_output_shape
    N, on_Ci, on_Hi, on_Wi = layer4_conv1.fhe_input_shape

    matrix_height = on_Co * on_Ho * on_Wo
    matrix_width = on_Ci * on_Hi * on_Wi

    print(f"Toeplitz matrix dimensions:")
    print(f"  Height (output): {matrix_height}")
    print(f"  Width (input):   {matrix_width}")
    print()

    # Compute number of blocks
    num_block_rows = math.ceil(matrix_height / slots)
    num_block_cols = math.ceil(matrix_width / slots)

    print(f"Number of blocks:")
    print(f"  Rows:    {num_block_rows}")
    print(f"  Cols:    {num_block_cols}")
    print(f"  Total:   {num_block_rows * num_block_cols}")
    print()

    # Check transform IDs
    num_transforms = len(layer4_conv1.transform_ids)
    print(f"Number of transform IDs: {num_transforms}")
    print()

    # Estimated level consumption
    print(f"Expected level consumption:")
    print(f"  Depth assignment:     {layer4_conv1.depth}")
    print(f"  Actual (rescales):    {num_block_rows}")
    print(f"  Discrepancy:          {num_block_rows - layer4_conv1.depth}")
    print()

    if num_block_rows != layer4_conv1.depth:
        print("⚠️  WARNING: Depth assignment does not match actual level consumption!")
        print("   This explains the error amplification after bootstrap.")
    else:
        print("✓  Depth assignment matches actual level consumption.")

    print("="*80)

if __name__ == "__main__":
    test_conv_blocks()
