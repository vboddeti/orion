"""
Debug: Check if all layers have herpn weight/bias loaded.
"""
import sys
import torch

sys.path.append('/research/hal-vishnu/code/orion-fhe')

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4
from face_recognition.models.weight_loader import load_checkpoint_for_config


def main():
    print("\nChecking if all layers have herpn weight/bias...\n")

    model = CryptoFaceNet4()
    load_checkpoint_for_config(model, input_size=64, verbose=False)

    backbone = model.nets[0]

    # Check each layer
    for i in range(1, 6):
        layer_name = f'layer{i}'
        layer = getattr(backbone, layer_name)

        print(f"{layer_name}:")
        has_w1 = hasattr(layer, 'herpn1_weight')
        has_b1 = hasattr(layer, 'herpn1_bias')
        has_w2 = hasattr(layer, 'herpn2_weight')
        has_b2 = hasattr(layer, 'herpn2_bias')

        print(f"  herpn1_weight: {has_w1}")
        print(f"  herpn1_bias: {has_b1}")
        print(f"  herpn2_weight: {has_w2}")
        print(f"  herpn2_bias: {has_b2}")

        if has_w1:
            print(f"  herpn1_weight shape: {layer.herpn1_weight.shape}")
        if has_b1:
            print(f"  herpn1_bias shape: {layer.herpn1_bias.shape}")

    # Check HerPNPool
    print("\nherpnpool:")
    has_w = hasattr(backbone.herpnpool, 'herpn_weight')
    has_b = hasattr(backbone.herpnpool, 'herpn_bias')
    print(f"  herpn_weight: {has_w}")
    print(f"  herpn_bias: {has_b}")

    if has_w:
        print(f"  herpn_weight shape: {backbone.herpnpool.herpn_weight.shape}")
    if has_b:
        print(f"  herpn_bias shape: {backbone.herpnpool.herpn_bias.shape}")


if __name__ == "__main__":
    main()
