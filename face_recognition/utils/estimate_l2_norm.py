"""
Estimate L2 normalization polynomial coefficients for CryptoFace.

This script computes the polynomial approximation coefficients (a, b, c) for
L2 normalization: a*x² + b*x + c ≈ 1/√x, where x is the sum of squares of
the embedding features.

The coefficients are estimated from validation datasets using the same method
as CryptoFace/ckks.py (lines 113-128).

Usage:
    python -m face_recognition.utils.estimate_l2_norm \
        --checkpoint checkpoints/backbone-64x64.ckpt \
        --input-size 64 \
        --data-dir /home/vishnu/datastore/processed/faces_emore \
        --output-dir face_recognition/checkpoints
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from face_recognition.models.cryptoface_pcnn import CryptoFaceNet4, CryptoFaceNet9, CryptoFaceNet16
from face_recognition.models.weight_loader import load_checkpoint_for_config


def k_fold_split(n, n_splits=10):
    """Simple k-fold cross validation split.

    Args:
        n: Number of samples
        n_splits: Number of folds

    Yields:
        train_indices, test_indices for each fold
    """
    fold_size = n // n_splits
    indices = np.arange(n)

    for fold_idx in range(n_splits):
        test_start = fold_idx * fold_size
        test_end = (fold_idx + 1) * fold_size if fold_idx < n_splits - 1 else n

        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

        yield train_indices, test_indices


# Dataset names for validation
FACE_DATASETS = {
    0: "lfw",
    1: "cfp_fp",
    2: "cplfw",
    3: "agedb_30",
    4: "calfw"
}


def load_bin_dataset(bin_dir, dataset_name, input_size=64):
    """Load a face verification dataset from .bin file.

    Args:
        bin_dir: Directory containing .bin files
        dataset_name: Name of dataset (lfw, cfp_fp, etc.)
        input_size: Input image size

    Returns:
        imgs1, imgs2, labels: Numpy arrays of image pairs and labels
    """
    from torchvision import transforms
    from PIL import Image
    import io

    bin_path = os.path.join(bin_dir, f"{dataset_name}.bin")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Dataset not found: {bin_path}")

    # Load bin file
    bins, issame = pickle.load(open(bin_path, 'rb'), encoding='bytes')

    # Transform
    trans = [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    if input_size != 112:
        trans.append(transforms.Resize(input_size, antialias=True))
    transform = transforms.Compose(trans)

    # Decode all images
    imgs1, imgs2, labels = [], [], []
    for idx in tqdm(range(len(issame)), desc=f"Loading {dataset_name}", leave=False):
        # Decode using PIL instead of mxnet
        img1 = Image.open(io.BytesIO(bins[2 * idx]))
        img1 = transform(img1).unsqueeze(0)
        img2 = Image.open(io.BytesIO(bins[2 * idx + 1]))
        img2 = transform(img2).unsqueeze(0)

        imgs1.append(img1)
        imgs2.append(img2)
        labels.append(issame[idx])

    imgs1 = torch.cat(imgs1, dim=0)
    imgs2 = torch.cat(imgs2, dim=0)
    labels = np.array(labels, dtype=np.int32)

    return imgs1, imgs2, labels


def compute_embeddings(model, images, batch_size=32, device='cuda'):
    """Compute embeddings for a batch of images.

    Args:
        model: CryptoFace model
        images: Tensor of images [N, C, H, W]
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        embeddings: Numpy array [N, embedding_dim]
    """
    model.eval()
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            emb = model(batch)  # [B, embedding_dim]
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)


def fit_l2_polynomial(sum_of_squares):
    """Fit polynomial approximation a*x² + b*x + c ≈ 1/√x.

    Uses the same method as CryptoFace/ckks.py (lines 113-128):
    - Compute mean, std, min, max of sum_of_squares
    - Use 3 points: x1=mean, x2=mean+std, x3=mean-std
    - Solve for (a, b, c) such that polynomial passes through (xi, 1/√xi)

    Args:
        sum_of_squares: Array of sum of squares values

    Returns:
        a, b, c: Polynomial coefficients
        x3, x1, x2: The three fitting points used
    """
    # Remove non-finite values and keep 99% of data
    val = sum_of_squares[np.isfinite(sum_of_squares)]
    val = np.sort(val)[0:int(len(val)*0.99)]

    # Compute statistics
    val_mean = np.mean(val)
    val_std = np.std(val)
    val_min = np.min(val)
    val_max = np.max(val)

    # Three fitting points
    x1 = val_mean
    x2 = min(val_mean + val_std, val_max)
    x3 = max(val_mean - val_std, val_min)

    # Target values: 1/√x
    y1, y2, y3 = 1/np.sqrt(x1), 1/np.sqrt(x2), 1/np.sqrt(x3)

    # Solve linear system: [x1², x1, 1] [a]   [y1]
    #                      [x2², x2, 1] [b] = [y2]
    #                      [x3², x3, 1] [c]   [y3]
    A = np.array([[x1**2, x1, 1],
                  [x2**2, x2, 1],
                  [x3**2, x3, 1]])
    b = np.array([y1, y2, y3])
    a, b_coef, c = np.linalg.solve(A, b)

    return a, b_coef, c, x3, x1, x2


def estimate_l2_coefficients(
    model,
    data_dir,
    input_size=64,
    batch_size=32,
    device='cuda',
    verbose=True
):
    """Estimate L2 normalization coefficients from validation datasets.

    Args:
        model: CryptoFace model (already loaded with weights and moved to device)
        data_dir: Directory containing .bin files for validation datasets
        input_size: Input image size (64, 96, or 128)
        batch_size: Batch size for inference
        device: Device to run on
        verbose: Print progress

    Returns:
        coefficients: Dict mapping dataset_name -> [x3, x1, x2, a, b, c, threshold]
    """
    # Model should already be on the correct device
    model.eval()

    coefficients = {}

    # Process each validation dataset
    for dataset_id, dataset_name in FACE_DATASETS.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f"Processing {dataset_name.upper()}")
            print(f"{'='*70}")

        # Load dataset
        imgs1, imgs2, labels = load_bin_dataset(data_dir, dataset_name, input_size)

        # Compute embeddings
        if verbose:
            print(f"Computing embeddings for {len(imgs1)} pairs...")
        emb1 = compute_embeddings(model, imgs1, batch_size, device)
        emb2 = compute_embeddings(model, imgs2, batch_size, device)

        # Convert to float64 for numerical precision
        emb1 = emb1.astype(np.float64)
        emb2 = emb2.astype(np.float64)

        # Compute sum of squares for each embedding
        emb1_sum = np.sum(emb1 ** 2, axis=1)
        emb2_sum = np.sum(emb2 ** 2, axis=1)

        if verbose:
            print(f"Embedding shape: {emb1.shape}")
            print(f"Sum of squares range: [{emb1_sum.min():.4f}, {emb1_sum.max():.4f}]")

        # K-fold cross validation (10 folds)
        nrof_pairs = emb1.shape[0]

        fold_coefficients = []
        for fold_idx, (train_set, test_set) in enumerate(k_fold_split(nrof_pairs, n_splits=10)):
            # Use training set to fit polynomial
            train_sum = np.concatenate((emb1_sum[train_set], emb2_sum[train_set]))

            # Fit polynomial coefficients
            a, b, c, x3, x1, x2 = fit_l2_polynomial(train_sum)

            if verbose and fold_idx == 0:
                print(f"\nFold {fold_idx}: Polynomial coefficients:")
                print(f"  x3 (mean-std): {x3:.6e}")
                print(f"  x1 (mean):     {x1:.6e}")
                print(f"  x2 (mean+std): {x2:.6e}")
                print(f"  a: {a:.6e}")
                print(f"  b: {b:.6e}")
                print(f"  c: {c:.6e}")

            # TODO: Compute threshold for verification (requires similarity metric)
            # For now, use placeholder
            threshold = 0.0

            fold_coefficients.append([x3, x1, x2, a, b, c, threshold])

        # Average across folds
        fold_coefficients = np.array(fold_coefficients)
        avg_coefficients = np.mean(fold_coefficients, axis=0)

        coefficients[dataset_name] = avg_coefficients

        if verbose:
            print(f"\nAverage coefficients for {dataset_name}:")
            print(f"  x3: {avg_coefficients[0]:.6e}")
            print(f"  x1: {avg_coefficients[1]:.6e}")
            print(f"  x2: {avg_coefficients[2]:.6e}")
            print(f"  a:  {avg_coefficients[3]:.6e}")
            print(f"  b:  {avg_coefficients[4]:.6e}")
            print(f"  c:  {avg_coefficients[5]:.6e}")

    return coefficients


def save_coefficients(coefficients, output_dir, prefix="cryptoface"):
    """Save coefficients to text files.

    Args:
        coefficients: Dict mapping dataset_name -> [x3, x1, x2, a, b, c, threshold]
        output_dir: Directory to save files
        prefix: Prefix for output files
    """
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name, coefs in coefficients.items():
        # Save all 10 folds with same coefficients (for compatibility with SEAL)
        output_path = os.path.join(output_dir, f"threshold_{dataset_name}.txt")
        coefs_array = np.tile(coefs, (10, 1))  # Repeat for 10 folds
        np.savetxt(output_path, coefs_array, fmt="%.6e", delimiter=",")
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate L2 normalization polynomial coefficients for CryptoFace"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        choices=[64, 96, 128],
        default=64,
        help="Input image size (64, 96, or 128)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/vishnu/datastore/processed/faces_emore",
        help="Directory containing validation .bin files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save coefficient files (default: same as checkpoint dir)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )

    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)

    print(f"\n{'='*70}")
    print(f"Estimating L2 Normalization Coefficients")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input size: {args.input_size}")
    print(f"Data dir:   {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device:     {args.device}")

    # Create model
    if args.input_size == 64:
        model = CryptoFaceNet4()
    elif args.input_size == 96:
        model = CryptoFaceNet9()
    elif args.input_size == 128:
        model = CryptoFaceNet16()
    else:
        raise ValueError(f"Unsupported input size: {args.input_size}")

    # Load checkpoint first (this calls init_orion_params internally)
    print(f"\nLoading checkpoint...")
    load_checkpoint_for_config(model, input_size=args.input_size, verbose=True)

    # THEN move entire model to device
    print(f"\nMoving model to {args.device}...")
    model = model.to(args.device)

    # Explicitly move HerPN weight/bias attributes (they're not Parameters, so .to() doesn't move them)
    for i in range(len(model.nets)):
        backbone = model.nets[i]
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'herpnpool']:
            layer = getattr(backbone, layer_name)
            # Move herpn1_weight, herpn1_bias, herpn2_weight, herpn2_bias, herpn_weight, herpn_bias
            for attr_name in ['herpn1_weight', 'herpn1_bias', 'herpn2_weight', 'herpn2_bias', 'herpn_weight', 'herpn_bias']:
                if hasattr(layer, attr_name):
                    attr = getattr(layer, attr_name)
                    setattr(layer, attr_name, attr.to(args.device))

    model.eval()

    # Estimate coefficients
    coefficients = estimate_l2_coefficients(
        model=model,
        data_dir=args.data_dir,
        input_size=args.input_size,
        batch_size=args.batch_size,
        device=args.device,
        verbose=True
    )

    # Save coefficients
    print(f"\n{'='*70}")
    print(f"Saving Coefficients")
    print(f"{'='*70}")
    save_coefficients(coefficients, args.output_dir, prefix="cryptoface")

    print(f"\n{'='*70}")
    print(f"✓ L2 Normalization Coefficients Estimated Successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
