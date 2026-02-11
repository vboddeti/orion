from abc import abstractmethod
import math

import torch
import torch.nn as nn

from .module import Module, timer
from ..core import packing

class BatchNormNd(Module):    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.fused = False
        self.set_depth(2 if affine else 1)

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    @abstractmethod
    def _check_input_dim(self, x):
        raise NotImplementedError("Subclasses must implement _check_input_dim")
        
    def init_orion_params(self):
        self.on_running_mean = self.running_mean.data.clone()
        self.on_running_var = self.running_var.data.clone()
        
        if self.affine:
            self.on_weight = self.weight.data.clone()
            self.on_bias = self.bias.data.clone()
        else:
            self.on_weight = torch.ones_like(self.on_running_mean)
            self.on_bias = torch.zeros_like(self.on_running_mean)

    def extra_repr(self):
        return super().extra_repr() + f", level={self.level}, fused={self.fused}"
                
    def compile(self, a, b, c, d):
        level = self.level
        encoder = self.scheme.encoder

        q1 = encoder.get_moduli_chain()[level]
        q2 = encoder.get_moduli_chain()[level - 1]

        # In order to ensure errorless neural network evaluation, we'll 
        # need to encode the scaling/shifting and affine maps at the 
        # correct scale value.
        self.on_running_mean_ptxt = encoder.encode(a, level=level, scale=q1)
        self.on_inv_running_std_ptxt = encoder.encode(b, level=level, scale=q1)

        if self.affine:
            self.on_weight_ptxt = encoder.encode(c, level=level-1, scale=q2)
            self.on_bias_ptxt = encoder.encode(d, level=level-1, scale=q2)

    @timer
    def forward(self, x):
        if not self.he_mode:
            self._check_input_dim(x)
                
        if self.training:
            exponential_average_factor = 0.0
            if self.momentum is not None:
                exponential_average_factor = self.momentum
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None: # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked
        else:
            exponential_average_factor = 0.0

        if not self.he_mode:            
            return torch.nn.functional.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                self.training,
                exponential_average_factor,
                self.eps
            )

        # In HE evaluation mode.
        if not self.fused:
            x -= self.on_running_mean_ptxt 
            x *= self.on_inv_running_std_ptxt

            if self.affine:
                x *= self.on_weight_ptxt
                x += self.on_bias_ptxt

        return x


class BatchNorm1d(BatchNormNd):    
    def _check_input_dim(self, x):
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f'expected 2D or 3D input (got {x.dim()}D input)')
            
    def compile(self):
        a, b, c, d = packing.pack_bn1d(self) 
        super().compile(a, b, c, d)


class BatchNorm2d(BatchNormNd):
    def _check_input_dim(self, x):
        if x.dim() != 4:
            raise ValueError(f'expected 4D input (got {x.dim()}D input)')
                
    def compile(self):
        a, b, c, d = packing.pack_bn2d(self) 
        super().compile(a, b, c, d)


class L2NormPoly(Module):
    """
    L2 Normalization using polynomial approximation of 1/sqrt(x).

    Implements the CryptoFace L2 normalization:
        1. Compute sum of squares: y = sum(x²) across feature dimension
        2. Apply polynomial approximation: norm_inv = a*y² + b*y + c ≈ 1/√y
        3. Normalize: x_norm = x * norm_inv

    This matches the SEAL implementation in CryptoFace's l2norm_seal function.

    Args:
        a, b, c: Polynomial coefficients for approximating 1/√y
        num_features: Number of features (embedding dimension)

    Level consumption:
        - x² (per feature): 1 level
        - Sum reduction: 0 levels (additions only)
        - y² (sum squared): 1 level
        - Polynomial computation (a*y² + b*y + c): 0 levels (after rescale)
        - Final multiplication (x * norm_inv): 1 level
        Total depth: 3 levels
    """

    def __init__(self, a, b, c, num_features):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.num_features = num_features
        self.depth = 3  # x², y², x*norm_inv

        # Add BatchNorm compatibility attributes for weight loader
        # These are not used by L2NormPoly, but needed for checkpoint loading
        self.eps = 1e-5
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def compile(self):
        """Prepare FHE-encoded constants for manual polynomial evaluation."""
        # Store coefficient values for runtime encoding
        self.a_value = self.a
        self.b_value = self.b
        self.c_value = self.c

    @timer
    def forward(self, x):
        """Apply L2 normalization with polynomial approximation.

        Args:
            x: Input tensor of shape (B, num_features)

        Returns:
            Normalized tensor of shape (B, num_features)
        """
        if self.he_mode:
            # Step 1: Compute sum of squares: y = sum(x²)
            x_sq = x * x
            
            y = self._sum_reduction_fhe(x_sq)

            # Step 2: Apply polynomial manually: norm_inv = a*y² + b*y + c
            y_sq = y * y

            # Use scalar multiplication instead of plaintext encoding
            # This uses evaluator.mul_scalar which handles small values better
            term_a = y_sq * self.a_value

            term_b = y * self.b_value

            # Align levels before addition: drop term_b to match term_a's level
            if term_b.level() > term_a.level():
                term_b = term_b.mod_switch_to(term_a)

            poly = term_a + term_b

            norm_inv = poly + self.c_value

            # Step 3: Normalize: x * norm_inv
            # Need to align x to match norm_inv's level before multiplication
            if x.level() > norm_inv.level():
                x = x.mod_switch_to(norm_inv)

            x_norm = x * norm_inv
            return x_norm
        else:
            # Cleartext: standard polynomial approximation
            y = torch.sum(x ** 2, dim=1, keepdim=True)
            norm_inv = self.a * y ** 2 + self.b * y + self.c
            x_norm = x * norm_inv
            return x_norm

    def _sum_reduction_fhe(self, x):
        """
        Sum reduction across feature dimension using tree reduction.

        For encrypted tensors, we can't use torch.sum directly.
        Instead, we need to use rotations and additions (like SEAL's sumSlots).

        However, for 1D feature vectors, we can use a simpler approach:
        sum all features using sequential additions (depth is 0 since additions
        don't consume levels when scales match).

        Args:
            x: Encrypted tensor of shape (B, F)

        Returns:
            Sum tensor of shape (B, 1)
        """
        # Use logarithmic rotate-and-sum reduction
        # This sums all features into every slot (cyclic convolution)
        # yielding the total sum in every slot of the result.
        # This avoids unsupported slicing and is much more efficient (O(logN) vs O(N)).
        
        result = x
        steps = int(math.log2(self.num_features))
        for i in range(steps):
            shift = 1 << i
            # Sum with rotated version
            result = result + result.roll(shift)

        return result


