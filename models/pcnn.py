"""
Patch CNN (PCNN) implementation for Orion FHE framework.

This implements a privacy-preserving neural network for encrypted inference
using HerPN activations and ChannelSquare operations.
"""

import math
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import orion.nn as on
from orion.nn.module import Module, timer

__all__ = ["PatchCNN", "Backbone", "ChannelSquare", "HerPN", "HerPNConv", "HerPNPool"]


class ChannelSquare(Module):
    def __init__(self, weight0, weight1, weight2=None):
        super().__init__()
        self.weight0_raw = weight0
        self.weight1_raw = weight1
        self.weight2_raw = weight2
        self.w0 = weight0  # For plaintext
        self.w1 = weight1
        self.w2 = weight2
        self.set_depth()

    def compile(self):
        # self.level represents the INPUT level (confirmed by LevelDAG and bootstrap placement)
        # Output level = self.level - self.depth
        input_level = self.level
        output_level = self.level - self.depth

        # Use the actual shape recorded during forward pass, fallback to fhe_input_shape
        target_shape = getattr(self, '_actual_fhe_input_shape', self.fhe_input_shape)
        if target_shape is None:
            raise ValueError(f"Cannot compile {self.__class__.__name__}: no input shape recorded")

        target_shape = list(target_shape)

        # Helper to convert scalar or tensor to target shape
        def to_tensor(w):
            if isinstance(w, (int, float)):
                return torch.full(target_shape, w, dtype=torch.float32)
            else:
                return w.expand(target_shape)

        if self.weight2_raw is not None:
            # Quadratic: w2*x² + w1*x + w0
            # x@L → x²@(L-1) → w2*x²@(L-2), w1*x@(L-1) → rescales to L-2 → add w0@(L-2)
            w2 = to_tensor(self.weight2_raw)
            w1 = to_tensor(self.weight1_raw)
            w0 = to_tensor(self.weight0_raw)

            self.w2_fhe = self.scheme.encoder.encode(w2, input_level - 1)
            self.w1_fhe = self.scheme.encoder.encode(w1, input_level)
            self.w0_fhe = self.scheme.encoder.encode(w0, output_level)
        else:
            # Linear: w1*x + w0 (depth=1)
            # x@L → x²@(L-1), w1*x@(L-1) → add w0@(L-1)
            w1 = to_tensor(self.weight1_raw)
            w0 = to_tensor(self.weight0_raw)

            self.w1_fhe = self.scheme.encoder.encode(w1, input_level)
            self.w0_fhe = self.scheme.encoder.encode(w0, output_level)

    def extra_repr(self):
        return (
            super().extra_repr()
            + f"w0={self.w0 is not None}, w1={self.w1 is not None}, w2={self.w2 is not None}"
        )

    def set_depth(self):
        if self.weight2_raw is not None:
            self.depth = 2
        else:
            self.depth = 1

    @timer
    def forward(self, x):
        # Store the actual input shape for use during compilation
        # This ensures we expand weights to the correct shape later
        if not self.he_mode and not hasattr(self, '_actual_fhe_input_shape'):
            self._actual_fhe_input_shape = x.shape
            
        if self.he_mode:
            if self.weight2_raw is not None:
                x_sq = x * x
                term2 = x_sq * self.w2_fhe
                term1 = x * self.w1_fhe
                # term1 and term2 are at different levels, but addition should handle this
                result = term1 + term2
                result += self.w0_fhe
                return result
            else:
                # Only linear term (no quadratic): x^2 + w1*x + w0
                x_sq = x * x
                term1 = x * self.w1_fhe
                result = x_sq + term1
                result += self.w0_fhe
                return result
        else:
            # Plaintext implementation
            if self.weight2_raw is not None:
                return (
                    self.weight2_raw * (x**2) + self.weight1_raw * x + self.weight0_raw
                )
            else:
                return x**2 + self.weight1_raw * x + self.weight0_raw


class HerPN(ChannelSquare):
    def __init__(
        self,
        bn0_mean,
        bn0_var,
        bn1_mean,
        bn1_var,
        bn2_mean,
        bn2_var,
        weight,
        bias,
        eps=1e-5,
    ):
        # These are the parameters from the trained HerPN layer from CryptoFace
        m0, v0 = bn0_mean, bn0_var
        m1, v1 = bn1_mean, bn1_var
        m2, v2 = bn2_mean, bn2_var
        g, b = weight.squeeze(), bias.squeeze()
        e = eps

        # Calculations from HerPN_Fuse
        w2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
        w1 = torch.divide(g, 2 * torch.sqrt(v1 + e))
        w0 = b + g * (
            torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + e)))
            - torch.divide(m1, 2 * torch.sqrt(v1 + e))
            - torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + e)))
        )

        # Unsqueeze to match channel dimension for broadcasting
        w0 = w0.unsqueeze(-1).unsqueeze(-1)
        w1 = w1.unsqueeze(-1).unsqueeze(-1)
        w2 = w2.unsqueeze(-1).unsqueeze(-1)

        super().__init__(weight0=w0, weight1=w1, weight2=w2)


class HerPNConv(on.Module):
    """
    HerPN-based convolutional block with residual connection.

    Structure: HerPN -> Conv -> HerPN -> Conv -> (+shortcut)
    """

    def __init__(self, in_planes, planes, stride=1):
        super(HerPNConv, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        # First HerPN activation (will be fused from trained BatchNorms)
        self.bn0_1 = on.BatchNorm2d(in_planes)
        self.bn1_1 = on.BatchNorm2d(in_planes)
        self.bn2_1 = on.BatchNorm2d(in_planes)

        # First convolution
        self.conv1 = on.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1
        )

        # Second HerPN activation (will be fused from trained BatchNorms)
        self.bn0_2 = on.BatchNorm2d(planes)
        self.bn1_2 = on.BatchNorm2d(planes)
        self.bn2_2 = on.BatchNorm2d(planes)

        # Second convolution
        self.conv2 = on.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)

        # Shortcut connection
        self.has_shortcut = stride != 1 or in_planes != planes
        if self.has_shortcut:
            self.shortcut_conv = on.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride
            )
            self.shortcut_bn = on.BatchNorm2d(planes)

        # HerPN activations (compiled from BatchNorms)
        self.herpn1 = None
        self.herpn2 = None
        self.shortcut_herpn = None  # For scaling shortcut in fused mode

    def init_orion_params(self):
        """Initialize HerPN parameters from trained BatchNorm statistics.
        
        CRITICAL: This method MUST be called BEFORE orion.fit() for correct level assignment.
        
        Why: This method fuses BatchNorm layers into HerPN operations. If orion.fit() is called
        before fusion, the tracer captures the unfused graph with separate BatchNorm operations,
        leading to incorrect depth calculations (~11 extra levels).
        
        Correct workflow:
            1. Collect BatchNorm statistics: model.eval(); for _ in range(20): model(data)
            2. Call this method: model.init_orion_params()  
            3. Trace fused graph: orion.fit(model, inp)
            4. Compile: orion.compile(model)
            
        See docs/CUSTOM_FIT_WORKFLOW.md for detailed explanation.
        """
        # First HerPN
        bn0_mean_1 = self.bn0_1.running_mean
        bn0_var_1 = self.bn0_1.running_var
        bn1_mean_1 = self.bn1_1.running_mean
        bn1_var_1 = self.bn1_1.running_var
        bn2_mean_1 = self.bn2_1.running_mean
        bn2_var_1 = self.bn2_1.running_var

        # Create weights for first HerPN (ones and zeros as per HerPN design)
        weight_1 = torch.ones(self.in_planes, 1, 1)
        bias_1 = torch.zeros(self.in_planes, 1, 1)

        # from orion.nn.activation import HerPN

        self.herpn1 = HerPN(
            bn0_mean_1,
            bn0_var_1,
            bn1_mean_1,
            bn1_var_1,
            bn2_mean_1,
            bn2_var_1,
            weight_1,
            bias_1,
            eps=self.bn0_1.eps,
        )

        # Copy input/output statistics from bn1_1 which has the same input as herpn1
        if hasattr(self.bn1_1, "input_min"):
            self.herpn1.input_min = self.bn1_1.input_min
            self.herpn1.input_max = self.bn1_1.input_max
            self.herpn1.output_min = self.bn1_1.output_min
            self.herpn1.output_max = self.bn1_1.output_max
            self.herpn1.fhe_input_shape = self.bn1_1.fhe_input_shape
            self.herpn1.fhe_output_shape = self.bn1_1.fhe_output_shape
        
        # Copy level if it exists
        if hasattr(self.bn1_1, "level"):
            self.herpn1.level = self.bn1_1.level

        # Second HerPN
        bn0_mean_2 = self.bn0_2.running_mean
        bn0_var_2 = self.bn0_2.running_var
        bn1_mean_2 = self.bn1_2.running_mean
        bn1_var_2 = self.bn1_2.running_var
        bn2_mean_2 = self.bn2_2.running_mean
        bn2_var_2 = self.bn2_2.running_var

        weight_2 = torch.ones(self.planes, 1, 1)
        bias_2 = torch.zeros(self.planes, 1, 1)

        self.herpn2 = HerPN(
            bn0_mean_2,
            bn0_var_2,
            bn1_mean_2,
            bn1_var_2,
            bn2_mean_2,
            bn2_var_2,
            weight_2,
            bias_2,
            eps=self.bn0_2.eps,
        )

        # Copy input/output statistics from bn1_2
        if hasattr(self.bn1_2, "input_min"):
            self.herpn2.input_min = self.bn1_2.input_min
            self.herpn2.input_max = self.bn1_2.input_max
            self.herpn2.output_min = self.bn1_2.output_min
            self.herpn2.output_max = self.bn1_2.output_max
            self.herpn2.fhe_input_shape = self.bn1_2.fhe_input_shape
            self.herpn2.fhe_output_shape = self.bn1_2.fhe_output_shape
        
        # Copy level if it exists
        if hasattr(self.bn1_2, "level"):
            self.herpn2.level = self.bn1_2.level

        # For shortcut, we need to extract the a2 scaling factor from herpn1
        # This is the coefficient for x^2 term, which scales the linear shortcut
        if self.has_shortcut:
            # Extract w2 coefficient (scales the x^2 term)
            m0, v0 = bn0_mean_1, bn0_var_1
            m1, v1 = bn1_mean_1, bn1_var_1
            m2, v2 = bn2_mean_1, bn2_var_1
            g = weight_1.squeeze()
            e = self.bn0_1.eps

            a2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
            a2 = a2.unsqueeze(-1).unsqueeze(-1)

            # Create a scaling module for shortcut
            # In fused mode, shortcut input is scaled by a2
            # ChannelSquare with w0=0, w1=a2, w2=0 gives: a2 * x
            zeros = torch.zeros_like(a2)
            self.shortcut_herpn = ChannelSquare(
                weight0=zeros,
                weight1=a2,
                weight2=None,  # Only linear term
            )

            # Copy statistics for shortcut_herpn
            if hasattr(self.bn1_1, "input_min"):
                self.shortcut_herpn.input_min = self.bn1_1.input_min
                self.shortcut_herpn.input_max = self.bn1_1.input_max
                self.shortcut_herpn.output_min = self.bn1_1.output_min
                self.shortcut_herpn.output_max = self.bn1_1.output_max
                self.shortcut_herpn.fhe_input_shape = self.bn1_1.fhe_input_shape
                self.shortcut_herpn.fhe_output_shape = self.bn1_1.fhe_output_shape
            
            # Copy level if it exists
            if hasattr(self.bn1_1, "level"):
                self.shortcut_herpn.level = self.bn1_1.level

    def compile(self):
        """Compile the dynamically created HerPN modules if they exist."""
        if self.herpn1 is not None and hasattr(self.herpn1, 'compile'):
            # Need to set level and other attributes for HerPN modules
            # They should have been copied from BatchNorms during init_orion_params
            if hasattr(self.herpn1, 'level') and self.herpn1.level is not None:
                self.herpn1.compile()
        
        if self.herpn2 is not None and hasattr(self.herpn2, 'compile'):
            if hasattr(self.herpn2, 'level') and self.herpn2.level is not None:
                self.herpn2.compile()
        
        if self.has_shortcut and self.shortcut_herpn is not None and hasattr(self.shortcut_herpn, 'compile'):
            if hasattr(self.shortcut_herpn, 'level') and self.shortcut_herpn.level is not None:
                self.shortcut_herpn.compile()

    def forward(self, x):
        if not self.he_mode:
            # Cleartext forward using fused HerPN to avoid graph branching
            # Check if HerPN modules are initialized (after init_orion_params)
            if self.herpn1 is not None:
                # Use fused HerPN (avoids branching in network DAG)
                out = self.herpn1(x)
                out = self.conv1(out)
                out = self.herpn2(out)
                out = self.conv2(out)

                # Shortcut
                if self.has_shortcut:
                    shortcut = self.shortcut_herpn(x)  # Scale by a2
                    shortcut = self.shortcut_conv(shortcut)
                    shortcut = self.shortcut_bn(shortcut)
                else:
                    shortcut = x

                out = out + shortcut
                return out
            else:
                # During training, use unfused BatchNorms
                # First HerPN
                x0 = self.bn0_1(torch.ones_like(x))
                x1 = self.bn1_1(x)
                x2 = self.bn2_1((torch.square(x) - 1) / math.sqrt(2))
                out = (
                    torch.divide(x0, math.sqrt(2 * math.pi))
                    + torch.divide(x1, 2)
                    + torch.divide(x2, math.sqrt(4 * math.pi))
                )

                # First conv
                out = self.conv1(out)

                # Second HerPN
                out0 = self.bn0_2(torch.ones_like(out))
                out1 = self.bn1_2(out)
                out2 = self.bn2_2((torch.square(out) - 1) / math.sqrt(2))
                out = (
                    torch.divide(out0, math.sqrt(2 * math.pi))
                    + torch.divide(out1, 2)
                    + torch.divide(out2, math.sqrt(4 * math.pi))
                )

                # Second conv
                out = self.conv2(out)

                # Shortcut
                if self.has_shortcut:
                    shortcut = self.shortcut_conv(x)
                    shortcut = self.shortcut_bn(shortcut)
                else:
                    shortcut = x

                out = out + shortcut
                return out
        else:
            # FHE mode using compiled HerPN modules
            identity = x  # Save original input for shortcut
            out = self.herpn1(x)
            out = self.conv1(out)
            out = self.herpn2(out)
            out = self.conv2(out)

            # Shortcut
            if self.has_shortcut:
                # Apply scaling and then shortcut convolution
                shortcut = self.shortcut_herpn(identity)  # Scale by a2
                shortcut = self.shortcut_conv(shortcut)
                shortcut = self.shortcut_bn(shortcut)
            else:
                shortcut = identity

            out = out + shortcut
            return out


class HerPNPool(on.Module):
    """
    HerPN activation followed by adaptive average pooling.

    Uses AdaptiveAvgPool2d for proper FHE support:
    - Automatically computes kernel size from input/output dimensions
    - Correctly handles FHE gaps and shapes during compilation
    - Works with any input/output size combination
    """

    def __init__(self, planes, output_size, input_size=None):
        super(HerPNPool, self).__init__()
        self.planes = planes
        self.output_size = output_size
        self.input_size = input_size  # Kept for backward compatibility (not used)

        # BatchNorms for HerPN (will be fused)
        self.bn0 = on.BatchNorm2d(planes)
        self.bn1 = on.BatchNorm2d(planes)
        self.bn2 = on.BatchNorm2d(planes)

        # Use AdaptiveAvgPool2d for proper FHE support
        # This handles kernel size computation internally and correctly manages FHE gaps
        self.pool = on.AdaptiveAvgPool2d(output_size)

        # HerPN activation (compiled from BatchNorms)
        self.herpn = None

    def init_orion_params(self):
        """Initialize HerPN parameters from trained BatchNorm statistics."""
        bn0_mean = self.bn0.running_mean
        bn0_var = self.bn0.running_var
        bn1_mean = self.bn1.running_mean
        bn1_var = self.bn1.running_var
        bn2_mean = self.bn2.running_mean
        bn2_var = self.bn2.running_var

        weight = torch.ones(self.planes, 1, 1)
        bias = torch.zeros(self.planes, 1, 1)

        self.herpn = HerPN(
            bn0_mean,
            bn0_var,
            bn1_mean,
            bn1_var,
            bn2_mean,
            bn2_var,
            weight,
            bias,
            eps=self.bn0.eps,
        )

        # Copy input/output statistics from bn1
        if hasattr(self.bn1, "input_min"):
            self.herpn.input_min = self.bn1.input_min
            self.herpn.input_max = self.bn1.input_max
            self.herpn.output_min = self.bn1.output_min
            self.herpn.output_max = self.bn1.output_max
            self.herpn.fhe_input_shape = self.bn1.fhe_input_shape
            self.herpn.fhe_output_shape = self.bn1.fhe_output_shape

    def forward(self, x):
        if not self.he_mode:
            # Cleartext forward using fused HerPN to avoid graph branching
            if self.herpn is not None:
                # Use fused HerPN (avoids branching in network DAG)
                out = self.herpn(x)
                # Apply adaptive pooling
                out = self.pool(out)
                return out
            else:
                # During training, use unfused BatchNorms
                x0 = self.bn0(torch.ones_like(x))
                x1 = self.bn1(x)
                x2 = self.bn2((torch.square(x) - 1) / math.sqrt(2))
                out = (
                    torch.divide(x0, math.sqrt(2 * math.pi))
                    + torch.divide(x1, 2)
                    + torch.divide(x2, math.sqrt(4 * math.pi))
                )

                # Apply adaptive pooling
                out = self.pool(out)
                return out
        else:
            # FHE mode
            out = self.herpn(x)
            out = self.pool(out)
            return out


class Backbone(on.Module):
    """
    Feature extraction backbone for PCNN.
    """

    def __init__(self, output_size, input_size=16):
        super(Backbone, self).__init__()
        self.output_size = output_size
        self.input_size = input_size

        # Initial convolution
        self.conv = on.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        # HerPN convolutional blocks
        self.layer1 = HerPNConv(16, 16)
        self.layer2 = HerPNConv(16, 32, stride=2)  # /2
        self.layer3 = HerPNConv(32, 32)
        self.layer4 = HerPNConv(32, 64, stride=2)  # /2
        self.layer5 = HerPNConv(64, 64)

        # Calculate size before pooling (after two stride=2 layers)
        pool_input_size = input_size // 4

        # HerPN pooling
        self.herpnpool = HerPNPool(
            64, output_size=output_size, input_size=pool_input_size
        )

        # Flatten
        self.flatten = on.Flatten()

        # Final batch norm
        self.bn = on.BatchNorm1d(output_size[0] * output_size[1] * 64)

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.herpnpool(out)
        out = self.flatten(out)
        out = self.bn(out)
        return out

    def init_orion_params(self):
        """Initialize HerPN parameters from trained BatchNorm statistics."""
        self.layer1.init_orion_params()
        self.layer2.init_orion_params()
        self.layer3.init_orion_params()
        self.layer4.init_orion_params()
        self.layer5.init_orion_params()
        self.herpnpool.init_orion_params()


class PatchCNN(on.Module):
    """
    Patch-based CNN for privacy-preserving inference.

    This model processes image patches independently and aggregates features
    for classification. Designed for FHE-friendly operations.
    """

    def __init__(self, input_size, patch_size, sqrt_weights, output_size=(4,4)):
        super(PatchCNN, self).__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.H = input_size // patch_size
        self.W = input_size // patch_size
        self.N = self.H * self.W
        
        self.dim = output_size[0] * output_size[1] * 64

        # Create backbone networks for each patch
        self.nets = nn.ModuleList(
            [Backbone(output_size, input_size=patch_size) for _ in range(self.N)]
        )

        self.linear = nn.ModuleList(
            [on.Linear(self.dim, 1024) for _ in range(self.N)]
        )

        self.normalization = ChannelSquare(sqrt_weights[0], sqrt_weights[1])

    def forward(self, x):
        """
        Forward pass for PCNN.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            out: Classification logits (B, num_classes)
            For training mode (not FHE), also returns jigsaw prediction and target
        """
        B = x.shape[0]
        H, W = self.H, self.W
        N = self.N
        P = self.patch_size

        # Rearrange into patches: (B, C, H_img, W_img) -> (N, B, C, P, P)
        # Split into patches manually
        patches = []
        for h in range(H):
            for w in range(W):
                patch = x[:, :, h * P : (h + 1) * P, w * P : (w + 1) * P]
                patches.append(patch)

        # Process each patch through its backbone
        if not self.he_mode:
            # Cleartext mode: sequential processing is fine
            y = 0
            for i in range(N):
                y_i = self.nets[i](patches[i])
                y_i = self.linear[i](y_i)
                y = y + y_i  # Aggregate features
        else:
            # FHE mode: parallel processing of independent patches
            # Each patch goes through a different backbone network
            def process_patch(i):
                """Process a single patch through its backbone and linear layer."""
                y_i = self.nets[i](patches[i])
                y_i = self.linear[i](y_i)
                return y_i

            # Execute all N backbones in parallel
            # ThreadPoolExecutor works well because FHE operations in Lattigo (Go)
            # release the GIL, allowing true parallelism
            with ThreadPoolExecutor(max_workers=N) as executor:
                y_outputs = list(executor.map(process_patch, range(N)))

            # Use binary tree reduction for efficient parallel addition
            # Reduces sequential depth from N to log2(N)
            y = self._tree_reduce_add(y_outputs)

        out = self.normalization(y)

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
        """Initialize HerPN parameters for all backbone networks."""
        for net in self.nets:
            net.init_orion_params()