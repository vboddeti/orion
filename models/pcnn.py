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

__all__ = ["PatchCNN", "Backbone", "ChannelSquare", "HerPN", "HerPNConv", "HerPNPool", "ScaleModule", "L2NormPoly"]


class ScaleModule(Module):
    """Simple module that scales input by a constant factor.

    This wrapper is needed to avoid FHE tracer shape validation issues
    when directly multiplying tensors with different shapes.
    """
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_raw = scale_factor  # Shape: [C, 1, 1]
        self.depth = 1  # Ciphertext multiplication consumes 1 level (requires rescaling)

    def compile(self):
        """Expand scale factor to match FHE input shape (with packing if needed)."""
        import torch.nn.functional as F

        fhe_shape = self.fhe_input_shape
        clear_shape = getattr(self, 'input_shape', None)
        gap = getattr(self, 'input_gap', 1)

        if fhe_shape is None:
            raise ValueError(f"Cannot compile {self.__class__.__name__}: no input shape recorded")

        fhe_shape = list(fhe_shape)
        clear_shape = list(clear_shape) if clear_shape is not None else fhe_shape

        if gap > 1:
            import math
            scale_expanded = self.scale_raw.unsqueeze(0).expand(clear_shape)

            # Pad channels to be divisible by gap² for packing
            N, Ci, Hi, Wi = scale_expanded.shape
            Co = math.ceil(Ci / (gap**2))
            if Co * gap**2 != Ci:
                padded = torch.zeros(N, Co * gap**2, Hi, Wi, dtype=scale_expanded.dtype, device=scale_expanded.device)
                padded[:, :Ci, :, :] = scale_expanded
                scale_expanded = padded

            scale_packed = F.pixel_shuffle(scale_expanded, gap)
            self.scale_fhe = self.scheme.encoder.encode(scale_packed, self.level)
        else:
            scale_expanded = self.scale_raw.unsqueeze(0).expand(fhe_shape)
            self.scale_fhe = self.scheme.encoder.encode(scale_expanded, self.level)

    def forward(self, x):
        if self.he_mode:
            # FHE mode: use pre-encoded expanded scale
            return x * self.scale_fhe
        else:
            # Cleartext: direct multiplication (Python broadcasting handles [C,1,1] * [B,C,H,W])
            return x * self.scale_raw.unsqueeze(0)


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
        input_level = self.level
        output_level = self.level - self.depth

        fhe_shape = self.fhe_input_shape
        clear_shape = getattr(self, 'input_shape', None)
        gap = getattr(self, 'input_gap', 1)

        if fhe_shape is None:
            raise ValueError(f"Cannot compile {self.__class__.__name__}: no input shape recorded")

        fhe_shape = list(fhe_shape)
        clear_shape = list(clear_shape) if clear_shape is not None else fhe_shape

        def to_tensor(w):
            """Convert scalar or tensor to FHE target shape (with packing if needed)."""
            import torch.nn.functional as F
            import math

            if isinstance(w, (int, float)):
                return torch.full(fhe_shape, w, dtype=torch.float32)
            else:
                if gap > 1:
                    w_expanded = w.expand(clear_shape)

                    # Pad channels to be divisible by gap² for packing
                    N, Ci, Hi, Wi = w_expanded.shape
                    Co = math.ceil(Ci / (gap**2))
                    if Co * gap**2 != Ci:
                        padded = torch.zeros(N, Co * gap**2, Hi, Wi, dtype=w_expanded.dtype, device=w_expanded.device)
                        padded[:, :Ci, :, :] = w_expanded
                        w_expanded = padded

                    w_packed = F.pixel_shuffle(w_expanded, gap)
                    return w_packed
                else:
                    return w.expand(fhe_shape)

        if self.weight2_raw is not None:
            # Quadratic: Use factored form (x² + a1·x + a0) since conv weights
            # have been pre-scaled by w2, reducing FHE multiplications
            a1_raw = self.weight1_raw / self.weight2_raw
            a0_raw = self.weight0_raw / self.weight2_raw

            a1 = to_tensor(a1_raw)
            a0 = to_tensor(a0_raw)

            self.w1_fhe = self.scheme.encoder.encode(a1, input_level)
            # Store w0 for runtime encoding to avoid implicit rescales
            self.w0_values = a0
        else:
            # Linear: w1*x + w0
            w1 = to_tensor(self.weight1_raw)
            w0 = to_tensor(self.weight0_raw)

            self.w1_fhe = self.scheme.encoder.encode(w1, input_level)
            # Store w0 for runtime encoding to avoid implicit rescales
            self.w0_values = w0

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
        if self.he_mode:
            # Use factored form: x² + a1·x + a0
            x_sq = x * x
            term1 = x * self.w1_fhe
            result = x_sq + term1

            # Encode w0 at runtime with result's scale to avoid implicit rescales
            if hasattr(self, 'w0_values'):
                w0_fhe = self.scheme.encoder.encode(self.w0_values, result.level(), result.scale())
            elif hasattr(self, 'w0_fhe'):
                w0_fhe = self.w0_fhe
            else:
                raise AttributeError("ChannelSquare has neither w0_values nor w0_fhe")

            result += w0_fhe
            return result
        else:
            # Cleartext: x² + a1·x + a0 (matches CryptoFace forward_fuse)
            if self.weight2_raw is not None:
                return torch.square(x) + self.weight1_raw * x + self.weight0_raw
            else:
                return x**2 + self.weight1_raw * x + self.weight0_raw


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
        """Prepare FHE-encoded constants."""
        input_level = self.level
        output_level = self.level - self.depth

        fhe_shape = self.fhe_input_shape
        if fhe_shape is None:
            raise ValueError(f"Cannot compile {self.__class__.__name__}: no input shape recorded")

        fhe_shape = list(fhe_shape)

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
            # Compute sum of squares: y = sum(x²)
            x_sq = x * x
            y = self._sum_reduction_fhe(x_sq)

            # Apply polynomial: norm_inv = a*y² + b*y + c
            y_sq = y * y

            a_fhe = self.scheme.encoder.encode(
                torch.full_like(y_sq, self.a_value),
                y_sq.level(),
                y_sq.scale()
            )
            term_a = a_fhe * y_sq

            b_fhe = self.scheme.encoder.encode(
                torch.full_like(y, self.b_value),
                y.level(),
                y.scale()
            )
            term_b = b_fhe * y

            poly = term_a + term_b

            c_fhe = self.scheme.encoder.encode(
                torch.full_like(poly, self.c_value),
                poly.level(),
                poly.scale()
            )
            norm_inv = poly + c_fhe

            # Normalize: x * norm_inv
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
        # For now, use sequential sum (can be optimized with rotations later)
        # This works because we're summing along the feature dimension
        result = x[:, 0:1]  # Start with first feature
        for i in range(1, self.num_features):
            result = result + x[:, i:i+1]

        return result


class HerPN(ChannelSquare):
    """
    HerPN activation with CryptoFace-style factoring for FHE efficiency.

    Computes: x² + a1·x + a0 (factored form)
    where a1 = w1/w2, a0 = w0/w2

    The w2 scaling factor is stored separately and absorbed into subsequent
    conv weights, reducing FHE multiplications by 1 per HerPN layer.
    """
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

        # Calculate full fusion coefficients (following CryptoFace HerPN_Fuse)
        w2 = torch.divide(g, torch.sqrt(8 * math.pi * (v2 + e)))
        w1 = torch.divide(g, 2 * torch.sqrt(v1 + e))
        w0 = b + g * (
            torch.divide(1 - m0, torch.sqrt(2 * math.pi * (v0 + e)))
            - torch.divide(m1, 2 * torch.sqrt(v1 + e))
            - torch.divide(1 + math.sqrt(2) * m2, torch.sqrt(8 * math.pi * (v2 + e)))
        )

        # Store w2 as scale_factor (will be absorbed into conv weights for FHE)
        # This is the CryptoFace approach: factor out w2 to reduce FHE multiplications
        self.scale_factor = w2.unsqueeze(-1).unsqueeze(-1)

        # Store full coefficients for cleartext (numerically stable)
        w0_full = w0.unsqueeze(-1).unsqueeze(-1)
        w1_full = w1.unsqueeze(-1).unsqueeze(-1)
        w2_full = w2.unsqueeze(-1).unsqueeze(-1)

        # Factor the coefficients for FHE: a1 = w1/w2, a0 = w0/w2
        # Output becomes: x² + a1·x + a0 (instead of w2·x² + w1·x + w0)
        a1 = w1 / w2
        a0 = w0 / w2

        # Unsqueeze factored coefficients for broadcasting
        a0 = a0.unsqueeze(-1).unsqueeze(-1)
        a1 = a1.unsqueeze(-1).unsqueeze(-1)

        # ChannelSquare with weight2=w2 computes: w2·x² + w1·x + w0 (cleartext)
        # or x² + a1·x + a0 (FHE, where a1=w1/w2, a0=w0/w2)
        # We pass the full coefficients, and ChannelSquare will factor them for FHE mode
        super().__init__(weight0=w0_full, weight1=w1_full, weight2=w2_full)


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
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        # Second HerPN activation (will be fused from trained BatchNorms)
        self.bn0_2 = on.BatchNorm2d(planes)
        self.bn1_2 = on.BatchNorm2d(planes)
        self.bn2_2 = on.BatchNorm2d(planes)

        # Second convolution
        self.conv2 = on.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # Shortcut connection
        self.has_shortcut = stride != 1 or in_planes != planes
        if self.has_shortcut:
            self.shortcut_conv = on.Conv2d(
                in_planes, planes, kernel_size=1, stride=stride, bias=False
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

        NOTE: This method is idempotent - calling it multiple times produces the same result.
        This is necessary because orion.fit() also calls init_orion_params() internally.
        """
        # Skip if already fused (idempotent)
        if self.herpn1 is not None:
            return

        # Store original weights on first call to enable idempotent behavior
        if not hasattr(self, '_conv1_weight_original'):
            self._conv1_weight_original = self.conv1.weight.data.clone()
            self._conv2_weight_original = self.conv2.weight.data.clone()
            if self.has_shortcut:
                self._shortcut_conv_weight_original = self.shortcut_conv.weight.data.clone()

        # Restore original weights before scaling (makes this method idempotent)
        self.conv1.weight.data = self._conv1_weight_original.clone()
        self.conv2.weight.data = self._conv2_weight_original.clone()
        if self.has_shortcut:
            self.shortcut_conv.weight.data = self._shortcut_conv_weight_original.clone()

        bn0_mean_1 = self.bn0_1.running_mean
        bn0_var_1 = self.bn0_1.running_var
        bn1_mean_1 = self.bn1_1.running_mean
        bn1_var_1 = self.bn1_1.running_var
        bn2_mean_1 = self.bn2_1.running_mean
        bn2_var_1 = self.bn2_1.running_var

        if hasattr(self, 'herpn1_weight') and hasattr(self, 'herpn1_bias'):
            weight_1 = self.herpn1_weight
            bias_1 = self.herpn1_bias
        else:
            device = bn0_mean_1.device
            weight_1 = torch.ones(self.in_planes, 1, 1, device=device)
            bias_1 = torch.zeros(self.in_planes, 1, 1, device=device)

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

        if hasattr(self.bn1_1, "input_min"):
            self.herpn1.input_min = self.bn1_1.input_min
            self.herpn1.input_max = self.bn1_1.input_max
            self.herpn1.output_min = self.bn1_1.output_min
            self.herpn1.output_max = self.bn1_1.output_max
            self.herpn1.fhe_input_shape = self.bn1_1.fhe_input_shape
            self.herpn1.fhe_output_shape = self.bn1_1.fhe_output_shape

        if hasattr(self.bn1_1, "level"):
            self.herpn1.level = self.bn1_1.level

        bn0_mean_2 = self.bn0_2.running_mean
        bn0_var_2 = self.bn0_2.running_var
        bn1_mean_2 = self.bn1_2.running_mean
        bn1_var_2 = self.bn1_2.running_var
        bn2_mean_2 = self.bn2_2.running_mean
        bn2_var_2 = self.bn2_2.running_var

        if hasattr(self, 'herpn2_weight') and hasattr(self, 'herpn2_bias'):
            weight_2 = self.herpn2_weight
            bias_2 = self.herpn2_bias
        else:
            device = bn0_mean_2.device
            weight_2 = torch.ones(self.planes, 1, 1, device=device)
            bias_2 = torch.zeros(self.planes, 1, 1, device=device)

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

        if hasattr(self.bn1_2, "level"):
            self.herpn2.level = self.bn1_2.level

        # NOTE: CryptoFace Factoring Strategy
        # 1. Scale conv weights by w2: conv.weight *= w2
        # 2. Factor HerPN coefficients: a1 = w1/w2, a0 = w0/w2
        # 3. Forward uses factored form: x² + a1·x + a0 (reduces FHE multiplications)
        # 4. Shortcut is scaled: shortcut(herpn1_out * w2)

        w2_1 = self.herpn1.scale_factor
        self.conv1.weight.data = self.conv1.weight.data * w2_1

        self.herpn1.weight1_raw.data = self.herpn1.weight1_raw.data / w2_1
        self.herpn1.weight0_raw.data = self.herpn1.weight0_raw.data / w2_1
        self.herpn1.depth = 1  # Factored form consumes 1 level, not 2
        self.herpn1.weight2_raw = None  # Prevent double factoring in compile()

        w2_2 = self.herpn2.scale_factor
        self.conv2.weight.data = self.conv2.weight.data * w2_2

        self.herpn2.weight1_raw.data = self.herpn2.weight1_raw.data / w2_2
        self.herpn2.weight0_raw.data = self.herpn2.weight0_raw.data / w2_2
        self.herpn2.depth = 1
        self.herpn2.weight2_raw = None

        # Shortcut: scale herpn1 output by w2_1, then apply shortcut conv
        # NOTE: We do NOT scale shortcut_conv weights (CryptoFace approach)
        self.shortcut_scale = ScaleModule(w2_1)

        # Fuse shortcut BatchNorm into shortcut_conv
        if self.has_shortcut:
            bn = self.shortcut_bn
            conv = self.shortcut_conv

            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight if bn.weight is not None else torch.ones_like(running_mean)
            beta = bn.bias if bn.bias is not None else torch.zeros_like(running_mean)
            eps = bn.eps

            scale = gamma / torch.sqrt(running_var + eps)
            fused_weight = conv.weight.data * scale.view(-1, 1, 1, 1)

            if conv.bias is not None:
                fused_bias = (conv.bias.data - running_mean) * scale + beta
            else:
                fused_bias = (-running_mean) * scale + beta
                conv.bias = torch.nn.Parameter(torch.zeros_like(fused_bias))

            conv.weight.data = fused_weight
            conv.bias.data = fused_bias
            self.shortcut_bn_fused = True

        if hasattr(self.herpn1, 'input_min'):
            self.shortcut_scale.input_min = self.herpn1.input_min
            self.shortcut_scale.input_max = self.herpn1.input_max
            self.shortcut_scale.output_min = self.herpn1.output_min
            self.shortcut_scale.output_max = self.herpn1.output_max
            self.shortcut_scale.fhe_input_shape = self.herpn1.fhe_output_shape
            self.shortcut_scale.fhe_output_shape = self.herpn1.fhe_output_shape

        if hasattr(self.herpn1, 'level'):
            self.shortcut_scale.level = self.herpn1.level

    def compile(self, trace=None):
        """Compile the dynamically created HerPN modules if they exist."""
        # Copy levels from trace if HerPN modules don't have them
        if self.herpn1 is not None:
            if not hasattr(self.herpn1, 'level') or self.herpn1.level is None:
                level = None
                if trace is not None and hasattr(self, 'bn1_1'):
                    for name, mod in trace.named_modules():
                        if mod is self.bn1_1 and hasattr(mod, 'level'):
                            level = mod.level
                            break
                if level is None and hasattr(self.bn1_1, 'level'):
                    level = self.bn1_1.level

                if level is not None:
                    self.herpn1.level = level
                    self.herpn1.depth = 1

        if self.herpn2 is not None:
            if not hasattr(self.herpn2, 'level') or self.herpn2.level is None:
                level = None
                if trace is not None and hasattr(self, 'bn1_2'):
                    for name, mod in trace.named_modules():
                        if mod is self.bn1_2 and hasattr(mod, 'level'):
                            level = mod.level
                            break
                if level is None and hasattr(self.bn1_2, 'level'):
                    level = self.bn1_2.level

                if level is not None:
                    self.herpn2.level = level
                    self.herpn2.depth = 1

        if self.herpn1 is not None and hasattr(self.herpn1, 'compile'):
            if hasattr(self.herpn1, 'level') and self.herpn1.level is not None:
                self.herpn1.compile()

        if self.herpn2 is not None and hasattr(self.herpn2, 'compile'):
            if hasattr(self.herpn2, 'level') and self.herpn2.level is not None:
                self.herpn2.compile()

        if hasattr(self, 'shortcut_scale') and hasattr(self.shortcut_scale, 'compile'):
            if hasattr(self.shortcut_scale, 'level') and self.shortcut_scale.level is not None:
                self.shortcut_scale.compile()

    def forward(self, x):
        if not self.he_mode:
            if self.herpn1 is not None:
                # Fused HerPN mode (post init_orion_params)
                herpn1_out = self.herpn1(x)
                out = self.conv1(herpn1_out)
                out = self.herpn2(out)
                out = self.conv2(out)

                shortcut = herpn1_out
                shortcut = self.shortcut_scale(shortcut)
                if self.has_shortcut:
                    shortcut = self.shortcut_conv(shortcut)
                    if not hasattr(self, 'shortcut_bn_fused') or not self.shortcut_bn_fused:
                        shortcut = self.shortcut_bn(shortcut)

                # Debug logging (disabled by default)
                main_min, main_max = out.min().item(), out.max().item()
                short_min, short_max = shortcut.min().item(), shortcut.max().item()
                out = out + shortcut
                add_min, add_max = out.min().item(), out.max().item()

                return out
            else:
                # Unfused mode (during training with BatchNorms)
                x0 = self.bn0_1(torch.ones_like(x))
                x1 = self.bn1_1(x)
                x2 = self.bn2_1((torch.square(x) - 1) / math.sqrt(2))
                herpn1_out = (
                    torch.divide(x0, math.sqrt(2 * math.pi))
                    + torch.divide(x1, 2)
                    + torch.divide(x2, math.sqrt(4 * math.pi))
                )
                if hasattr(self, 'herpn1_weight') and hasattr(self, 'herpn1_bias'):
                    herpn1_out = self.herpn1_weight * herpn1_out + self.herpn1_bias

                out = self.conv1(herpn1_out)

                out0 = self.bn0_2(torch.ones_like(out))
                out1 = self.bn1_2(out)
                out2 = self.bn2_2((torch.square(out) - 1) / math.sqrt(2))
                herpn2_out = (
                    torch.divide(out0, math.sqrt(2 * math.pi))
                    + torch.divide(out1, 2)
                    + torch.divide(out2, math.sqrt(4 * math.pi))
                )
                if hasattr(self, 'herpn2_weight') and hasattr(self, 'herpn2_bias'):
                    herpn2_out = self.herpn2_weight * herpn2_out + self.herpn2_bias

                out = self.conv2(herpn2_out)

                if self.has_shortcut:
                    shortcut = self.shortcut_conv(herpn1_out)
                    shortcut = self.shortcut_bn(shortcut)
                else:
                    shortcut = herpn1_out

                out = out + shortcut
                return out
        else:
            # FHE mode
            herpn1_out = self.herpn1(x)
            out = self.conv1(herpn1_out)
            out = self.herpn2(out)
            out = self.conv2(out)

            shortcut = herpn1_out
            shortcut = self.shortcut_scale(shortcut)

            if self.has_shortcut:
                shortcut = self.shortcut_conv(shortcut)
                if not hasattr(self, 'shortcut_bn_fused') or not self.shortcut_bn_fused:
                    shortcut = self.shortcut_bn(shortcut)

            # Align levels using mod_switch_to (SEAL-style error-free alignment)
            if hasattr(out, 'level') and hasattr(shortcut, 'level'):
                out_level = out.level()
                shortcut_level = shortcut.level()

                if out_level != shortcut_level:
                    if shortcut_level > out_level:
                        shortcut.mod_switch_to(out, in_place=True)
                    elif out_level > shortcut_level:
                        out.mod_switch_to(shortcut, in_place=True)

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

        # Get weights for HerPN (loaded from checkpoint, or default to ones/zeros)
        if hasattr(self, 'herpn_weight') and hasattr(self, 'herpn_bias'):
            weight = self.herpn_weight
            bias = self.herpn_bias
        else:
            # Create on same device as BatchNorm tensors
            device = bn0_mean.device
            weight = torch.ones(self.planes, 1, 1, device=device)
            bias = torch.zeros(self.planes, 1, 1, device=device)

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

        # NOTE: CryptoFace factoring strategy for HerPNPool
        # After factoring herpn coefficients (a1 = w1/w2, a0 = w0/w2),
        # the herpn outputs factored form: x² + a1·x + a0
        # Then pool output is scaled by w2 (CryptoFace/models/layers.py line 115)
        w2 = self.herpn.scale_factor

        # Factor herpn coefficients: a1 = w1/w2, a0 = w0/w2
        self.herpn.weight1_raw.data /= w2
        self.herpn.weight0_raw.data /= w2
        # CRITICAL: Update depth to 1 for factored form
        self.herpn.depth = 1
        # CRITICAL: Set weight2_raw to None so compile() doesn't divide by w2 again
        self.herpn.weight2_raw = None

        # Pool scale: multiply pool output by w2
        self.pool_scale = ScaleModule(w2)

        # Copy statistics from herpn to pool_scale (applied after pooling)
        if hasattr(self.herpn, 'input_min'):
            # Pool_scale input is after pool, use herpn output range as approximation
            self.pool_scale.input_min = self.herpn.output_min
            self.pool_scale.input_max = self.herpn.output_max
            self.pool_scale.output_min = self.herpn.output_min
            self.pool_scale.output_max = self.herpn.output_max
            # Shape will be recorded during forward pass

        # Copy level if it exists
        if hasattr(self.herpn, 'level'):
            self.pool_scale.level = self.herpn.level

    def compile(self, trace=None):
        """Compile the HerPN module and pool scale.

        Args:
            trace: Optional traced model to look up levels from (since levels are only assigned in trace)
        """
        # Copy level and fhe_input_shape from traced herpn module if it doesn't have them yet
        # Levels are only assigned to modules in the trace, so we need to look them up there
        if self.herpn is not None:
            if not hasattr(self.herpn, 'level') or self.herpn.level is None:
                # After fusion, bn1 is replaced by herpn, so look for herpn in trace
                level = None
                fhe_input_shape = None
                input_shape = None
                input_gap = None
                if trace is not None:
                    # Search for herpnpool.herpn specifically (not just any herpn)
                    for name, mod in trace.named_modules():
                        # Check if this is herpnpool's herpn module
                        if 'herpnpool' in name and 'herpn' in name and isinstance(mod, ChannelSquare):
                            if hasattr(mod, 'level'):
                                # Found the herpnpool herpn with a level, copy all shape attributes
                                level = mod.level
                                if hasattr(mod, 'fhe_input_shape'):
                                    fhe_input_shape = mod.fhe_input_shape
                                if hasattr(mod, 'input_shape'):
                                    input_shape = mod.input_shape
                                if hasattr(mod, 'input_gap'):
                                    input_gap = mod.input_gap
                                break

                if level is not None:
                    self.herpn.level = level
                    self.herpn.depth = 1  # HerPN always has depth 1
                    if fhe_input_shape is not None:
                        self.herpn.fhe_input_shape = fhe_input_shape
                    if input_shape is not None:
                        self.herpn.input_shape = input_shape
                    if input_gap is not None:
                        self.herpn.input_gap = input_gap

        if self.herpn is not None and hasattr(self.herpn, 'compile'):
            if hasattr(self.herpn, 'level') and self.herpn.level is not None:
                self.herpn.compile()

        # Compile pool scale module
        if hasattr(self, 'pool_scale') and hasattr(self.pool_scale, 'compile'):
            if hasattr(self.pool_scale, 'level') and self.pool_scale.level is not None:
                self.pool_scale.compile()

    def forward(self, x):
        if not self.he_mode:
            # Cleartext forward using fused HerPN to avoid graph branching
            if self.herpn is not None:
                # Use fused HerPN (factored: x² + a1·x + a0)
                out = self.herpn(x)
                # Apply adaptive pooling, then scale by HerPN scale_factor
                # Matches CryptoFace: pool(herpn_factored(x)) * a2
                out = self.pool(out)
                # Use ScaleModule to avoid tracer shape validation issues
                out = self.pool_scale(out)
                return out
            else:
                # During training, use unfused BatchNorms (CryptoFace HerPN.forward lines 128-134)
                x0 = self.bn0(torch.ones_like(x))
                x1 = self.bn1(x)
                x2 = self.bn2((torch.square(x) - 1) / math.sqrt(2))
                out = (
                    torch.divide(x0, math.sqrt(2 * math.pi))
                    + torch.divide(x1, 2)
                    + torch.divide(x2, math.sqrt(4 * math.pi))
                )
                # Apply weight and bias (CryptoFace line 133)
                if hasattr(self, 'herpn_weight') and hasattr(self, 'herpn_bias'):
                    out = self.herpn_weight * out + self.herpn_bias

                # Apply adaptive pooling
                out = self.pool(out)
                return out
        else:
            # FHE mode
            out = self.herpn(x)
            # Apply adaptive pooling, then scale by HerPN scale_factor
            # Matches CryptoFace: pool(herpn_factored(x)) * a2
            out = self.pool(out)
            # Use ScaleModule (uses pre-encoded scale_fhe in FHE mode)
            out = self.pool_scale(out)
            return out


class Backbone(on.Module):
    """
    Feature extraction backbone for PCNN.
    """

    def __init__(self, output_size, input_size=16):
        super(Backbone, self).__init__()
        self.output_size = output_size
        self.input_size = input_size

        # Initial convolution (bias=False to match CryptoFace)
        self.conv = on.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

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
            [on.Linear(self.dim, 256) for _ in range(self.N)]
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