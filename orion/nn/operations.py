import math
import torch

from .module import Module, timer

class Add(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(0)

    def compute_fhe_output_shape(self, **kwargs):
        fhe_input_shape = kwargs["fhe_input_shape"]

        if isinstance(fhe_input_shape, list):
            return fhe_input_shape[0]
        return fhe_input_shape

    def forward(self, x, y):
        return x + y
    

class Mult(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def compute_fhe_output_shape(self, **kwargs):
        fhe_input_shape = kwargs["fhe_input_shape"]

        if isinstance(fhe_input_shape, list):
            return fhe_input_shape[0]
        return fhe_input_shape

    def forward(self, x, y):
        return x * y
    

class Rescale(Module):
    """Explicit rescale operation that consumes one level.

    This module marks an explicit rescaling point in the computation graph,
    allowing the level assignment algorithm to account for it properly.
    In Lattigo, the actual rescaling happens automatically when there's a
    level mismatch, but having an explicit Rescale module prevents surprise
    implicit rescales that introduce uncontrolled errors.

    This is similar to SEAL's explicit rescale_to_next_inplace() approach.
    """
    def __init__(self):
        super().__init__()
        self.set_depth(1)  # Consumes one level

    def forward(self, x):
        # Pass-through in both cleartext and FHE mode
        # Lattigo handles the actual rescaling automatically when needed
        return x


class Bootstrap(Module):
    def __init__(self, input_min, input_max, input_level):
        super().__init__()
        self.input_min = input_min
        self.input_max = input_max
        self.input_level = input_level
        self.prescale = 1
        self.postscale = 1
        self.constant = 0

    def extra_repr(self):
        l_eff = len(self.scheme.params.get_logq()) - 1
        return f"l_eff={l_eff}"

    def fit(self):
        center = (self.input_min + self.input_max) / 2 
        half_range = (self.input_max - self.input_min) / 2
        self.low = (center - (self.margin * half_range))
        self.high = (center + (self.margin * half_range))

        # We'll want to scale from [A, B] into [-1, 1] using a value of the
        # form 1 / integer, so that way our multiplication back to the range
        # [A, B] (by integer) after bootstrapping doesn't consume a level.
        if self.high - self.low > 2:
            self.postscale = math.ceil((self.high - self.low) / 2)
            self.prescale = 1 / self.postscale

        self.constant = -(self.low + self.high) / 2 

    def compile(self):
        # We'll then encode the prescale at the level of the input ciphertext
        # to ensure its rescaling is errorless
        elements = self.fhe_input_shape.numel()
        curr_slots = 2 ** math.ceil(math.log2(elements))

        prescale_vec = torch.zeros(curr_slots)
        prescale_vec[:elements] = self.prescale

        ql = self.scheme.encoder.get_moduli_chain()[self.input_level]
        self.prescale_ptxt = self.scheme.encoder.encode(
            prescale_vec, level=self.input_level, scale=ql)

    @timer
    def forward(self, x):
        if not self.he_mode:
            return x
        
        # Shift and scale into range [-1, 1]. Important caveat -- here we first
        # shift, then scale. This let's us zero out unused slots and enables
        # sparse bootstrapping (i.e., where slots < N/2).
        if self.constant != 0:
            x += self.constant
        x *= self.prescale_ptxt
 
        x = x.bootstrap()

        # Scale and shift back to the original range
        if self.postscale != 1:
            x *= self.postscale 
        if self.constant != 0:
            x -= self.constant

        return x


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

    @timer
    def forward(self, x):
        if self.he_mode:
            # FHE mode: use pre-encoded expanded scale
            return x * self.scale_fhe
        else:
            # Cleartext: direct multiplication (Python broadcasting handles [C,1,1] * [B,C,H,W])
            return x * self.scale_raw.unsqueeze(0)




