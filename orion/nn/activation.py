import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from orion.nn.module import Module, timer
from orion.nn.operations import Mult


class Activation(Module):
    def __init__(self, coeffs):
        super().__init__()
        self.coeffs = coeffs 
        self.output_scale = None
        self.set_depth()

    def extra_repr(self):
        return super().extra_repr() + f", degree={len(self.coeffs)-1}"
    
    def set_depth(self):
        self.depth = int(math.ceil(math.log2(len(self.coeffs))))

    def set_output_scale(self, output_scale):
        self.output_scale = output_scale

    def compile(self):
        self.poly = self.scheme.poly_evaluator.generate_monomial(self.coeffs)

    @timer
    def forward(self, x):
        if self.he_mode:
            return self.scheme.poly_evaluator.evaluate_polynomial( 
                x, self.poly, self.output_scale)
        
        # Horner's method
        out = 0
        for coeff in self.coeffs:
            out = coeff + x * out
            
        return out
    

class Quad(Module):
    def __init__(self):
        super().__init__()
        self.set_depth(1)

    def forward(self, x):
        out = x * x 
        if self.he_mode:
            out.set_scale(x.scale()) 
        return out
    

class Chebyshev(Module):
    def __init__(self, degree: int, fn, within_composite=False):
        super().__init__()
        self.degree = degree
        self.fn = fn
        self.within_composite = within_composite
        self.coeffs = None
       
        self.output_scale = None
        self.prescale = 1 
        self.constant = 0

    def extra_repr(self):
        return super().extra_repr() + f", degree={self.degree}"

    def fit(self):
        if not self.within_composite:
            center = (self.input_min + self.input_max) / 2 
            half_range = (self.input_max - self.input_min) / 2
            self.low = (center - (self.margin * half_range))
            self.high = (center + (self.margin * half_range))

            nodes = np.polynomial.chebyshev.chebpts1(self.degree + 1)
            if self.low < -1 or self.high > 1:
                self.prescale = 2 / (self.high - self.low) 
                self.constant = -self.prescale * (self.low + self.high) / 2 
                evals = (nodes + 1) * (self.high - self.low) / 2 + self.low
            else:
                evals = nodes
            
            evals = torch.tensor(evals)
            T = np.polynomial.Chebyshev.fit(nodes, self.fn(evals), self.degree)
            self.set_coeffs(T.coef.tolist())
            self.set_depth()

    def set_coeffs(self, coeffs):
        self.coeffs = coeffs

    def set_depth(self):
        self.depth = int(math.ceil(math.log2(self.degree+1)))
        if self.prescale != 1: # additional level needed
            self.depth += 1

    def set_output_scale(self, output_scale):
        self.output_scale = output_scale

    def compile(self):
        self.poly = self.scheme.poly_evaluator.generate_chebyshev(self.coeffs)

    @timer
    def forward(self, x):  
        if not self.he_mode:
            return self.fn(x)

        # Scale into [-1, 1] if needed.
        if not self.fused:
            if self.prescale != 1:
                x *= self.prescale 
            if self.constant != 0:
                x += self.constant

        return self.scheme.poly_evaluator.evaluate_polynomial(
            x, self.poly, self.output_scale)
    

class ELU(Chebyshev):
    def __init__(self, alpha=1.0, degree=31):
        self.alpha = alpha
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))
    

class Hardshrink(Chebyshev): 
    def __init__(self, degree=31, lambd=0.5):
        self.lambd = lambd
        super().__init__(degree, self.fn) 
    
    def fn(self, x):
        return torch.where((x > self.lambd) | (x < -self.lambd), x, torch.tensor(0.0))
    

class GELU(Chebyshev): 
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
    
    def fn(self, x):
        return F.gelu(x)
    

class SiLU(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 

    def fn(self, x):
        return F.silu(x)
    

class Sigmoid(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return F.sigmoid(x)
    

class SELU(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    

class Softplus(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return F.softplus(x)
    

class Mish(Chebyshev):
    def __init__(self, degree=31):
        super().__init__(degree, self.fn) 
        
    def fn(self, x):
        return x * torch.tanh(F.softplus(x))
    

class _Sign(Module):
    def __init__(
        self, 
        degrees=[15,15,27],
        prec=128,
        logalpha=6,
        logerr=12,
    ):
        super().__init__()
        self.degrees = degrees
        self.prec = prec 
        self.logalpha = logalpha 
        self.logerr = logerr 
        self.mult = Mult()

        acts = []
        for i, degree in enumerate(degrees):
            is_last = (i == len(degrees) - 1)
            fn = self.fn1 if not is_last else self.fn2
            act = Chebyshev(degree, fn, within_composite=True)
            acts.append(act)
        
        self.acts = nn.Sequential(*acts)

    def extra_repr(self):
        return super().extra_repr() + f", degrees={self.degrees}"
            
    def fit(self):
        debug = self.scheme.params.get_debug_status()
        self.coeffs = self.scheme.poly_evaluator.generate_minimax_sign_coeffs(
            self.degrees, self.prec, self.logalpha, self.logerr, debug)
        
        for i, coeffs in enumerate(self.coeffs):
            self.acts[i].set_coeffs(coeffs)
            self.acts[i].set_depth()
                
    def fn1(self, x):
        return torch.where(x <= 0, torch.tensor(-1.0), torch.tensor(1.0))
    
    def fn2(self, x):
        return torch.where(x <= 0, torch.tensor(0.0), torch.tensor(1.0))

    def forward(self, x):
        if self.he_mode:
            l1 = x.level() 
            l2 = self.acts[-1].level - self.acts[-1].depth 
            
            # We'll calculate the output level of sign on the fly by 
            # comparing and taking the minimum of x and sign(x), as FHE
            # multiplication will do the same. Then, we'll set the output
            # scale of sign to be the modulus in the chain at this level.
            # This way, rescaling divides ql / ql and is exact.
            output_level = min(l1, l2)
            ql = self.scheme.encoder.get_moduli_chain()[output_level]
            self.acts[-1].set_output_scale(ql)

        # Composite polynomial evaluation
        for act in self.acts:
            x = act(x)
        return x


class ReLU(Module):
    def __init__(self, 
                 degrees=[15,15,27],
                 prec=128,
                 logalpha=6,
                 logerr=12,
    ):
        super().__init__()
        self.degrees = degrees 
        self.prec = prec 
        self.logalpha = logalpha
        self.logerr = logerr 
        self.sign = _Sign(degrees, prec, logalpha, logerr)
        self.mult1 = Mult()
        self.mult2 = Mult()

        self.prescale = 1 
        self.postscale = 1

    def extra_repr(self):
        return super().extra_repr() + f", degrees={self.degrees}"
    
    def fit(self):
        self.input_min = self.mult1.input_min 
        self.input_max = self.mult1.input_max

        absmax = max(abs(self.input_min), abs(self.input_max)) * self.margin
        if absmax > 1:
            self.postscale = int(math.ceil(absmax))
            self.prescale = 1 / self.postscale
    
    @timer
    def forward(self, x):
        x = self.mult1(x, self.prescale)
        x = self.mult2(x, self.sign(x))
        x *= self.postscale # integer mult, no level consumed
        return x


class Cleanse(Module):
    def __init__(self, iter=3):
        super().__init__()
        self.polys = nn.Sequential(*[Activation(coeffs=[-2, 3, 0, 0]) for _ in range(iter)])

    def forward(self, x):
        for poly in self.polys:
            x = poly(x)
        return x


class SqMethod1(Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.poly = Activation(coeffs=[-2/self.p**2, 0, 1])

    def forward(self, x):
        return self.poly(x)


class SqMethod2(Module):
    def __init__(self, r):
        super().__init__()
        self.r = r
        self.quad = nn.Sequential(*[Quad() for _ in range(self.r)])

    def forward(self, x):
        for quad in self.quad:
            x = quad(x)
        return x


class SqMethod(Module):
    def __init__(self, p, r):
        super().__init__()
        self.p = p
        self.r = r
        self.sq1 = SqMethod1(p)
        self.sq2 = SqMethod2(r)

    def forward(self, x):
        x = self.sq1(x)
        x = self.sq2(x)
        return x


class EIF(Module):
    def __init__(self, p, r, iter):
        super().__init__()
        self.p = p
        self.r = r
        self.iter = iter
        self.cleanse = Cleanse(iter)
        self.sq = SqMethod(p, r)

    def forward(self, x):
        x = self.sq(x)
        x = self.cleanse(x)
        return x


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