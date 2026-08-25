import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.kernels import USE_TRITON
from nanovllm.kernels.activation import silu_and_mul


class SiluAndMul(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not USE_TRITON:
            return self.forward_torch(x)
        return silu_and_mul(x)

    @torch.compile
    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
