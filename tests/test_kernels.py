import torch
import torch.nn.functional as F
import pytest

from nanovllm.kernels.layernorm import rmsnorm, fused_add_rmsnorm
from nanovllm.kernels.activation import silu_and_mul


def ref_rmsnorm(x, w, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w


@pytest.mark.parametrize("shape", [(1, 1024), (128, 1024), (8192, 1024), (777, 128)])
def test_rmsnorm(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(shape[-1], dtype=torch.bfloat16, device="cuda")
    out = rmsnorm(x, w, 1e-6)
    ref = ref_rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(128, 1024), (8192, 1024), (333, 128)])
def test_fused_add_rmsnorm(shape):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    r = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(shape[-1], dtype=torch.bfloat16, device="cuda")

    ref_r = (x.float() + r.float()).to(x.dtype)
    ref_x = ref_rmsnorm(ref_r, w, 1e-6)

    out_x, out_r = fused_add_rmsnorm(x.clone(), r.clone(), w, 1e-6)
    torch.testing.assert_close(out_r, ref_r, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out_x, ref_x, rtol=1e-2, atol=1e-2)


def test_rmsnorm_3d():
    """对应 q_norm / k_norm 的 (N, H, head_dim) 输入"""
    x = torch.randn(64, 16, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(rmsnorm(x, w, 1e-6), ref_rmsnorm(x, w, 1e-6), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(128, 6144), (8192, 6144), (55, 512)])
def test_silu_and_mul(shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    a, b = x.chunk(2, -1)
    ref = F.silu(a.float()) * b.float()
    torch.testing.assert_close(silu_and_mul(x).float(), ref, rtol=1e-2, atol=1e-2)