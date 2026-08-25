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


@pytest.mark.parametrize("num_tokens", [1, 64, 777])
def test_rmsnorm_qkv_view(num_tokens):
    """真实调用点：q / k 是 qkv.split() 切出来的非连续视图。

    Qwen3-0.6B 的布局：16 个 q head、8 个 kv head、head_dim=128，
    所以 q 的行内 stride 是 128、跨 token 却要跳 4096，不能当连续张量处理。
    """
    n_heads, n_kv_heads, head_dim = 16, 8, 128
    q_size, kv_size = n_heads * head_dim, n_kv_heads * head_dim
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
    q, k, _ = qkv.split([q_size, kv_size, kv_size], dim=-1)

    for t, n_h in [(q.view(-1, n_heads, head_dim), n_heads),
                   (k.view(-1, n_kv_heads, head_dim), n_kv_heads)]:
        assert num_tokens == 1 or not t.is_contiguous()   # 只有一行时切片本来就连续
        out = rmsnorm(t, w, 1e-6)
        assert out.shape == (num_tokens, n_h, head_dim)
        torch.testing.assert_close(out, ref_rmsnorm(t, w, 1e-6), rtol=1e-2, atol=1e-2)


def test_fused_add_rmsnorm_3d_view():
    """fused 版本走非连续视图时，两个原地写回都要落在正确的位置上"""
    base = torch.randn(64, 2, 16, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    x, r = base[:, 0], base[:, 1]
    assert not x.is_contiguous()
    ref_r = (x.float() + r.float()).to(x.dtype)
    ref_x = ref_rmsnorm(ref_r, w, 1e-6)
    guard = base.clone()

    fused_add_rmsnorm(x, r, w, 1e-6)
    torch.testing.assert_close(r, ref_r, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x, ref_x, rtol=1e-2, atol=1e-2)
    # 原地写回不能越界踩到相邻的 head
    assert not torch.equal(guard, base)


@pytest.mark.parametrize("shape", [(128, 6144), (8192, 6144), (55, 512)])
def test_silu_and_mul(shape):
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    a, b = x.chunk(2, -1)
    ref = F.silu(a.float()) * b.float()
    torch.testing.assert_close(silu_and_mul(x).float(), ref, rtol=1e-2, atol=1e-2)