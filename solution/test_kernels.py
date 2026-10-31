"""Correctness tests for custom kernels against plain PyTorch, forward and backward.

Run on a GPU: `python -m pytest solution -q`. Add a test like this for every kernel you write.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

TOLERANCE = {torch.float16: 2e-3, torch.float32: 1e-4}


# ============ SwiGLU tests (custom 2D block-tiled kernel) ============

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(4, 512, 8960), (3, 77, 1001)])
def test_swiglu_matches_pytorch(dtype, shape):
    from solution.swiglu import swiglu

    torch.manual_seed(0)
    gate = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    up = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device="cuda", dtype=dtype)

    # Reference 1: FP32 upcast reference (Triton computes in FP32)
    ref_fp32 = torch.nn.functional.silu(gate.float()) * up.float()
    ref_fp32.backward(grad.float())
    ref_dgate_fp32, ref_dup_fp32 = gate.grad.float().clone(), up.grad.float().clone()
    gate.grad = up.grad = None

    # Reference 2: Native dtype reference
    ref_native = torch.nn.functional.silu(gate) * up
    ref_native.backward(grad)
    ref_dgate_native, ref_dup_native = gate.grad.clone(), up.grad.clone()
    gate.grad = up.grad = None

    out = swiglu(gate, up)
    out.backward(grad)
    tol = TOLERANCE[dtype]

    # Verify matching FP32 precision
    torch.testing.assert_close(out.float(), ref_fp32, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad.float(), ref_dgate_fp32, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad.float(), ref_dup_fp32, rtol=tol, atol=tol)

    # Verify matching native dtype within tolerance
    torch.testing.assert_close(out, ref_native, rtol=tol, atol=tol)
    torch.testing.assert_close(gate.grad, ref_dgate_native, rtol=tol, atol=tol)
    torch.testing.assert_close(up.grad, ref_dup_native, rtol=tol, atol=tol)


# ============ RMSNorm tests ============

def _reference_rms_norm(x, weight, eps):
    """Reference RMSNorm in PyTorch, matching Qwen2RMSNorm behavior."""
    x_float = x.float()
    variance = x_float.pow(2).mean(-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    return (weight.float() * x_normed).to(weight.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [
    (4, 512, 1536),   # Qwen2.5-1.5B hidden size
    (2, 128, 1536),
    (1, 1, 1536),
    (3, 77, 768),      # Non-power-of-2 sequence length
])
def test_rms_norm_forward(dtype, shape):
    from solution.rms_norm import triton_rms_norm

    torch.manual_seed(42)
    eps = 1e-6
    N = shape[-1]

    x = torch.randn(shape, device="cuda", dtype=dtype)
    weight = torch.randn(N, device="cuda", dtype=torch.float32)

    ref = _reference_rms_norm(x, weight, eps)
    out = triton_rms_norm(x, weight, eps)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("weight_requires_grad", [True, False])
@pytest.mark.parametrize("shape", [
    (4, 512, 1536),
    (2, 128, 1536),
    (1, 7, 768),
])
def test_rms_norm_backward(dtype, weight_requires_grad, shape):
    from solution.rms_norm import triton_rms_norm

    torch.manual_seed(42)
    eps = 1e-6
    N = shape[-1]

    x_ref = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w_ref = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=weight_requires_grad)
    grad = torch.randn(shape, device="cuda", dtype=torch.float32)

    ref_out = _reference_rms_norm(x_ref, w_ref, eps)
    ref_out.backward(grad)
    ref_dx = x_ref.grad.clone()
    ref_dw = w_ref.grad.clone() if weight_requires_grad else None

    x_tri = x_ref.detach().clone().requires_grad_(True)
    w_tri = w_ref.detach().clone().requires_grad_(weight_requires_grad)

    tri_out = triton_rms_norm(x_tri, w_tri, eps)
    tri_out.backward(grad)

    tol = 5e-3 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(tri_out.float(), ref_out.float().detach(), rtol=tol, atol=tol)
    torch.testing.assert_close(x_tri.grad.float(), ref_dx.float(), rtol=tol, atol=tol)
    if weight_requires_grad:
        assert w_tri.grad is not None
        torch.testing.assert_close(w_tri.grad.float(), ref_dw.float(), rtol=tol, atol=tol)
    else:
        assert w_tri.grad is None


# ============ RoPE tests ============

def _reference_rope(q, k, cos, sin):
    """Reference RoPE in PyTorch from transformers.models.qwen2.modeling_qwen2."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos_u = cos.unsqueeze(1) if cos.ndim == 3 else cos
    sin_u = sin.unsqueeze(1) if sin.ndim == 3 else sin

    q_out = (q * cos_u) + (rotate_half(q) * sin_u)
    k_out = (k * cos_u) + (rotate_half(k) * sin_u)
    return q_out, k_out


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("B, H, S, D", [
    (4, 12, 512, 128),  # Qwen2.5-1.5B Q shape
    (4, 2, 512, 128),   # Qwen2.5-1.5B K shape (GQA)
    (2, 8, 128, 128),
    (1, 4, 37, 128),    # Non-power-of-2 seq_len
])
def test_rope_forward(dtype, B, H, S, D):
    from solution.rope import apply_triton_rope

    torch.manual_seed(42)
    q = torch.randn((B, H, S, D), device="cuda", dtype=dtype)
    cos = torch.randn((B, S, D), device="cuda", dtype=dtype)
    sin = torch.randn((B, S, D), device="cuda", dtype=dtype)

    ref_q, _ = _reference_rope(q, q, cos, sin)
    tri_q = apply_triton_rope(q, cos, sin)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(tri_q, ref_q, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("B, H, S, D", [
    (4, 12, 512, 128),
    (4, 2, 512, 128),
    (1, 4, 37, 128),
])
def test_rope_backward(dtype, B, H, S, D):
    from solution.rope import apply_triton_rope

    torch.manual_seed(42)
    q_ref = torch.randn((B, H, S, D), device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn((B, S, D), device="cuda", dtype=dtype)
    sin = torch.randn((B, S, D), device="cuda", dtype=dtype)
    grad = torch.randn((B, H, S, D), device="cuda", dtype=dtype)

    ref_q, _ = _reference_rope(q_ref, q_ref, cos, sin)
    ref_q.backward(grad)
    ref_dq = q_ref.grad.clone()

    q_tri = q_ref.detach().clone().requires_grad_(True)
    tri_q = apply_triton_rope(q_tri, cos, sin)
    tri_q.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(tri_q, ref_q.detach(), rtol=tol, atol=tol)
    torch.testing.assert_close(q_tri.grad, ref_dq, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("B, H, S, D", [
    (4, 12, 512, 128),
    (4, 2, 512, 128),
])
def test_rope_realistic_frequencies(dtype, B, H, S, D):
    """Test RoPE with authentic rotary frequency embeddings (bounded on unit circle)."""
    from solution.rope import apply_triton_rope

    torch.manual_seed(42)
    inv_freq = 1.0 / (1000000.0 ** (torch.arange(0, D, 2, device="cuda", dtype=torch.float32) / D))
    t = torch.arange(S, device="cuda", dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0).expand(B, -1, -1).to(dtype)
    sin = emb.sin().unsqueeze(0).expand(B, -1, -1).to(dtype)

    q_ref = torch.randn((B, H, S, D), device="cuda", dtype=dtype, requires_grad=True)
    grad = torch.randn((B, H, S, D), device="cuda", dtype=dtype)

    ref_q, _ = _reference_rope(q_ref, q_ref, cos, sin)
    ref_q.backward(grad)
    ref_dq = q_ref.grad.clone()

    q_tri = q_ref.detach().clone().requires_grad_(True)
    tri_q = apply_triton_rope(q_tri, cos, sin)
    tri_q.backward(grad)

    tol = TOLERANCE[dtype]
    torch.testing.assert_close(tri_q, ref_q.detach(), rtol=tol, atol=tol)
    torch.testing.assert_close(q_tri.grad, ref_dq, rtol=tol, atol=tol)
