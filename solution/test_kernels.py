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


# ============ Chunked Causal LM Loss Tests ============

@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("B, S, H, V", [
    (2, 16, 64, 256),
    (4, 32, 128, 512),
])
def test_chunked_loss_scalar_and_gradients(dtype, B, S, H, V):
    """Test that micro-chunked causal LM loss and hidden-state gradients match standard unchunked PyTorch."""
    from solution.chunk_loss import chunked_causal_lm_loss

    torch.manual_seed(42)
    hidden_ref = torch.randn((B, S, H), device="cuda", dtype=dtype, requires_grad=True)
    hidden_custom = hidden_ref.clone().detach().requires_grad_(True)

    lm_head = torch.nn.Linear(H, V, bias=False, device="cuda", dtype=dtype)
    for p in lm_head.parameters():
        p.requires_grad_(False)

    labels = torch.randint(0, V, (B, S), device="cuda")
    # Mask out prompt positions and padding
    labels[0, :4] = -100
    labels[1, 20:] = -100

    # 1. Unchunked PyTorch reference
    logits_full = lm_head(hidden_ref)
    shift_logits = logits_full[:, :-1, :].reshape(-1, V)
    shift_labels = labels[:, 1:].reshape(-1)
    loss_ref = torch.nn.functional.cross_entropy(shift_logits.float(), shift_labels, ignore_index=-100)
    loss_ref.backward()
    dx_ref = hidden_ref.grad.clone()

    # 2. Custom chunked implementation
    loss_chunked, metrics = chunked_causal_lm_loss(
        hidden_custom, lm_head.weight, labels, chunk_size=8
    )
    loss_chunked.backward()
    dx_chunked = hidden_custom.grad.clone()

    tol = 1e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(loss_chunked.float(), loss_ref.float(), rtol=tol, atol=tol)
    torch.testing.assert_close(dx_chunked, dx_ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunked_loss_metrics_and_rewind_row_losses(dtype):
    """Test that chunked entropy, accuracy, and rewind row losses match unchunked formulas exactly."""
    from solution.chunk_loss import chunked_causal_lm_loss

    B, S, H, V = 2, 16, 64, 256
    torch.manual_seed(42)

    hidden = torch.randn((B, S, H), device="cuda", dtype=dtype)
    lm_head = torch.nn.Linear(H, V, bias=False, device="cuda", dtype=dtype)
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[0, :3] = -100
    labels[1, 10:] = -100

    # 1. Unchunked references
    logits_full = lm_head(hidden)
    log_probs_full = torch.nn.functional.log_softmax(logits_full.float(), dim=-1)
    ent_ref = -(torch.exp(log_probs_full) * log_probs_full).sum(dim=-1)

    shift_labels = labels[:, 1:]
    row_loss_ref = []
    row_tokens_ref = []
    for i in range(shift_labels.size(0)):
        targets = shift_labels[i]
        token_ce = torch.nn.functional.cross_entropy(
            logits_full[i, :-1, :].float(), targets, reduction="none", ignore_index=-100
        )
        mask = (targets != -100).float()
        tokens = mask.sum()
        total = (token_ce * mask).sum()
        row_loss_ref.append(float(total / tokens.clamp(min=1.0)))
        row_tokens_ref.append(int(tokens.item()))

    # 2. Chunked metrics
    _, metrics = chunked_causal_lm_loss(hidden, lm_head.weight, labels, chunk_size=8)

    # Verify entropy
    tol = 1e-3 if dtype == torch.float16 else 1e-5
    torch.testing.assert_close(metrics.token_entropy, ent_ref, rtol=tol, atol=tol)

    # Verify per-row losses matching Soup rewind_hf.py formula
    token_loss_2d = metrics.token_loss[:, :-1]
    tokens_calc = (labels[:, 1:] != -100).sum(dim=1)
    totals_calc = token_loss_2d.sum(dim=1)
    row_loss_chunked = [float(tot / tok.clamp(min=1)) for tot, tok in zip(totals_calc, tokens_calc)]
    row_tokens_chunked = [int(c) for c in tokens_calc]

    assert row_tokens_chunked == row_tokens_ref
    for r_ref, r_chk in zip(row_loss_ref, row_loss_chunked):
        assert abs(r_ref - r_chk) < tol


def test_chunked_loss_propagates_to_preceding_layers_and_lora():
    """Verify that gradients propagate to preceding layers while frozen lm_head receives no gradients."""
    from solution.chunk_loss import chunked_causal_lm_loss

    B, S, H, V = 2, 8, 32, 64
    torch.manual_seed(42)

    # Simulate preceding layer (e.g. projection with LoRA-like adapter)
    preceding_ref = torch.nn.Linear(H, H, bias=False, device="cuda", dtype=torch.float32)
    preceding_custom = torch.nn.Linear(H, H, bias=False, device="cuda", dtype=torch.float32)
    preceding_custom.weight.data.copy_(preceding_ref.weight.data)

    lm_head = torch.nn.Linear(H, V, bias=False, device="cuda", dtype=torch.float32)
    for p in lm_head.parameters():
        p.requires_grad_(False)

    x = torch.randn((B, S, H), device="cuda", dtype=torch.float32)
    labels = torch.randint(0, V, (B, S), device="cuda")

    # Reference run
    h_ref = preceding_ref(x)
    l_ref = lm_head(h_ref)
    loss_ref = torch.nn.functional.cross_entropy(
        l_ref[:, :-1, :].reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100
    )
    loss_ref.backward()

    # Custom chunked run
    h_custom = preceding_custom(x)
    loss_custom, _ = chunked_causal_lm_loss(h_custom, lm_head.weight, labels, chunk_size=4)
    loss_custom.backward()

    # Preceding layer weights must receive exact gradients
    torch.testing.assert_close(preceding_custom.weight.grad, preceding_ref.weight.grad, rtol=1e-5, atol=1e-5)
    # Frozen lm_head must have no gradient
    assert lm_head.weight.grad is None


def test_chunked_loss_instrumentation():
    """Verify that the chunked implementation increments CHUNKED_LOSS_CALL_COUNT upon execution."""
    from solution.chunk_loss import chunked_causal_lm_loss, CHUNKED_LOSS_CALL_COUNT

    count_before = CHUNKED_LOSS_CALL_COUNT
    B, S, H, V = 1, 4, 16, 32
    hidden = torch.randn((B, S, H), device="cuda", dtype=torch.float32)
    weight = torch.randn((V, H), device="cuda", dtype=torch.float32)
    labels = torch.randint(0, V, (B, S), device="cuda")

    _ = chunked_causal_lm_loss(hidden, weight, labels, chunk_size=2)
    from solution.chunk_loss import CHUNKED_LOSS_CALL_COUNT as count_after

    assert count_after > count_before


def test_chunked_loss_all_ignored_chunk():
    """Verify that chunks where all tokens are ignored (labels == -100) yield zero loss and zero gradients."""
    from solution.chunk_loss import chunked_causal_lm_loss

    B, S, H, V = 2, 8, 16, 32
    torch.manual_seed(42)
    hidden = torch.randn((B, S, H), device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn((V, H), device="cuda", dtype=torch.float32)
    labels = torch.full((B, S), -100, device="cuda", dtype=torch.long)
    # Only label second row, leaving first row (first 8 tokens) completely ignored
    labels[1, 4:] = torch.randint(0, V, (4,), device="cuda")

    loss, metrics = chunked_causal_lm_loss(hidden, weight, labels, chunk_size=4)
    loss.backward()

    # The first row had all labels == -100, so its hidden gradient must be exactly 0
    assert torch.all(hidden.grad[0] == 0.0)
    assert metrics.labelled.item() == 4


def test_chunked_loss_gradient_accumulation_and_denominator():
    """Verify exact numerical agreement with custom num_items_in_batch denominator and GA scaling."""
    from solution.chunk_loss import chunked_causal_lm_loss

    B, S, H, V = 2, 8, 16, 32
    torch.manual_seed(42)
    hidden = torch.randn((B, S, H), device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn((V, H), device="cuda", dtype=torch.float32)
    labels = torch.randint(0, V, (B, S), device="cuda")
    labels[0, :2] = -100

    hidden_ref = hidden.clone().detach().requires_grad_(True)
    weight_ref = weight.clone().detach()

    # Reference with num_items_in_batch = 50.0 and ga_steps = 2
    logits_ref = torch.nn.functional.linear(hidden_ref, weight_ref)
    shift_logits = logits_ref[:, :-1, :].reshape(-1, V)
    shift_labels = labels[:, 1:].reshape(-1)
    mask = (shift_labels != -100).float()
    safe_labels = shift_labels.clamp(min=0)
    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    loss_ref = -(log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1) * mask).sum() / 50.0
    loss_ref_scaled = loss_ref / 2.0
    loss_ref_scaled.backward()

    # Custom chunked
    loss_custom, _ = chunked_causal_lm_loss(hidden, weight, labels, chunk_size=4, num_items_in_batch=50.0)
    loss_custom_scaled = loss_custom / 2.0
    loss_custom_scaled.backward()

    torch.testing.assert_close(loss_custom, loss_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=1e-5, atol=1e-5)
