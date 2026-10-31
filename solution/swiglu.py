"""Fused SwiGLU (silu(gate) * up) Triton kernels for Qwen2 MLP — forward and backward.

Optimized 2D block-tiled implementation that fuses the SiLU activation (gate * sigmoid(gate))
and elementwise multiplication with up into a single memory-bandwidth-efficient kernel.
In the backward pass, fuses upstream gradient propagation for both gate and up projections.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 16
BLOCK_N = 64


@triton.jit
def swiglu_fwd_kernel(
    gate_ptr, up_ptr, out_ptr,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_om, stride_on,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    g_ptrs = gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    u_ptrs = up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)
    res = g * sig * u

    tl.store(o_ptrs, res.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_bwd_kernel(
    grad_ptr, gate_ptr, up_ptr, dgate_ptr, dup_ptr,
    stride_grad_m, stride_grad_n,
    stride_gm, stride_gn,
    stride_um, stride_un,
    stride_dgm, stride_dgn,
    stride_dum, stride_dun,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    grad_ptrs = grad_ptr + offs_m[:, None] * stride_grad_m + offs_n[None, :] * stride_grad_n
    g_ptrs = gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    u_ptrs = up_ptr + offs_m[:, None] * stride_um + offs_n[None, :] * stride_un
    dg_ptrs = dgate_ptr + offs_m[:, None] * stride_dgm + offs_n[None, :] * stride_dgn
    du_ptrs = dup_ptr + offs_m[:, None] * stride_dum + offs_n[None, :] * stride_dun

    grad = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptrs, mask=mask, other=0.0).to(tl.float32)

    sig = tl.sigmoid(g)

    du = grad * g * sig
    dg = grad * u * sig * (1.0 + g * (1.0 - sig))

    tl.store(dg_ptrs, dg.to(dgate_ptr.dtype.element_ty), mask=mask)
    tl.store(du_ptrs, du.to(dup_ptr.dtype.element_ty), mask=mask)


class SwiGLU(torch.autograd.Function):
    """silu(gate) * up in 2D block-tiled Triton kernels for forward and backward."""

    @staticmethod
    def forward(ctx, gate, up):
        orig_shape = gate.shape
        gate_2d = gate.reshape(-1, orig_shape[-1]).contiguous()
        up_2d = up.to(gate.dtype).reshape(-1, orig_shape[-1]).contiguous()

        M, N = gate_2d.shape
        out_2d = torch.empty_like(gate_2d)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        swiglu_fwd_kernel[grid](
            gate_2d, up_2d, out_2d,
            gate_2d.stride(0), gate_2d.stride(1),
            up_2d.stride(0), up_2d.stride(1),
            out_2d.stride(0), out_2d.stride(1),
            M, N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        ctx.save_for_backward(gate_2d, up_2d)
        ctx.shape_info = (orig_shape, M, N)
        return out_2d.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad):
        gate_2d, up_2d = ctx.saved_tensors
        orig_shape, M, N = ctx.shape_info

        grad_2d = grad.to(gate_2d.dtype).reshape(-1, orig_shape[-1]).contiguous()
        dgate_2d = torch.empty_like(gate_2d)
        dup_2d = torch.empty_like(up_2d)

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        swiglu_bwd_kernel[grid](
            grad_2d, gate_2d, up_2d, dgate_2d, dup_2d,
            grad_2d.stride(0), grad_2d.stride(1),
            gate_2d.stride(0), gate_2d.stride(1),
            up_2d.stride(0), up_2d.stride(1),
            dgate_2d.stride(0), dgate_2d.stride(1),
            dup_2d.stride(0), dup_2d.stride(1),
            M, N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return dgate_2d.reshape(orig_shape), dup_2d.reshape(orig_shape)


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)


def patch_qwen2_mlp():
    """Send Qwen2MLP.forward through the fused Triton kernel."""
    from transformers.models.qwen2 import modeling_qwen2

    original = modeling_qwen2.Qwen2MLP.forward

    def forward(self, x):
        act = getattr(getattr(self, "config", None), "hidden_act", "silu")
        if act != "silu" or not (isinstance(x, torch.Tensor) and getattr(x, "is_cuda", False)):
            return original(self, x)
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

    modeling_qwen2.Qwen2MLP.forward = forward
