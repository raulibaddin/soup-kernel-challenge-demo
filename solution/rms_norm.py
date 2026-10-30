"""Custom RMSNorm Triton kernel for Qwen2 — forward and backward.

Targets the ~44% of GPU time spent on element-wise operations and dtype casts.
The pinned Qwen2 RMSNorm computes in FP32 due to prepare_model_for_kbit_training
casting norm weights to FP32, causing FP32->FP16->FP32 cascades through Linear4bit.

This kernel fuses the normalization + weight multiply + optional dtype cast in one launch.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=["M"])
def rms_norm_fwd_kernel(
    X_ptr, W_ptr, Out_ptr, Rstd_ptr,
    M, N: tl.constexpr, eps,
    stride_x_row, stride_out_row,
    BLOCK_N: tl.constexpr,
):
    """Forward pass of RMSNorm: out = x / rms(x) * weight.
    
    One program per token (row). Computes variance in FP32 for numerical stability.
    M = total tokens (batch * seq_len), N = hidden dimension.
    """
    row = tl.program_id(0)
    
    x_row_ptr = X_ptr + row * stride_x_row
    out_row_ptr = Out_ptr + row * stride_out_row
    
    # Accumulate variance in FP32
    var_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        var_acc += x * x
    
    var = tl.sum(var_acc, axis=0) * (1.0 / N)
    rstd = tl.rsqrt(var + eps)
    
    # Store rstd for backward
    tl.store(Rstd_ptr + row, rstd)
    
    # Normalize and multiply by weight
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = x * rstd * w
        tl.store(out_row_ptr + cols, out.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit(do_not_specialize_on_alignment=["M"])
def rms_norm_bwd_kernel(
    DOut_ptr, X_ptr, W_ptr, Rstd_ptr,
    DX_ptr, DW_partial_ptr,
    M, N: tl.constexpr, 
    stride_dout_row, stride_x_row, stride_dx_row,
    stride_dw_partial_row,
    BLOCK_N: tl.constexpr,
):
    """Backward pass of RMSNorm.
    
    Computes dx and partial dw for one row. dw is accumulated across rows later.
    
    dx = rstd * (dout * w - x * rstd^2 * mean(dout * w * x) / N) ... wait, let me derive properly.
    
    Forward: y = x * rstd * w, where rstd = 1/sqrt(mean(x^2) + eps)
    
    dy/dx: Let s = mean(x^2) + eps, rstd = s^{-1/2}
    dy_i/dx_j = w_i * d(x_i * rstd)/dx_j
              = w_i * [rstd * delta_ij + x_i * d(rstd)/dx_j]
    d(rstd)/dx_j = -0.5 * s^{-3/2} * 2*x_j/N = -rstd^3 * x_j / N
    
    So: dy_i/dx_j = w_i * rstd * delta_ij - w_i * x_i * rstd^3 * x_j / N
    
    dx_j = sum_i (dout_i * dy_i/dx_j)
         = dout_j * w_j * rstd - (rstd^3 / N) * x_j * sum_i(dout_i * w_i * x_i)
    
    Let c = sum_i(dout_i * w_i * x_i) / N
    dx_j = rstd * (dout_j * w_j - rstd^2 * c * x_j)
    
    dw_i = sum_rows(dout_i * x_i * rstd)  (accumulated across rows)
    """
    row = tl.program_id(0)
    
    dout_row_ptr = DOut_ptr + row * stride_dout_row
    x_row_ptr = X_ptr + row * stride_x_row
    dx_row_ptr = DX_ptr + row * stride_dx_row
    
    rstd = tl.load(Rstd_ptr + row)
    
    # First pass: compute c = sum(dout * w * x) / N
    c_acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        dout = tl.load(dout_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        c_acc += dout * w * x
    
    c = tl.sum(c_acc, axis=0) * (1.0 / N)
    
    # Second pass: compute dx and partial dw
    dw_row_ptr = DW_partial_ptr + row * stride_dw_partial_row
    
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        dout = tl.load(dout_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        # dx = rstd * (dout * w - rstd^2 * c * x)
        dx = rstd * (dout * w - rstd * rstd * c * x)
        tl.store(dx_row_ptr + cols, dx.to(DX_ptr.dtype.element_ty), mask=mask)
        
        # partial dw for this row
        dw = dout * x * rstd
        tl.store(dw_row_ptr + cols, dw.to(DW_partial_ptr.dtype.element_ty), mask=mask)


class TritonRMSNorm(torch.autograd.Function):
    """RMSNorm with Triton kernels for forward and backward."""
    
    @staticmethod
    def forward(ctx, x, weight, eps):
        # x shape: [*, N] where N is the hidden dim
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, N = x_2d.shape
        
        # Determine output dtype: match what the original RMSNorm produces
        # Original: weight (FP32) * normalized_x -> FP32 output (due to FP32 weight)
        # We preserve this behavior
        out_dtype = torch.float32 if weight.dtype == torch.float32 else x.dtype
        
        out = torch.empty(M, N, device=x.device, dtype=out_dtype)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        
        # Choose BLOCK_N: must be power of 2, >= N for single-pass or loop
        BLOCK_N = min(triton.next_power_of_2(N), 4096)
        
        grid = (M,)
        rms_norm_fwd_kernel[grid](
            x_2d, weight, out, rstd,
            M, N, eps,
            x_2d.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
        )
        
        ctx.save_for_backward(x_2d, weight, rstd)
        ctx.orig_shape = orig_shape
        ctx.BLOCK_N = BLOCK_N
        ctx.N = N
        ctx.eps = eps
        
        return out.reshape(orig_shape)
    
    @staticmethod
    def backward(ctx, grad_output):
        x_2d, weight, rstd = ctx.saved_tensors
        N = ctx.N
        BLOCK_N = ctx.BLOCK_N
        
        grad_2d = grad_output.reshape(-1, N)
        M = grad_2d.shape[0]
        
        dx = torch.empty_like(x_2d)
        # Partial dw: one row per token, will sum across M
        dw_partial = torch.empty(M, N, device=x_2d.device, dtype=torch.float32)
        
        grid = (M,)
        rms_norm_bwd_kernel[grid](
            grad_2d, x_2d, weight, rstd,
            dx, dw_partial,
            M, N,
            grad_2d.stride(0), x_2d.stride(0), dx.stride(0),
            dw_partial.stride(0),
            BLOCK_N=BLOCK_N,
        )
        
        # Sum partial dw across all tokens only if weight requires grad
        dw = dw_partial.sum(dim=0) if ctx.needs_input_grad[1] else None
        
        return dx.reshape(ctx.orig_shape), dw, None


def triton_rms_norm(x, weight, eps=1e-6):
    """Drop-in replacement for RMSNorm forward."""
    return TritonRMSNorm.apply(x, weight, eps)


def patch_qwen2_rmsnorm():
    """Replace Qwen2RMSNorm.forward with our Triton implementation."""
    from transformers.models.qwen2 import modeling_qwen2
    
    original_forward = modeling_qwen2.Qwen2RMSNorm.forward
    
    def new_forward(self, hidden_states):
        # Guard: only use Triton on CUDA tensors and real tensors
        if not (isinstance(hidden_states, torch.Tensor) and getattr(hidden_states, "is_cuda", False)):
            return original_forward(self, hidden_states)
        return triton_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    
    modeling_qwen2.Qwen2RMSNorm.forward = new_forward

