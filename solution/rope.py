"""Custom RoPE (Rotary Position Embedding) Triton kernel for Qwen2 — forward and backward.

Fuses the rotate_half, cos/sin multiplication, and addition into a single kernel launch.
In standard PyTorch, apply_rotary_pos_emb executes multiple kernel launches per layer:
slice, cat, mul, add for q and for k (28 layers * 2 * ~5 ops = ~280 ops per step in fwd).
This Triton implementation replaces all of that with 1 kernel launch per tensor in fwd and bwd.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=["M", "S", "HS"])
def rope_fwd_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Out_ptr,
    M, S, HS,
    HAS_B_COS: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    """Forward pass of RoPE:
    y1 = x1 * cos1 - x2 * sin1
    y2 = x2 * cos2 + x1 * sin2
    
    One program per token head (row in X of length D).
    """
    row_x = tl.program_id(0)
    
    # Compute row in Cos/Sin
    s = row_x % S
    b = row_x // HS
    b_cos = b if HAS_B_COS else 0
    row_cos = b_cos * S + s
    
    # Pointer offsets
    x_first_ptr = X_ptr + row_x * D + tl.arange(0, HALF_D)
    x_second_ptr = X_ptr + row_x * D + HALF_D + tl.arange(0, HALF_D)
    
    cos_first_ptr = Cos_ptr + row_cos * D + tl.arange(0, HALF_D)
    cos_second_ptr = Cos_ptr + row_cos * D + HALF_D + tl.arange(0, HALF_D)
    
    sin_first_ptr = Sin_ptr + row_cos * D + tl.arange(0, HALF_D)
    sin_second_ptr = Sin_ptr + row_cos * D + HALF_D + tl.arange(0, HALF_D)
    
    out_first_ptr = Out_ptr + row_x * D + tl.arange(0, HALF_D)
    out_second_ptr = Out_ptr + row_x * D + HALF_D + tl.arange(0, HALF_D)
    
    # Load inputs in float32 for precision
    x1 = tl.load(x_first_ptr).to(tl.float32)
    x2 = tl.load(x_second_ptr).to(tl.float32)
    
    c1 = tl.load(cos_first_ptr).to(tl.float32)
    c2 = tl.load(cos_second_ptr).to(tl.float32)
    
    s1 = tl.load(sin_first_ptr).to(tl.float32)
    s2 = tl.load(sin_second_ptr).to(tl.float32)
    
    # RoPE transformation
    out1 = x1 * c1 - x2 * s1
    out2 = x2 * c2 + x1 * s2
    
    # Store results in original dtype
    tl.store(out_first_ptr, out1.to(Out_ptr.dtype.element_ty))
    tl.store(out_second_ptr, out2.to(Out_ptr.dtype.element_ty))


@triton.jit(do_not_specialize_on_alignment=["M", "S", "HS"])
def rope_bwd_kernel(
    DOut_ptr, Cos_ptr, Sin_ptr, DX_ptr,
    M, S, HS,
    HAS_B_COS: tl.constexpr,
    D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    """Backward pass of RoPE:
    dx1 = dout1 * cos1 + dout2 * sin2
    dx2 = dout2 * cos2 - dout1 * sin1
    
    One program per token head (row in DOut of length D).
    """
    row_x = tl.program_id(0)
    
    # Compute row in Cos/Sin
    s = row_x % S
    b = row_x // HS
    b_cos = b if HAS_B_COS else 0
    row_cos = b_cos * S + s
    
    # Pointer offsets
    dout_first_ptr = DOut_ptr + row_x * D + tl.arange(0, HALF_D)
    dout_second_ptr = DOut_ptr + row_x * D + HALF_D + tl.arange(0, HALF_D)
    
    cos_first_ptr = Cos_ptr + row_cos * D + tl.arange(0, HALF_D)
    cos_second_ptr = Cos_ptr + row_cos * D + HALF_D + tl.arange(0, HALF_D)
    
    sin_first_ptr = Sin_ptr + row_cos * D + tl.arange(0, HALF_D)
    sin_second_ptr = Sin_ptr + row_cos * D + HALF_D + tl.arange(0, HALF_D)
    
    dx_first_ptr = DX_ptr + row_x * D + tl.arange(0, HALF_D)
    dx_second_ptr = DX_ptr + row_x * D + HALF_D + tl.arange(0, HALF_D)
    
    # Load gradients and weights in float32
    dout1 = tl.load(dout_first_ptr).to(tl.float32)
    dout2 = tl.load(dout_second_ptr).to(tl.float32)
    
    c1 = tl.load(cos_first_ptr).to(tl.float32)
    c2 = tl.load(cos_second_ptr).to(tl.float32)
    
    s1 = tl.load(sin_first_ptr).to(tl.float32)
    s2 = tl.load(sin_second_ptr).to(tl.float32)
    
    # Transposed RoPE transformation
    dx1 = dout1 * c1 + dout2 * s2
    dx2 = dout2 * c2 - dout1 * s1
    
    # Store dx in original dtype
    tl.store(dx_first_ptr, dx1.to(DX_ptr.dtype.element_ty))
    tl.store(dx_second_ptr, dx2.to(DX_ptr.dtype.element_ty))


class TritonRoPEFunction(torch.autograd.Function):
    """Autograd Function wrapping fused RoPE forward and backward."""
    
    @staticmethod
    def forward(ctx, x, cos, sin):
        # x: [B, H, S, D]
        orig_shape = x.shape
        B, H, S, D = orig_shape
        HALF_D = D // 2
        
        # Ensure contiguous x
        x_contig = x.contiguous()
        
        # Flatten x to [M, D] where M = B * H * S
        M = B * H * S
        HS = H * S
        
        # Standardize cos, sin shapes: [B_cos, S, D]
        # In transformers, cos is either [B, S, D] or [1, S, D] or [B, 1, S, D] or [S, D]
        if cos.ndim == 4:
            cos_2d = cos.squeeze(1)
            sin_2d = sin.squeeze(1)
        elif cos.ndim == 2:
            cos_2d = cos.unsqueeze(0)
            sin_2d = sin.unsqueeze(0)
        else:
            cos_2d = cos
            sin_2d = sin
        
        B_cos = cos_2d.shape[0]
        HAS_B_COS = (B_cos > 1)
        
        cos_contig = cos_2d.contiguous()
        sin_contig = sin_2d.contiguous()
        
        out = torch.empty_like(x_contig)
        
        grid = (M,)
        rope_fwd_kernel[grid](
            x_contig, cos_contig, sin_contig, out,
            M, S, HS,
            HAS_B_COS=HAS_B_COS,
            D=D,
            HALF_D=HALF_D,
        )
        
        ctx.save_for_backward(cos_contig, sin_contig)
        ctx.shape_info = (orig_shape, S, HS, HAS_B_COS, D, HALF_D, M)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        cos_contig, sin_contig = ctx.saved_tensors
        orig_shape, S, HS, HAS_B_COS, D, HALF_D, M = ctx.shape_info
        
        grad_contig = grad_output.contiguous()
        dx = torch.empty_like(grad_contig)
        
        grid = (M,)
        rope_bwd_kernel[grid](
            grad_contig, cos_contig, sin_contig, dx,
            M, S, HS,
            HAS_B_COS=HAS_B_COS,
            D=D,
            HALF_D=HALF_D,
        )
        
        return dx, None, None


def apply_triton_rope(x, cos, sin):
    """Apply RoPE using Triton kernel."""
    return TritonRoPEFunction.apply(x, cos, sin)


def patch_qwen2_rope():
    """Replace transformers.models.qwen2.modeling_qwen2.apply_rotary_pos_emb with Triton version."""
    from transformers.models.qwen2 import modeling_qwen2
    
    original_apply = modeling_qwen2.apply_rotary_pos_emb
    
    def fused_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        # Guard: check if inputs are CUDA tensors
        if not (isinstance(q, torch.Tensor) and getattr(q, "is_cuda", False)):
            return original_apply(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
        
        # Apply fused Triton RoPE for q and k
        q_out = apply_triton_rope(q, cos, sin)
        k_out = apply_triton_rope(k, cos, sin)
        return q_out, k_out
    
    modeling_qwen2.apply_rotary_pos_emb = fused_apply_rotary_pos_emb
