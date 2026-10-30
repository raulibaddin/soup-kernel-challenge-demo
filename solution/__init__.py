"""Your solution for the Soup Kernel Challenge — Decentrathon 6.0.

The harness imports this package and calls `apply()` once, in the same process as `soup train`,
before Soup builds the model and the trainer. Patch whatever you need (transformers, peft,
bitsandbytes, Soup itself) so the training run uses your Triton kernels.

Optimizations:
1. Fused RMSNorm Triton kernels (forward + backward) — replaces Qwen2RMSNorm
   - Eliminates repeated FP32 -> FP16 -> FP32 dtype casts
   - Fuses normalization + weight multiplication + variance computation into one kernel launch
   - Runs in every layer (57 RMSNorm instances per step)

2. Fused RoPE (Rotary Position Embedding) Triton kernels (forward + backward) — replaces apply_rotary_pos_emb
   - Fuses slice, rotate_half, cos/sin multiplication, and addition into a single kernel launch
   - Eliminates intermediate tensor allocations across all 28 attention heads (56 calls per step)

3. Fused SwiGLU (example, active) — silu(gate) * up
"""

from .swiglu import patch_qwen2_mlp
from .rms_norm import patch_qwen2_rmsnorm
from .rope import patch_qwen2_rope


def apply():
    patch_qwen2_mlp()       # example: fused SwiGLU
    patch_qwen2_rmsnorm()   # custom kernel: fused RMSNorm (fwd + bwd)
    patch_qwen2_rope()      # custom kernel: fused RoPE (fwd + bwd)
