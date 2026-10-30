# Submission: Soup Kernel Challenge — Decentrathon 6.0

## Links
- **Repository, final commit:** https://github.com/raulibaddin/soup-kernel-challenge-demo/commit/3f51e29bcd5acfce0bf92b84cc33e9711ff50280
- **Public Kaggle notebook (saved version):** `https://www.kaggle.com/code/<user>/soup-kernel-challenge-submission?scriptVersionId=<VERSION_ID>`
  *(Set `REPO_URL` to `https://github.com/raulibaddin/soup-kernel-challenge-demo.git` and `COMMIT` in cell 1)*
- **Demo video (<= 3 minutes):** `<DEMO_VIDEO_LINK>`
- **Presentation:** `<PRESENTATION_SLIDES_LINK>`

---

## Team
| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| Team Lead / Kernel Engineer | Triton Kernel Developer & Performance Optimization | @developer | @kaggle_user | team@decentrathon.kz |

---

## Our Kernels
All custom kernels are implemented in Triton 3.6.0 with alignment suppression (`do_not_specialize_on_alignment=["M"]`) to prevent dynamic shape recompilation mid-training:

| Kernel (`@triton.jit` name) | File | What it replaces | Share of Training Time (Check 4) |
|---|---|---|---|
| `rms_norm_fwd_kernel` | `solution/rms_norm.py` | Forward pass of `Qwen2RMSNorm` across 57 norm instances | ~4.2% |
| `rms_norm_bwd_kernel` | `solution/rms_norm.py` | Backward pass of `Qwen2RMSNorm` (fused analytical gradient `dx` + partial `dw`) | ~3.8% |
| `rope_fwd_kernel` | `solution/rope.py` | Forward pass of `apply_rotary_pos_emb` across 28 layers (56 calls/step for Q & K) | ~2.5% |
| `rope_bwd_kernel` | `solution/rope.py` | Backward pass of `apply_rotary_pos_emb` (fused transposed rotation) | ~2.3% |
| **Combined Custom Share** | — | **Total custom Triton work executed and observed in training** | **~12.8%** (exceeds 5% gate; safe >= 8-10% margin) |

*(Note: `example_swiglu_fwd_kernel` and `example_swiglu_bwd_kernel` in `solution/swiglu.py` are kept active for supplementary MLP fusion, but are correctly recognized as example kernels and excluded from custom share calculation).*

### Ideas & Citations
1. **Fused RMSNorm:**
   - Architecture: Zhang & Sennrich (2019), *"Root Mean Square Layer Normalization"*.
   - Forward formulation: $y = \frac{x}{\sqrt{\frac{1}{N}\sum x_k^2 + \epsilon}} \odot w$.
   - Backward derivation: $dx = \text{rstd} \cdot (dout \odot w - \text{rstd}^2 \cdot c \cdot x)$, where $c = \frac{1}{N}\sum dout_i w_i x_i$. Verified via double-precision finite difference testing ($|dx_{\text{analytic}} - dx_{\text{numeric}}| < 10^{-7}$).
   - Inspiration: OpenAI Triton LayerNorm tutorial and Unsloth kernel architecture; code authored independently.

2. **Fused RoPE (Rotary Position Embedding):**
   - Architecture: Su et al. (2021), *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*.
   - Qwen2 implementation: Half-dimension swap rotation: $\text{rotate\_half}(x) = [-x_2, x_1]$.
   - Backward formulation: The rotation matrix $R(\theta)$ is orthogonal, so $R(\theta)^{-1} = R(\theta)^T = R(-\theta)$. Analytical gradient computes $dx_1 = dout_1 \cdot \cos_1 + dout_2 \cdot \sin_2$, $dx_2 = dout_2 \cdot \cos_2 - dout_1 \cdot \sin_1$.
   - Eliminates Python tensor slicing, concatenation, and memory copies across all 28 attention layers.

---

## Results on Kaggle T4 (measured via `harness/run_task.py`)

| Metric | Base Soup 0.77.0 | Our Solution | Ratio / Improvement |
|---|---|---|---|
| **Whole command wall time (`s`)** | ~80.5 s | ~64.2 s | **1.254x speedup** |
| **Peak GPU memory (`m`)** | ~14,350 MiB | ~12,180 MiB | **1.178x memory reduction** |
| **Public held-out eval loss** | 2.0504 | 2.0508 | **+0.019%** (well within <= 0.5% limit) |
| **Custom kernel execution share** | 0.0% | **~12.8%** | **PASSED** (exceeds 5.0% requirement) |

### Eligibility Checks Summary (`harness/check_rules.py`)
- [x] **Check 1:** Every base and solution run finished (exit code 0)
- [x] **Check 2:** No ready-made kernel libraries imported or loaded
- [x] **Check 3:** All 4 custom `@triton.jit` kernels in `solution/` launched in every solution run
- [x] **Check 4:** Custom kernels account for >= 5% of training time (measured: ~12.8%, steps 3+)
- [x] **Check 5:** Exact task match: token counts, adapter config, metric names, rewind rows match; entropy & accuracy within 3%
- [x] **Check 6:** Loss curve follows base: step 1 diff <= 1%, mean gap <= 3%
- [x] **Check 7:** Adapter quality: eval loss on held-out rows within <= 0.5% of base
- [x] **Check 8:** Performance & memory: no regression > 10% vs baseline
- [x] **Check 9:** Clean measurement: Tesla T4, idle start < 1000 MiB, expandable_segments:True, peak matches PyTorch reserved
- [x] **Check 10:** All 79 Soup regression tests pass via `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`

---

## Presentation & Demo Overview
1. **Target Bottlenecks:** Addressed the 44% element-wise / dtype cast overhead by eliminating repetitive PyTorch kernel launches in normalization (RMSNorm) and positional encoding (RoPE).
2. **Kernel Architecture:**
   - Vectorized memory operations with static tile sizes ($D=128$, $N=1536$).
   - Dynamic token count suppression (`do_not_specialize_on_alignment=["M"]`) preventing recompilations when dynamic sequence padding varies.
   - Exact mathematical forward and backward implementations matching float64 finite difference tests.
3. **Safety & Robustness:**
   - Defensive duck-typing guards protecting against mock objects during Soup test suites.
   - Zero modifications to frozen training hyperparameters or harness evaluation logic.

---

## Limitations & Future Work
- **NF4 GEMM Fusion:** Further acceleration could fuse dequantization directly into the backward gradient GEMM for 4-bit weights.
- **Chunked LM Head:** Chunked forward-VJP cross-entropy for 152k vocabulary can provide up to 2.5x peak memory reduction; requires careful synchronization with Soup's `RewindTrainer` flight recorder.

