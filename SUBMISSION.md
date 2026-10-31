# Submission: Soup Kernel Challenge — Decentrathon 6.0

## Links
- **Repository, final commit:** `https://github.com/raulibaddin/soup-kernel-challenge-demo/commit/<SOURCE_COMMIT_SHA>`
  *(Must be set to the exact source commit SHA of the candidate branch before submission)*
- **Public Kaggle notebook (saved version):** `[INCOMPLETE - RUN KAGGLE NOTEBOOK AND INSERT SAVED VERSION LINK HERE]`
  *(Set `REPO_URL = "https://github.com/raulibaddin/soup-kernel-challenge-demo.git"` and `COMMIT = "<SOURCE_COMMIT_SHA>"` in cell 1)*
- **Demo video (<= 3 minutes):** `[INCOMPLETE - INSERT DEMO VIDEO LINK HERE BEFORE SUBMISSION]`
- **Presentation:** `[INCOMPLETE - INSERT PRESENTATION SLIDES LINK HERE BEFORE SUBMISSION]`

> [!IMPORTANT]
> **Two-Commit Pinning Workflow:**
> - An empty `COMMIT = ""` is permitted strictly during iterative development.
> - Before official validation, `COMMIT` must be set to the exact source commit SHA of the candidate branch to ensure immutable evaluation rather than silently pulling from `main`.
> - The source implementation is committed first; that resulting commit SHA is then pinned in the notebook in a subsequent commit.

---

## Team
> [!IMPORTANT]
> **Action Required:** Team members must replace the placeholder fields below with real names, handles, and captain contact before final submission. This table accommodates individual or multi-member teams (add or remove rows as needed).

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| `[Member 1 / Team Lead]` | `[Role / Contribution]` | `@[github_user]` | `@[kaggle_user]` | `[email@domain.com]` |
| `[Member 2 (if applicable)]` | `[Role / Contribution]` | `@[github_user]` | `@[kaggle_user]` | `[contact]` |

---

## Our Kernels
All custom kernels are implemented in Triton 3.6.0 with alignment suppression (`do_not_specialize_on_alignment=["M"]` and `["M", "S", "HS"]`) to suppress specified alignment-based specialization (runtime compilation reduction has not yet been measured on hardware):

| Kernel (`@triton.jit` name) | File | What it replaces | Training Time Share (Check 4) |
|---|---|---|---|
| `rms_norm_fwd_kernel` | `solution/rms_norm.py` | Forward pass of `Qwen2RMSNorm` across 57 norm instances | Pending official measurement |
| `rms_norm_bwd_kernel` | `solution/rms_norm.py` | Backward pass of `Qwen2RMSNorm` (fused analytical gradient `dx` + partial `dw`, skipped when frozen) | Pending official measurement |
| `rope_fwd_kernel` | `solution/rope.py` | Forward pass of `apply_rotary_pos_emb` across 28 layers (56 calls/step for Q & K) | Pending official measurement |
| `rope_bwd_kernel` | `solution/rope.py` | Backward pass of `apply_rotary_pos_emb` (fused transposed rotation) | Pending official measurement |
| **Combined Custom Share** | — | **Total custom Triton work executed and timed across steps 3+** | **Pending official measurement** (Requirement: $\ge 5.00\%$, safety target $\ge 8\text{--}10\%$) |

*(Note: `example_swiglu_fwd_kernel` and `example_swiglu_bwd_kernel` in `solution/swiglu.py` are kept active for supplementary MLP fusion, but are correctly recognized by AST as example copies and excluded from custom share calculation).*

### Ideas & Citations
1. **Fused RMSNorm:**
   - Architecture: Zhang & Sennrich (2019), *"Root Mean Square Layer Normalization"*.
   - Forward formulation: $y = \frac{x}{\sqrt{\frac{1}{N}\sum x_k^2 + \epsilon}} \odot w$.
   - Backward derivation: $dx = \text{rstd} \cdot (dout \odot w - \text{rstd}^2 \cdot c \cdot x)$, where $c = \frac{1}{N}\sum dout_i w_i x_i$. Verified via double-precision finite difference testing ($|dx_{\text{analytic}} - dx_{\text{numeric}}| < 10^{-7}$).
   - Memory optimization: When `ctx.needs_input_grad[1]` is False (standard LoRA where norm weights are frozen), partial weight-gradient allocation and kernel writes are completely bypassed (`HAS_DW = False`).
   - Inspiration: OpenAI Triton LayerNorm tutorial; code authored independently.

2. **Fused RoPE (Rotary Position Embedding):**
   - Architecture: Su et al. (2021), *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*.
   - Qwen2 implementation: Half-dimension swap rotation: $\text{rotate\_half}(x) = [-x_2, x_1]$.
   - Backward formulation: The rotation matrix $R(\theta)$ is orthogonal, so $R(\theta)^{-1} = R(\theta)^T = R(-\theta)$. Analytical gradient computes $dx_1 = dout_1 \cdot \cos_1 + dout_2 \cdot \sin_2$, $dx_2 = dout_2 \cdot \cos_2 - dout_1 \cdot \sin_1$.
   - Eliminates Python tensor slicing, concatenation, and memory copies across all 28 attention layers.

### Verification Status
- **Mathematical Formulations:** Forward and analytical gradient formulas for RMSNorm ($dx, dw$) and RoPE ($dx_1, dx_2$) were checked and verified offline against double-precision PyTorch numerical references in CPU scratch simulations ($|dx_{\text{analytic}} - dx_{\text{numeric}}| < 10^{-7}$).
- **Triton GPU Kernel Execution:** The custom Triton GPU implementations (`solution/rms_norm.py`, `solution/rope.py`, and `solution/test_kernels.py`) have **not yet been executed on GPU hardware** (due to local Windows environment constraints) and have not yet passed runtime tests. Actual GPU execution will be validated during the official Kaggle Tesla T4 run.

---

## Target Objectives on Kaggle T4 (to be measured via `harness/run_task.py --repeats 2`)

| Metric / Check | Mandatory Eligibility Limit (Check 8 / Harness) | Competitive Scoring Threshold (`score.py`) | Solution Design Target | Status |
|---|---|---|---|---|
| **Whole command wall time (`s`)** | $\le 1.10\times$ base time (Check 8: max 10% slowdown) | $s \ge 1.05\times$ (+5.0% min gain to earn speed points) | Target $s \ge 1.15\text{--}1.25\times$ speedup | Pending Kaggle T4 Run |
| **Peak GPU memory (`m`)** | $\le 1.10\times$ base peak (Check 8: max 10% growth) | $m \ge 1.05\times$ (+5.0% min gain to earn memory points) | Target $m \ge 1.10\text{--}1.18\times$ reduction | Pending Kaggle T4 Run |
| **Public held-out eval loss** | $\le +0.50\%$ above base (Check 7) | Required gate for admission | Target $\le +0.10\%$ delta | Pending Kaggle T4 Run |
| **Custom kernel execution share** | $\ge 5.00\%$ of training time (Check 4) | Required gate for admission | Safety target $\ge 8\text{--}10\%$ | Pending Official Measurement |

### Eligibility Checks Status (`harness/check_rules.py`)

No checker has executed yet. All 10 mandatory checks are **pending official execution** on Kaggle:

**Static Source & AST Pre-Flight Audit (Local Inspection):**
- [ ] **Check 2 (Source):** Zero banned ready-made kernel libraries imported in `solution/` (AST scan verified 0 banned imports; runtime check pending T4 run).
- [ ] **Check 3 (Source):** Exactly 4 own `@triton.jit` kernels discovered in `solution/`, all 4 hooked into `apply()` (AST scan verified; runtime execution pending T4 run).

**Official Kaggle T4 Execution (To be established on Kaggle T4 Run All):**
- [ ] **Check 1:** Every base and solution run finished (exit code 0 across `--repeats 2`)
- [ ] **Check 2 (Runtime):** No banned kernel libraries loaded during training
- [ ] **Check 3 (Runtime):** All 4 custom `@triton.jit` kernels launched in training
- [ ] **Check 4:** Custom kernels account for $\ge 5\%$ of training time (measured from steps 3+)
- [ ] **Check 5:** Exact task match: token counts, adapter config, metric names, rewind rows match; entropy & accuracy within 3%
- [ ] **Check 6:** Loss curve follows base: step 1 diff $\le 1\%$, mean gap $\le 3\%$
- [ ] **Check 7:** Adapter quality: eval loss on held-out rows within $\le 0.5\%$ of base
- [ ] **Check 8:** Performance & memory: no regression $> 10\%$ vs baseline (wall time $\le 1.10\times$, memory $\le 1.10\times$)
- [ ] **Check 9:** Clean measurement: Tesla T4, idle start $< 1000$ MiB, expandable segments allocator, peak matches PyTorch reserved
- [ ] **Check 10:** All 79 Soup regression tests pass via `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q`

---

## Presentation & Demo Overview
1. **Target Bottlenecks:** Addressed the 44% element-wise / dtype cast overhead by eliminating repetitive PyTorch kernel launches in normalization (RMSNorm) and positional encoding (RoPE).
2. **Kernel Architecture:**
   - Vectorized memory operations with static tile sizes ($D=128$, $N=1536$).
   - Dynamic token count and sequence length alignment suppression (`do_not_specialize_on_alignment=["M", "S", "HS"]`) to suppress specified alignment-based specialization (runtime compilation reduction has not yet been measured on hardware).
   - Exact mathematical forward and backward implementations matching float64 finite difference tests.
3. **Safety & Robustness:**
   - Defensive duck-typing guards protecting against mock objects during Soup test suites.
   - Zero modifications to frozen training hyperparameters or harness evaluation logic.

---

## Limitations & Future Work
- **NF4 GEMM Fusion:** Further acceleration could fuse dequantization directly into the backward gradient GEMM for 4-bit weights.
- **Chunked LM Head:** Chunked forward-VJP cross-entropy for 152k vocabulary can provide up to 2.5x peak memory reduction; requires careful synchronization with Soup's `RewindTrainer` flight recorder.

