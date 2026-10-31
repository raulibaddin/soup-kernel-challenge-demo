"""Custom Chunked Vocabulary Output and Loss Optimization for Qwen2.

Eliminates the ~4.5 GB peak memory spike during causal language model training.
In standard unchunked training:
1. lm_head(hidden_states) materializes a full [B, S, V] = [4, 512, 151936] FP16 logits tensor (593.5 MiB).
2. ForCausalLMLoss upcasts it to FP32 (1,187 MiB).
3. PyTorch cross_entropy saves FP32 log-softmax activations (1,187 MiB) and backward allocates
   two transient FP32 gradient buffers (2,374 MiB).
Total peak memory consumed by logits and loss exceeds 4.5 GB.

This implementation computes the output head projection and cross-entropy loss in token micro-chunks
(default size 128). In both forward and backward, each chunk's logits are immediately released from GPU
memory before the next chunk is processed. The full [B, S, V] matrix is never materialized.

All metrics required by Soup (entropy, accuracy) and per-row losses for rewind.jsonl are computed
online in streaming fashion, exactly matching PyTorch FP32 cross-entropy arithmetic.
"""

from dataclasses import dataclass
from typing import Any, Optional
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CustomChunkedMetrics:
    """Passive container for streaming token metrics.
    
    Compatible with Soup RewindTrainer's row_losses_from_token_losses and TRL metrics logger.
    """
    token_entropy: torch.Tensor   # [B, S] entropy per position
    token_loss: torch.Tensor      # [B, S] token NLL for per-row rewind logging
    correct: torch.Tensor         # Scalar count of argmax == label for labelled tokens
    labelled: torch.Tensor        # Scalar count of valid labelled positions


# Execution tracking for diagnostics and verification
CHUNKED_LOSS_CALL_COUNT = 0


class CustomChunkedLossFunction(torch.autograd.Function):
    """Custom autograd function computing causal LM loss in micro-chunks.
    
    Forward:
        Computes forward projection and loss in chunks of `chunk_size`.
        Discards chunk logits immediately.
        Saves only hidden_states, lm_head_weight, targets_1d, and denominator for backward.
        
    Backward:
        Recomputes one chunk of logits at a time.
        Computes dlogits = (softmax - one_hot) * scale in FP32.
        Backpropagates through frozen lm_head: dx = dlogits @ lm_head_weight.
        Accumulates into dx and frees chunk buffers immediately.
    """

    @staticmethod
    def forward(ctx, hidden_states_2d, lm_head_weight, targets_1d, chunk_size, denominator, ignore_index):
        global CHUNKED_LOSS_CALL_COUNT
        CHUNKED_LOSS_CALL_COUNT += 1

        M, H = hidden_states_2d.shape
        device = hidden_states_2d.device

        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        token_loss = torch.zeros(M, dtype=torch.float32, device=device)
        token_entropy = torch.zeros(M, dtype=torch.float32, device=device)
        total_correct = torch.zeros((), dtype=torch.long, device=device)

        # Chunked forward loop
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            chunk_x = hidden_states_2d[start:end]
            chunk_y = targets_1d[start:end]

            # 1. Forward linear projection for chunk: [C, V]
            chunk_logits = F.linear(chunk_x, lm_head_weight)

            # 2. Upcast to FP32 for loss & metrics matching PyTorch ForCausalLMLoss
            chunk_logits_f32 = chunk_logits.float()

            # 3. Log-softmax in FP32
            log_probs = F.log_softmax(chunk_logits_f32, dim=-1)

            # 4. Valid mask
            valid_mask = (chunk_y != ignore_index)
            safe_y = chunk_y.clamp(min=0)

            # 5. Token loss: -log_probs[target]
            token_nll = -log_probs.gather(dim=-1, index=safe_y.unsqueeze(-1)).squeeze(-1)
            token_nll = token_nll * valid_mask.float()
            token_loss[start:end] = token_nll
            total_loss = total_loss + token_nll.sum()

            # 6. Entropy: -sum(p * log_p)
            probs = torch.exp(log_probs)
            ent = -(probs * log_probs).sum(dim=-1)
            token_entropy[start:end] = ent

            # 7. Accuracy
            pred = chunk_logits.argmax(dim=-1)
            total_correct = total_correct + ((pred == chunk_y) & valid_mask).sum()

            del chunk_logits, chunk_logits_f32, log_probs, probs, ent, pred

        # Save ONLY hidden states, weights reference, and targets.
        # Zero chunk logits are saved in the autograd context!
        ctx.save_for_backward(hidden_states_2d, lm_head_weight, targets_1d)
        ctx.chunk_size = chunk_size
        ctx.denominator = denominator
        ctx.ignore_index = ignore_index

        normalized_loss = total_loss / denominator
        return normalized_loss, token_entropy, token_loss, total_correct

    @staticmethod
    def backward(ctx, grad_loss, grad_entropy, grad_token_loss, grad_correct):
        hidden_states_2d, lm_head_weight, targets_1d = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        denominator = ctx.denominator
        ignore_index = ctx.ignore_index

        M, H = hidden_states_2d.shape
        dx = torch.empty_like(hidden_states_2d)

        # Scale factor from incoming upstream gradient: grad_loss / denominator
        scale = (grad_loss / denominator).float()

        # Chunked backward loop
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            chunk_x = hidden_states_2d[start:end]
            chunk_y = targets_1d[start:end]

            # 1. Recompute chunk logits: [C, V]
            chunk_logits = F.linear(chunk_x, lm_head_weight)

            # 2. Softmax in FP32
            probs = F.softmax(chunk_logits.float(), dim=-1)

            # 3. In-place gradient: (probs - 1_y) * scale
            valid_mask = (chunk_y != ignore_index)
            if not valid_mask.any():
                dx[start:end] = 0.0
                del chunk_logits, probs
                continue

            safe_y = chunk_y.clamp(min=0)
            probs.scatter_add_(-1, safe_y.unsqueeze(-1), torch.full_like(safe_y.unsqueeze(-1), -1.0, dtype=torch.float32))
            probs = probs * valid_mask.unsqueeze(-1).float()

            chunk_dz = (probs * scale).to(chunk_x.dtype)

            # 4. Backprop through frozen lm_head: dx = dz @ W
            chunk_dx = torch.matmul(chunk_dz, lm_head_weight)
            dx[start:end] = chunk_dx

            del chunk_logits, probs, chunk_dz, chunk_dx

        return dx, None, None, None, None, None


def shift_for_causal_lm(labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Move labels one position left for causal LM: position t predicts token t+1.
    
    Position S-1 has no next token and is filled with ignore_index.
    """
    return F.pad(labels, (0, 1), value=ignore_index)[..., 1:]


def chunked_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 128,
    ignore_index: int = -100,
    num_items_in_batch: Optional[Any] = None,
) -> tuple[torch.Tensor, CustomChunkedMetrics]:
    """Compute causal LM loss and streaming metrics in micro-chunks without full logits.
    
    Args:
        hidden_states: [B, S, H] tensor from transformer decoder backbone.
        lm_head_weight: [V, H] weight matrix of the lm_head.
        labels: [B, S] ground truth token ids.
        chunk_size: number of tokens per micro-chunk (default: 128).
        ignore_index: token index to ignore (default: -100).
        num_items_in_batch: optional denominator override from HF Trainer.
    """
    orig_shape = hidden_states.shape
    B, S, H = orig_shape
    M = B * S

    # Causal label shifting
    targets = shift_for_causal_lm(labels, ignore_index=ignore_index).to(hidden_states.device)
    targets_1d = targets.reshape(-1)
    hidden_2d = hidden_states.reshape(-1, H)

    labelled = (targets_1d != ignore_index).sum()
    if num_items_in_batch is None:
        denominator = labelled.clamp(min=1).float()
    elif torch.is_tensor(num_items_in_batch):
        denominator = num_items_in_batch.to(hidden_states.device).clamp(min=1).float()
    else:
        denominator = max(float(num_items_in_batch), 1.0)

    loss, token_entropy_1d, token_loss_1d, correct = CustomChunkedLossFunction.apply(
        hidden_2d, lm_head_weight, targets_1d, chunk_size, denominator, ignore_index
    )

    metrics = CustomChunkedMetrics(
        token_entropy=token_entropy_1d.reshape(B, S),
        token_loss=token_loss_1d.reshape(B, S),
        correct=correct,
        labelled=labelled,
    )

    return loss, metrics


# Thread-local storage for latest custom chunked step metrics
_CURRENT_CHUNKED_STEP: Optional[CustomChunkedMetrics] = None


def get_and_clear_current_chunked_step() -> Optional[CustomChunkedMetrics]:
    global _CURRENT_CHUNKED_STEP
    step = _CURRENT_CHUNKED_STEP
    _CURRENT_CHUNKED_STEP = None
    return step


def patch_qwen2_chunked_loss():
    """Replace Qwen2ForCausalLM.forward with chunked loss execution during training."""
    from transformers.models.qwen2 import modeling_qwen2
    from transformers.modeling_outputs import CausalLMOutputWithPast

    original_forward = modeling_qwen2.Qwen2ForCausalLM.forward

    def chunked_forward(self, input_ids=None, attention_mask=None, position_ids=None,
                        past_key_values=None, inputs_embeds=None, labels=None,
                        use_cache=None, logits_to_keep=0, **kwargs):
        # Fall back to original forward for non-training, non-CUDA, or missing labels (e.g. generation/tests)
        if not self.training or labels is None:
            return original_forward(
                self, input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                logits_to_keep=logits_to_keep, **kwargs
            )

        # Execute transformer decoder backbone
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs
        )
        hidden_states = outputs.last_hidden_state

        # Guard: check if hidden_states is on CUDA
        if not getattr(hidden_states, "is_cuda", False):
            return original_forward(
                self, input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache,
                logits_to_keep=logits_to_keep, **kwargs
            )

        # Compute chunked causal LM loss without materializing full [B, S, V] logits
        num_items = kwargs.get("num_items_in_batch", None)
        loss, step_metrics = chunked_causal_lm_loss(
            hidden_states=hidden_states,
            lm_head_weight=self.lm_head.weight,
            labels=labels,
            chunk_size=128,
            num_items_in_batch=num_items,
        )

        global _CURRENT_CHUNKED_STEP
        _CURRENT_CHUNKED_STEP = step_metrics

        # Return logits=None to signal chunked loss to RewindTrainer and eliminate full logits retention
        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    modeling_qwen2.Qwen2ForCausalLM.forward = chunked_forward


def log_custom_trl_metrics(trainer: Any, mode: str, inputs: Any, step: CustomChunkedMetrics) -> None:
    """Log entropy, num_tokens and mean_token_accuracy to trainer._metrics matching TRL format."""
    metrics = getattr(trainer, "_metrics", None)
    gather = getattr(getattr(trainer, "accelerator", None), "gather_for_metrics", None)
    if metrics is None or gather is None:
        return

    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        entropy = (step.token_entropy * attention_mask).sum() / attention_mask.sum().clamp(min=1)
        tokens = gather(attention_mask.sum()).sum().item()
    else:
        entropy = step.token_entropy.mean()
        tokens = None

    metrics[mode]["entropy"].append(gather(entropy).mean().item())
    if mode == "train" and tokens is not None and hasattr(trainer, "_total_train_tokens"):
        trainer._total_train_tokens += tokens
    if hasattr(trainer, "_total_train_tokens"):
        metrics[mode]["num_tokens"] = [trainer._total_train_tokens]

    labelled = gather(step.labelled).sum()
    accuracy = (gather(step.correct).sum() / labelled).item() if labelled > 0 else 0.0
    metrics[mode]["mean_token_accuracy"].append(accuracy)


def patch_sft_trainer():
    """Wrap SFTTrainer.compute_loss to consume chunked step metrics and feed RewindTrainer."""
    try:
        import trl
        from transformers import Trainer
    except ImportError:
        return

    if not hasattr(trl, "SFTTrainer"):
        return

    original_compute_loss = trl.SFTTrainer.compute_loss

    def chunked_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if not getattr(self.model, "training", False) or "labels" not in inputs:
            return original_compute_loss(
                self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        mode = "train" if self.model.training else "eval"
        inputs.pop("_prediction_loss_only", None)
        inputs["use_cache"] = False

        # Delegate to base Trainer.compute_loss
        loss, outputs = Trainer.compute_loss(
            self, model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        step = get_and_clear_current_chunked_step()
        if step is not None:
            # Set on trainer for RewindTrainer's row_losses_from_token_losses
            setattr(self, "_soup_chunked_step", step)
            log_custom_trl_metrics(self, mode, inputs, step)

        return (loss, outputs) if return_outputs else loss

    trl.SFTTrainer.compute_loss = chunked_compute_loss

