"""
Off-Policy Monitoring for Replay Buffer Training

Computes importance weight statistics and KL divergence between current policy
and behavior policy for off-policy diagnostics.
"""

import torch
from typing import Dict, Optional, Any
from roll.distributed.scheduler.protocol import DataProto
from roll.utils.logging import get_logger

logger = get_logger()


def compute_offpolicy_metrics(
    current_batch: DataProto,
    training_metrics: Optional[DataProto] = None,
    actor_train_cluster: Any = None,
    pg_clip: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute off-policy metrics comparing current policy with behavior policy.

    Uses 'old_log_probs' (fresh batch) or 'behavior_log_probs' (replay batch) as behavior baseline.
    Can reuse log_probs from training_metrics to avoid redundant forward pass.

    Args:
        current_batch: DataProto with behavior log probs and response_mask
        training_metrics: Optional DataProto from train_step with pre-computed log_probs
        actor_train_cluster: Actor cluster for computing log probs (fallback if no training_metrics)
        pg_clip: PPO clip range for clipping analysis

    Returns:
        Dictionary of metrics with 'offpolicy/' prefix
    """
    metrics: Dict[str, float] = {}

    try:
        if current_batch is None or current_batch.batch is None:
            return metrics

        # Find behavior log probs field
        if "old_log_probs" in current_batch.batch:
            behavior_field = "old_log_probs"
        elif "behavior_log_probs" in current_batch.batch:
            behavior_field = "behavior_log_probs"
        else:
            return metrics

        if "response_mask" not in current_batch.batch:
            return metrics

        # Get current policy log probs (reuse from training if available)
        current_log_probs = None
        if training_metrics is not None and training_metrics.batch is not None:
            if "log_probs" in training_metrics.batch:
                current_log_probs = training_metrics.batch["log_probs"]

        if current_log_probs is None:
            if actor_train_cluster is None:
                return metrics
            import ray
            current_lp_refs = actor_train_cluster.compute_log_probs(current_batch, blocking=False)
            current_lp_data = DataProto.materialize_concat(data_refs=current_lp_refs)
            if "log_probs" not in current_lp_data.batch:
                return metrics
            current_log_probs = current_lp_data.batch["log_probs"]

        behavior_log_probs = current_batch.batch[behavior_field]
        response_mask = current_batch.batch["response_mask"][:, 1:].bool()

        # Align shapes
        min_len = min(current_log_probs.shape[1], behavior_log_probs.shape[1], response_mask.shape[1])
        current_log_probs = current_log_probs[:, :min_len]
        behavior_log_probs = behavior_log_probs[:, :min_len]
        response_mask = response_mask[:, :min_len]

        valid_current = current_log_probs[response_mask]
        valid_behavior = behavior_log_probs[response_mask]

        if valid_current.numel() == 0:
            return metrics

        # Token-level importance weights
        log_ratio = valid_current - valid_behavior
        ratio = log_ratio.exp()

        metrics["offpolicy/importance_weight/mean"] = ratio.mean().detach().item()
        metrics["offpolicy/importance_weight/std"] = ratio.std().detach().item()
        metrics["offpolicy/importance_weight/max"] = ratio.max().detach().item()
        metrics["offpolicy/importance_weight/min"] = ratio.min().detach().item()

        # Sample-level statistics (geometric mean per sample)
        batch_size, seq_len = response_mask.shape
        full_log_ratio = torch.zeros(batch_size, seq_len, device=log_ratio.device, dtype=log_ratio.dtype)
        full_log_ratio[response_mask] = log_ratio
        valid_tokens_per_sample = response_mask.sum(dim=1).clamp(min=1)
        sample_ratio = torch.exp((full_log_ratio * response_mask).sum(dim=1) / valid_tokens_per_sample)

        metrics["offpolicy/sample_importance_weight/mean"] = sample_ratio.mean().detach().item()
        metrics["offpolicy/sample_importance_weight/max"] = sample_ratio.max().detach().item()
        metrics["offpolicy/sample_importance_weight/min"] = sample_ratio.min().detach().item()

        # Fraction statistics
        metrics["offpolicy/fraction_near_one"] = ((ratio >= 0.9) & (ratio <= 1.1)).float().mean().detach().item()
        metrics["offpolicy/fraction_below_half"] = (ratio < 0.5).float().mean().detach().item()
        metrics["offpolicy/fraction_above_double"] = (ratio > 2.0).float().mean().detach().item()

        # PPO clip analysis
        if pg_clip is not None and pg_clip > 0:
            in_clip = (ratio >= 1 - pg_clip) & (ratio <= 1 + pg_clip)
            metrics["offpolicy/fraction_in_ppo_clip_range"] = in_clip.float().mean().detach().item()

        # ESS and KL
        ess = (ratio.sum() ** 2) / (ratio ** 2).sum()
        metrics["offpolicy/effective_sample_size_ratio"] = (ess / ratio.numel()).detach().item()
        metrics["offpolicy/approx_kl_divergence"] = log_ratio.mean().detach().item()

        # Token stats
        metrics["offpolicy/valid_token_count"] = float(valid_current.numel())
        metrics["offpolicy/reused_log_probs"] = 1.0 if training_metrics is not None else 0.0

    except Exception as e:
        logger.error(f"Failed to compute off-policy metrics: {e}", exc_info=True)
        metrics["offpolicy/error"] = 1.0

    return metrics
