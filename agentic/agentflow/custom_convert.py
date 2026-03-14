"""
AgentFlow multi-turn custom convert.

Splits each multi-turn trajectory into independent training samples (one per turn),
with all turns from the same trajectory sharing the same normalized advantage.

Usage:
    --custom-convert-samples-to-train-data-path agentic/agentflow/custom_convert.py:custom_convert
"""

import logging

import torch

logger = logging.getLogger(__name__)


def custom_convert(args, samples):
    # ── 1. Trajectory-level GRPO reward normalization ──
    raw_rewards = [s.get_reward_value(args) for s in samples]
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
    ):
        n = getattr(args, "n_samples_per_prompt", 1)
        if rewards_tensor.shape[-1] == n * getattr(args, "rollout_batch_size", 1):
            rewards_tensor = rewards_tensor.reshape(-1, n)
        else:
            rewards_tensor = rewards_tensor.view(-1, rewards_tensor.shape[-1])

        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        rewards_tensor = rewards_tensor - mean

        if (
            getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
            and getattr(args, "grpo_std_normalization", False)
        ):
            std = rewards_tensor.std(dim=-1, keepdim=True)
            rewards_tensor = rewards_tensor / (std + 1e-6)

    normalized_rewards = rewards_tensor.flatten().tolist()

    # ── 2. Split turns into independent training samples ──
    tokens_list = []
    response_lengths = []
    loss_masks = []
    rewards = []
    raw_reward_list = []
    truncated_list = []
    sample_indices = []
    rollout_log_probs_list = []
    has_rollout_log_probs = False

    for i, sample in enumerate(samples):
        meta = sample.train_metadata
        if meta is None or "turns" not in meta:
            tokens_list.append(sample.tokens)
            response_lengths.append(sample.response_length)
            lm = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
            if sample.remove_sample:
                lm = [0] * sample.response_length
            loss_masks.append(lm)
            rewards.append(normalized_rewards[i])
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(1 if sample.status == sample.Status.TRUNCATED else 0)
            sample_indices.append(sample.index)
            if sample.rollout_log_probs is not None:
                rollout_log_probs_list.append(sample.rollout_log_probs)
                has_rollout_log_probs = True
            continue

        turns = meta["turns"]
        norm_reward = normalized_rewards[i]
        is_truncated = 1 if sample.status == sample.Status.TRUNCATED else 0

        for turn in turns:
            tokens_list.append(turn["tokens"])
            response_lengths.append(turn["response_length"])
            lm = turn["loss_mask"]
            if sample.remove_sample:
                lm = [0] * turn["response_length"]
            loss_masks.append(lm)
            rewards.append(norm_reward)
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(is_truncated)
            sample_indices.append(sample.index)
            if turn.get("rollout_log_probs"):
                rollout_log_probs_list.append(turn["rollout_log_probs"])
                has_rollout_log_probs = True

    # ── 3. Trim to global_batch_size multiple ──
    # After turn expansion the sample count is no longer guaranteed to be
    # divisible by global_batch_size.  The training data-iterator requires
    # num_local_samples % (global_batch_size / dp_size) == 0, which is
    # satisfied when the total is a multiple of global_batch_size.
    gbs = args.global_batch_size
    total = len(tokens_list)
    trim_to = (total // gbs) * gbs
    if trim_to == 0:
        trim_to = total
    if trim_to < total:
        logger.info(
            "custom_convert: trimming expanded samples from %d to %d "
            "(global_batch_size=%d)",
            total, trim_to, gbs,
        )
        tokens_list = tokens_list[:trim_to]
        response_lengths = response_lengths[:trim_to]
        loss_masks = loss_masks[:trim_to]
        rewards = rewards[:trim_to]
        raw_reward_list = raw_reward_list[:trim_to]
        truncated_list = truncated_list[:trim_to]
        sample_indices = sample_indices[:trim_to]
        if has_rollout_log_probs:
            rollout_log_probs_list = rollout_log_probs_list[:trim_to]

    train_data = {
        "tokens": tokens_list,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "raw_reward": raw_reward_list,
        "truncated": truncated_list,
        "sample_indices": sample_indices,
    }
    if has_rollout_log_probs:
        train_data["rollout_log_probs"] = rollout_log_probs_list

    return train_data
