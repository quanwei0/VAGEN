# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return_with_loss_mask(token_level_rewards: torch.Tensor, values: torch.Tensor, 
                                 loss_mask: torch.Tensor, gamma: float, lam: float):
    """Modified GAE calculation that handle multi-turn with loss mask
    Here we should also ensure that the trajectory score is given at the last valid token instead of last token
    Seems it's true in reward manager
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        for b in range(batch_size):
            lastgaelam = 0.0
            
            # Find the valid token positions (where loss_mask is 1)
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            
            if len(valid_positions) == 0:
                continue
                
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                
                # Get the next value
                if i < len(valid_positions) - 1:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = token_level_rewards[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            # Calculate returns for valid positions
            for i, pos in enumerate(valid_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
        

        advantages = verl_F.masked_whiten(advantages, loss_mask)
        
    return advantages, returns

def compute_bi_level_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        reward_mask: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        reward_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for reward position (end of each llm response)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = updated_reward[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
                updated_reward[b, pos] = advantages[b, pos] + values[b, pos]
            
            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    next_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_pos]
                else:
                    # Last valid position
                    nextvalue = 0.0
                    lastgaelam = 0.0
                delta = updated_reward[b, curr_pos] + gamma * nextvalue - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
                returns[b, curr_pos] = lastgaelam + values[b, curr_pos]
        
        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns


def compute_turn_wise_gae_advantage_return(
        token_level_rewards: torch.Tensor,
        reward_mask: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        lam: float,
        high_level_gamma: float
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        reward_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for reward position (end of each llm response)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            eos_positions=reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0
            for i in range(len(eos_positions) - 1, -1, -1):
                curr_pos = eos_positions[i]
                
                # Get the next value
                if i < len(eos_positions) - 1:
                    # Next valid position
                    next_pos = eos_positions[i + 1]
                    nextvalue = values[b, next_pos]
                    
                else:
                    # Last valid position
                    nextvalue = 0.0
                
                # Calculate delta using the next valid token
                delta = token_level_rewards[b, curr_pos] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            for i, pos in enumerate(eos_positions):
                returns[b, pos] = advantages[b, pos] + values[b, pos]
            
            # each token in the sequence has the same advantage
            cur_adv = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                if curr_pos not in eos_positions:
                    # Next valid position
                    advantages[b, curr_pos] = cur_adv
                else:
                    # Last valid position
                    cur_adv=advantages[b, curr_pos]
        
        advantages = verl_F.masked_whiten(advantages, reward_mask)
    
    return advantages, returns
                    
        
        

def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0 
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t] # TD error
            lastgaelam = delta + gamma * lam * lastgaelam # gae
            advantages_reversed.append(lastgaelam) # store the gae
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


# def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
#     kl = old_log_prob - ref_log_prob
#     return token_level_scores - kl * kl_ratio

def compute_policy_loss( 
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange,
    detach_ratio='soft',
    importance_sampling_level='token',
    turn_indices=None,
):
    """
    Computes PPO policy loss with optional detached importance sampling ratio.
    If detach_ratio=True, ratio is linearly decayed to 0 outside the clipping range.

    Args:
        old_log_prob: old log probabilities for each token
        log_prob: new log probabilities for each token  
        advantages: advantage values for each token
        eos_mask: mask for valid tokens (completion_mask)
        cliprange: clipping range for PPO
        detach_ratio: detachment strategy ('soft', 'hard', or None)
        importance_sampling_level: importance sampling strategy
            - 'token': each token uses its own importance weight
            - 'sequence': traditional sequence-level where all tokens share the same weight (average over sequence)
            - 'partial_sequence': partial-sequence-level GRPO where each token at position t uses 
                                 cumulative average of log_ratio from position 1 to t
            - 'turn': turn-level where all tokens within the same turn share the same importance weight
        turn_indices: tensor of shape (batch_size, max_indices) containing turn indices,
                     required when importance_sampling_level='turn'

    Returns:
        pg_loss: scalar tensor
        pg_clipfrac: float tensor
        ppo_kl: scalar tensor
    """
    log_ratio = log_prob - old_log_prob
    
    # Compute importance weights based on the specified level
    if importance_sampling_level == "token":
        log_importance_weights = log_ratio
        print("="*80)
        print(f"[Debug] token-level importance sampling")
        print("="*80)
    elif importance_sampling_level == "sequence":
        print("="*80)
        print(f"[Debug] traditional sequence-level importance sampling")
        print("="*80)
        # Traditional sequence-level importance sampling: all tokens share same weight (average over sequence)
        log_importance_weights = (log_ratio * eos_mask).sum(-1) / eos_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    elif importance_sampling_level == "partial_sequence":
        print("="*80)
        print(f"[Debug] partial-sequence-level importance sampling")
        print("="*80)
        # Partial-Sequence-level importance sampling: 
        # For each token at position t, use cumulative average of log_ratio from position 1 to t
        # w_{i,t} = exp(1/t * Σ_{s=1}^t log_ratio_s)
        
        batch_size, seq_len = log_ratio.shape
        log_importance_weights = torch.zeros_like(log_ratio)
        
        for b in range(batch_size):
            mask = eos_mask[b]  # mask for this batch
            valid_positions = mask.nonzero(as_tuple=True)[0]  # positions where mask=1
            
            if len(valid_positions) == 0:
                continue
                
            # Extract log_ratios for valid positions only
            valid_log_ratios = log_ratio[b, valid_positions]  # shape: [num_valid_tokens]
            
            # Compute cumulative sum and divide by position indices to get cumulative averages
            cumsum = torch.cumsum(valid_log_ratios, dim=0)  # [num_valid_tokens]
            position_indices = torch.arange(1, len(valid_positions) + 1, 
                                          dtype=cumsum.dtype, device=cumsum.device)  # [1, 2, 3, ...]
            cumulative_averages = cumsum / position_indices  # [num_valid_tokens]
            
            # Assign back to the original positions
            log_importance_weights[b, valid_positions] = cumulative_averages
    elif importance_sampling_level == "turn":
        print("="*80)
        print(f"[Debug] turn-level importance sampling")
        print("="*80)
        # Turn-level importance sampling: all tokens within the same turn share the same importance weight
        # The weight for each turn is the average of log_ratios of all tokens within that turn
        
        if turn_indices is None:
            raise ValueError("turn_indices must be provided when importance_sampling_level='turn'")
        
        batch_size, seq_len = log_ratio.shape
        log_importance_weights = torch.zeros_like(log_ratio)
        
        for b in range(batch_size):
            mask = eos_mask[b]  # mask for this batch
            
            # Get turn indices for this batch sample - stored as flattened [start1, end1, start2, end2, ...]
            turn_indices_b = turn_indices[b]  # shape: (max_indices,)
            
            # Find valid turn indices (not equal to -1)
            valid_indices = turn_indices_b[turn_indices_b != -1]
            
            if len(valid_indices) == 0 or len(valid_indices) % 2 != 0:
                # If no valid indices or odd number of indices, fall back to token-level
                log_importance_weights[b] = log_ratio[b]
                continue
            
            # Parse pairs of (start, end) indices
            num_turns = len(valid_indices) // 2
            
            for turn_idx in range(num_turns):
                start_pos = valid_indices[turn_idx * 2].item()
                end_pos = valid_indices[turn_idx * 2 + 1].item()
                
                # Ensure positions are within sequence bounds
                start_pos = max(0, min(start_pos, seq_len - 1))
                end_pos = max(0, min(end_pos, seq_len - 1))
                
                if start_pos > end_pos:
                    continue
                
                # Get log_ratios for tokens in this turn that are valid (masked)
                turn_mask = mask[start_pos:end_pos + 1]
                turn_log_ratios = log_ratio[b, start_pos:end_pos + 1]
                
                if turn_mask.sum() > 0:
                    # Compute average log_ratio for this turn (only for valid tokens)
                    turn_avg_log_ratio = (turn_log_ratios * turn_mask).sum() / turn_mask.sum()
                    
                    # Assign this average to all tokens in the turn
                    log_importance_weights[b, start_pos:end_pos + 1] = turn_avg_log_ratio

    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token', "
            "'sequence', 'partial_sequence', and 'turn'."
        )
    
    negative_approx_kl = log_importance_weights
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    
    if detach_ratio=='hard':
        print("="*80)
        print(f"[Debug] hard detach ratio with {importance_sampling_level}-level importance sampling and the cliprange is",cliprange)
        print("="*80)
        # Detach ratio but still apply clipping
        ratio_detached = ratio.detach()
        clipped_ratio_detached = torch.clamp(ratio_detached, 1.0 - cliprange, 1.0 + cliprange)

        pg_losses1 = -ratio_detached * advantages * log_prob
        pg_losses2 = -clipped_ratio_detached * advantages * log_prob

        pg_loss = verl_F.masked_mean(torch.max(pg_losses1, pg_losses2), eos_mask)
        pg_clipfrac = verl_F.masked_mean((pg_losses2 > pg_losses1).float(), eos_mask)
    elif detach_ratio=='soft':
        print("="*80)
        print(f"[Debug] soft detach ratio with {importance_sampling_level}-level importance sampling and the cliprange is",cliprange)
        print("="*80)
        ratio_detached = ratio.detach()

        # Define a smooth mask in the range [1 - 2ε, 1 - ε] ∪ [1 + ε, 1 + 2ε]
        lower = 1.0 - cliprange
        upper = 1.0 + cliprange
        lower_decay = 1.0 - 1.05 * cliprange 
        upper_decay = 1.0 + 1.3 * cliprange

        # Create smooth decay mask: multiplier ∈ [1, 0]
        decay_mask = torch.ones_like(ratio_detached)

        # Left decay: [1 - 2ε, 1 - ε]
        left_mask = (ratio_detached >= lower_decay) & (ratio_detached < lower)
        decay_mask[left_mask] = (ratio_detached[left_mask] - lower_decay) / cliprange

        # Right decay: [1 + ε, 1 + 2ε]
        right_mask = (ratio_detached > upper) & (ratio_detached <= upper_decay)
        decay_mask[right_mask] = (upper_decay - ratio_detached[right_mask]) / cliprange

        # Outside full decay region: ratio < 1 - 2ε or > 1 + 2ε → mask = 0
        decay_mask[ratio_detached < lower_decay] = 0.0
        decay_mask[ratio_detached > upper_decay] = 0.0

        # Apply soft decay mask
        effective_ratio = ratio_detached * decay_mask

        pg_losses = -advantages * log_prob * effective_ratio
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
        pg_clipfrac = verl_F.masked_mean((decay_mask < 1.0).float(), eos_mask)

    else:
        # Standard PPO
        print("="*80)
        print(f"[Debug] normal detach ratio with {importance_sampling_level}-level importance sampling and the cliprange is",cliprange)
        print("="*80)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

        pg_loss = verl_F.masked_mean(torch.max(pg_losses1, pg_losses2), eos_mask)
        pg_clipfrac = verl_F.masked_mean((pg_losses2 > pg_losses1).float(), eos_mask)

    return pg_loss, pg_clipfrac, ppo_kl

def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
