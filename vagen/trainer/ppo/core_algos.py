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
import random


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


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
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


def compute_bi_level_gae_advantage_return_v2(
        token_level_rewards: torch.Tensor,
        values: torch.Tensor, 
        loss_mask: torch.Tensor,
        gamma: float,
        lam: float,
        high_level_gamma: float,
        high_level_lam: float,
        response_mask: torch.Tensor = None
    ):
    """Modified GAE calculation that compute two level of advantage and return:
    high level: per-turn wise
    low level: token wise
    there're two level of MDP, where high level is the agentic MDP and low level is the token MDP
    Args:
        token_level_rewards: `(torch.Tensor)` (multi-turn reward, per turn reward is given at eos token for each response token sequence)
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`
            shape: (bs, response_length). 1 for llm_raw_response, 0 for environment info and paddings
        gamma: `(float)`
            discounted factor used in RL for token rewards
        high_level_gamma: `(float)`
            discounted factor used in RL for per-turn reward
        high_level_lam: `(float)`
            lambda value when computing Generalized Advantage Estimation
        response_mask: `(torch.Tensor)` optional
            shape: (bs, response_length). 1 for LLM generation, 0 for observation. Used to find turn boundaries.

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    with torch.no_grad():
        token_level_rewards = token_level_rewards.float()
        
        ##########################################################################################
        # Example:
        # response_mask = [0,0,0,1,1,1,0,0,1,1,1,1,0,0,0,1,1,1]
        # reward_mask   = [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0]

        if response_mask is not None:
            batch_size, seq_len = response_mask.shape
            reward_mask = torch.zeros_like(response_mask, dtype=torch.float)

            for b in range(batch_size):
                response_seq = response_mask[b]

                # Identify turn start points: positions where response begins (0 → 1 transition)
                # This gives the indices of the first token of each response turn
                turn_start_pos = ((response_seq[1:] == 1) & (response_seq[:-1] == 0)).nonzero(as_tuple=True)[0] + 1

                reward_mask[b, turn_start_pos] = 1.0

        else:
            # Use traditional reward mask
            reward_mask = token_level_rewards.bool()
        ##########################################################################################
        
        batch_size, gen_len = token_level_rewards.shape
        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        updated_reward = token_level_rewards.clone()
        
        for b in range(batch_size):
            # First, calculate high level advantage and return for eos token of each turn using high level gamma
            turn_start_pos = reward_mask[b].nonzero(as_tuple=True)[0]
            lastgaelam = 0.0            
            for i in range(len(turn_start_pos) - 1, -1, -1):
                curr_pos = turn_start_pos[i]
                
                # Get the next value
                if i < len(turn_start_pos) - 1:
                    # Next valid position
                    next_pos = turn_start_pos[i + 1]
                    nextvalue = values[b, next_pos]
                    
                    # Calculate delta using the next valid token
                    delta = 0 + high_level_gamma * nextvalue - values[b, curr_pos]                    
                    
                else:
                    # Last valid position
                    nextvalue = 0.0

                    # Calculate delta using the next valid token
                    delta = updated_reward[b, -1] + high_level_gamma * nextvalue - values[b, curr_pos]
                
                # Update advantage estimate
                lastgaelam = delta + high_level_gamma * high_level_lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam
            
            turn_level_adv = advantages.clone()
            # for i, pos in enumerate(turn_start_pos):
                # returns[b, pos] = advantages[b, pos] + values[b, pos]
                # updated_reward[b, pos] = advantages[b, pos] + values[b, pos]

            # Then, calculate low level advantage and return for each token using gamma, assume the reward for the sequence now is the return at eos token
            lastgaelam = 0.0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for i in range(len(valid_positions) - 1, -1, -1):
                curr_valid_pos = valid_positions[i]

                # for last turn
                if curr_valid_pos >= turn_start_pos[-1]:
                    # for non-last token in the last turn
                    if i != len(valid_positions) - 1:
                        next_valid_pos = valid_positions[i + 1]
                        nextvalue = values[b, next_valid_pos]
                        delta = 0 + gamma * nextvalue - values[b, curr_valid_pos]
                    # for last token in the last turn
                    else:
                        nextvalue = 0.0
                        lastgaelam = 0.0
                        delta = updated_reward[b, -1] + gamma * nextvalue - values[b, curr_valid_pos]

                # for non-last turn
                else:
                    next_valid_pos = valid_positions[i + 1]
                    nextvalue = values[b, next_valid_pos]
                    
                    if next_valid_pos in (turn_start_pos).tolist():
                        lastgaelam = turn_level_adv[b, next_valid_pos]

                    delta = 0 + gamma * nextvalue - values[b, curr_valid_pos]
                               
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_valid_pos] = lastgaelam
                returns[b, curr_valid_pos] = lastgaelam + values[b, curr_valid_pos]

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    
    return advantages, returns

def compute_multiturn_gae_hierarchical(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    lam: float,
    alpha: float = 0.7,  # weight for token-level advantages
    high_level_gamma: float = None,  # gamma for turn-level GAE, defaults to token-level gamma
    high_level_lam: float = None,  # lambda for turn-level GAE, defaults to token-level lam
):
    """
    Hierarchical GAE: compute token-level and turn-level advantages separately, then combine
    
    Args:
        turn_level_method: "average" - use averaging method, "gae" - use GAE method for turn-level advantages
        high_level_gamma: gamma value for turn-level GAE, uses token-level gamma if None
    """
    if high_level_gamma is None:
        high_level_gamma = gamma
    if high_level_lam is None:
        high_level_lam = lam
        
    with torch.no_grad():
        bs, seq_len = token_level_rewards.shape
        
        # Step 1: Compute token-level advantages using existing method
        lastgaelam = 0
        token_advantages = []
        response_mask_f = response_mask.float()
        gamma_masked = response_mask_f * gamma + 1 - response_mask_f
        lam_masked = response_mask_f * lam + 1 - response_mask_f
        nextvalues_skip_obs = 0

        for t in reversed(range(seq_len)):
            next_step_mask = response_mask_f[:, t + 1] if t < seq_len - 1 else 1.0
            nextvalues = values[:, t + 1] if t < seq_len - 1 else 0.0
            nextvalues_skip_obs = (1 - next_step_mask) * nextvalues_skip_obs + next_step_mask * nextvalues
            this_step_gamma = gamma_masked[:, t]
            this_step_lam = lam_masked[:, t]
            delta = token_level_rewards[:, t] + this_step_gamma * nextvalues_skip_obs - values[:, t]
            delta *= response_mask_f[:, t]
            lastgaelam = delta + this_step_gamma * this_step_lam * lastgaelam
            token_advantages.append(lastgaelam)
        
        token_advantages = torch.stack(token_advantages[::-1], dim=1)
        
        turn_advantages = _compute_turn_level_gae(
            token_level_rewards, values, response_mask, response_mask_f, 
            high_level_gamma, lam=high_level_lam, bs=bs, seq_len=seq_len
        )
        # Step 3: Combine advantages
        combined_advantages = alpha * token_advantages + (1-alpha) * turn_advantages
        # returns = combined_advantages + values
        returns = token_advantages + values
        combined_advantages = verl_F.masked_whiten(combined_advantages, response_mask_f)
        
    return combined_advantages, returns

def _compute_turn_level_gae(token_level_rewards, values, response_mask, response_mask_f, 
                           high_level_gamma, lam, bs, seq_len, reward_aggregation='sparse'):
    """Use GAE method to compute turn-level advantages with shared advantage within each turn
    
    Args:
        reward_aggregation: How to aggregate rewards for each turn
            - 'sparse': Only use reward at terminal token of each turn
            - 'sum': Use sum of all rewards within each turn
            - 'average': Use average of all rewards within each turn
    """
    turn_advantages = torch.zeros_like(token_level_rewards)
    
    # Compute GAE for each batch separately
    for b in range(bs):
        # Find turn boundaries: first and last position of each response turn
        turn_starts = []
        turn_ends = []
        
        in_response = False
        turn_start = None
        
        for t in range(seq_len):
            if response_mask[b, t] > 0:  # Response token
                if not in_response:
                    # Starting a new turn
                    turn_start = t
                    in_response = True
            else:  # Non-response token (or end of sequence)
                if in_response:
                    # Ending current turn
                    turn_starts.append(turn_start)
                    turn_ends.append(t - 1)  # Last response token
                    in_response = False
        
        # Handle case where sequence ends with a response
        if in_response:
            turn_starts.append(turn_start)
            turn_ends.append(seq_len - 1)
        
        # Compute GAE backwards through turns
        if len(turn_starts) > 0:
            lastgaelam = 0.0
            
            for i in range(len(turn_starts) - 1, -1, -1):
                start_pos = turn_starts[i]
                end_pos = turn_ends[i]
                
                # Get reward for current turn based on aggregation method
                if reward_aggregation == 'sparse':
                    # Only use reward at terminal token
                    reward = token_level_rewards[b, end_pos]
                elif reward_aggregation == 'sum':
                    # Sum all rewards in this turn
                    reward = 0.0
                    for t in range(start_pos, end_pos + 1):
                        if response_mask[b, t] > 0:
                            reward += token_level_rewards[b, t]
                elif reward_aggregation == 'average':
                    # Average all rewards in this turn
                    reward_sum = 0.0
                    count = 0
                    for t in range(start_pos, end_pos + 1):
                        if response_mask[b, t] > 0:
                            reward_sum += token_level_rewards[b, t]
                            count += 1
                    reward = reward_sum / count if count > 0 else 0.0
                else:
                    raise ValueError(f"Unknown reward_aggregation: {reward_aggregation}")
                
                # Get value at start of current turn
                current_value = values[b, start_pos]
                
                # Get value at start of next turn (if exists)
                if i < len(turn_starts) - 1:
                    next_start_pos = turn_starts[i + 1]
                    next_value = values[b, next_start_pos]
                else:
                    next_value = 0.0
                
                # Calculate delta: reward + gamma * next_value - current_value
                delta = reward + high_level_gamma * next_value - current_value
                lastgaelam = delta + high_level_gamma * lam * lastgaelam
                
                # # Assign the same advantage to all tokens in this turn
                # for t in range(start_pos, end_pos + 1):
                #     if response_mask[b, t] > 0:
                #         turn_advantages[b, t] = lastgaelam
    
                n_tokens = (response_mask[b, start_pos:end_pos+1] > 0).sum()
                avg_adv = lastgaelam / n_tokens
                for t in range(start_pos, end_pos + 1):
                    if response_mask[b, t] > 0:
                        turn_advantages[b, t] = avg_adv
                        
    return turn_advantages


if __name__ == "__main__":
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)
    high_level_gamma = random.uniform(0.0, 1.0)
    

    # rewards = torch.tensor([
    #     [ 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0]
    # ], dtype=torch.float)
    sparse_rewards = torch.tensor([
        [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float)
    
    step_rewards = torch.tensor([
        [ 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float)
    
    values1 = torch.tensor([
        [ random.uniform(-100.0, 100.0), random.random(), 4.0, 5.0, 6.0, random.uniform(-100.0, 0), random.random(), 7.0, 9.0]
    ], dtype=torch.float)
    
    values2 = torch.tensor([
        [ random.random(), random.uniform(-100.0, 100.0), 4.0, 5.0, 6.0, random.random(), random.uniform(0.0, 100.0), 7.0, 9.0]
    ], dtype=torch.float)
    
    eos_mask = torch.tensor([
        [ 0, 0, 1, 1, 1, 0, 0, 1, 1] 
    ], dtype=torch.float)
    
    reward_mask = torch.tensor([
        [ 0, 0, 0, 0, 1, 0, 0, 0, 1] 
    ], dtype=torch.float)
    # adv1, ret1 = compute_bi_level_gae_advantage_return(rewards, values1, eos_mask, gamma=1, lam=1, high_level_gamma=1, loss_mask=eos_mask)
    # adv2, ret2 = compute_bi_level_gae_advantage_return(rewards, values2, eos_mask, gamma, lam, high_level_gamma=0.95, response_mask=eos_mask, high_level_lam=1.0)
    # adv1, ret1 = compute_weighted_cross_level_gae_advantage_return(rewards, values1, eos_mask, gamma=1, lam=1, high_level_gamma=1, response_mask=eos_mask, high_level_lam=1, turn_level_weight=0)
    
    # adv1, ret1 = compute_gae_advantage_return_multi_turn_old(rewards, values1, eos_mask, gamma, lam)
    # adv2, ret2 = compute_gae_advantage_return_multi_turn(rewards, values1, eos_mask, gamma=1, lam=1)
    
    # adv1, ret1 = compute_multiturn_gae_hierarchical(rewards, values1, eos_mask, gamma, lam, alpha=1.0, turn_level_method="gae", high_level_gamma=0.95)
    # adv2, ret2 = compute_multiturn_gae_hierarchical(rewards, values2, eos_mask, gamma=1, lam=1, alpha=0, turn_level_method="gae", high_level_gamma=1, high_level_lam=1)
    
    
    # adv1, ret1 = compute_turn_wise_gae_advantage_return(token_level_rewards=sparse_rewards,
    #                                                      reward_mask=reward_mask,values=values1,
    #                                                      loss_mask=eos_mask, lam=lam, high_level_gamma=1.0)
    # adv2, ret2 = compute_turn_wise_gae_advantage_return(token_level_rewards=sparse_rewards,
    #                                                      reward_mask=reward_mask,values=values2,
    #                                                      loss_mask=eos_mask, lam=lam, high_level_gamma=1.0)
    
    ##### Bi-level GAE Advantage Return
    # adv1, ret1 = compute_bi_level_gae_advantage_return(token_level_rewards=sparse_rewards,
    #                                                      reward_mask=reward_mask,values=values1,gamma=gamma,
    #                                                      loss_mask=eos_mask, lam=lam, high_level_gamma=high_level_gamma)
    # adv2, ret2 = compute_bi_level_gae_advantage_return(token_level_rewards=sparse_rewards,
    #                                                      reward_mask=reward_mask,values=values2,gamma=gamma,
    #                                                      loss_mask=eos_mask, lam=lam, high_level_gamma=high_level_gamma)
    
    ##### Turn-wise GAE Advantage Return
    adv1, ret1 = compute_turn_wise_gae_advantage_return(token_level_rewards=sparse_rewards,
                                                         reward_mask=reward_mask,values=values1,
                                                         loss_mask=eos_mask, lam=lam, high_level_gamma=high_level_gamma)
    adv2, ret2 = compute_turn_wise_gae_advantage_return(token_level_rewards=sparse_rewards,
                                                         reward_mask=reward_mask,values=values2,
                                                         loss_mask=eos_mask, lam=lam, high_level_gamma=high_level_gamma)        
    
    ### Our Bi-level GAE Advantage Return
    
    
    # adv1, ret1 = compute_bi_level_gae_advantage_return_v2(token_level_rewards=step_rewards,values=values1,gamma=gamma,
    #                                                      loss_mask=eos_mask, lam=lam, high_level_gamma=high_level_gamma,high_level_lam=lam, response_mask=eos_mask)
    # adv2, ret2 = compute_bi_level_gae_advantage_return_v2(token_level_rewards=step_rewards,values=values2,gamma=gamma,
    #                                                      loss_mask=eos_mask, lam=lam, high_level_gamma=high_level_gamma,high_level_lam=lam, response_mask=eos_mask)
    
    
    # adv2, ret2 = compute_bi_level_gae_advantage_return(token_level_rewards=rewards, values=values1, reward_mask=reward_mask, loss_mask=eos_mask, high_level_gamma=1,gamma=1, lam=1)
    # adv2, ret2 = compute_gae_advantage_return_with_loss_mask(
        # token_level_rewards=sparse_rewards, values=values1, loss_mask=eos_mask, gamma=gamma, lam=lam
    # )
    # ret1 *= eos_mask
    # ret2 *= eos_mask
    # assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    # assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    # print(f' [CORRECT] \n\n{adv1=}, \n\n{adv2=}')
    print(f' [CORRECT] \n\n{adv1=}, \n\n{adv2=}, \n\n{ret1=}, \n\n{ret2=}')