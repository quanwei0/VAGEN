#!/usr/bin/env python3
"""
Test script for turn-level importance sampling implementation
"""
import torch
import sys
import os

# Add the project root to the path
sys.path.append('/home/li003968/VAGEN')

from verl.trainer.ppo import core_algos

def test_turn_importance_sampling():
    """Test turn-level importance sampling functionality"""
    
    print("🧪 Testing Turn-Level Importance Sampling")
    print("=" * 60)
    
    # Create test data
    batch_size = 2
    seq_len = 10
    
    # Mock log probabilities
    old_log_prob = torch.randn(batch_size, seq_len) * 0.1
    log_prob = old_log_prob + torch.randn(batch_size, seq_len) * 0.05
    
    # Mock advantages
    advantages = torch.randn(batch_size, seq_len)
    
    # Mock EOS mask (1 for valid tokens, 0 for padding)
    eos_mask = torch.ones(batch_size, seq_len)
    eos_mask[0, 8:] = 0  # First sample has padding after position 7
    eos_mask[1, 9:] = 0  # Second sample has padding after position 8
    
    # Mock turn indices: [start1, end1, start2, end2, ...]
    # Sample 0: turns at (1,3) and (5,7)
    # Sample 1: turns at (0,2), (4,6) and (8,8)
    turn_indices = torch.full((batch_size, 20), -1, dtype=torch.long)
    turn_indices[0, :4] = torch.tensor([1, 3, 5, 7])  # Two turns
    turn_indices[1, :6] = torch.tensor([0, 2, 4, 6, 8, 8])  # Three turns
    
    print(f"📊 Test Data Shape:")
    print(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"  EOS mask sample 0: {eos_mask[0].tolist()}")
    print(f"  EOS mask sample 1: {eos_mask[1].tolist()}")
    print(f"  Turn indices sample 0: {turn_indices[0][turn_indices[0] != -1].tolist()}")
    print(f"  Turn indices sample 1: {turn_indices[1][turn_indices[1] != -1].tolist()}")
    
    # Test different importance sampling levels
    levels = ['token', 'sequence', 'partial_sequence', 'turn']
    
    for level in levels:
        print(f"\n🎯 Testing {level.upper()}-level importance sampling:")
        print("-" * 40)
        
        try:
            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                eos_mask=eos_mask,
                cliprange=0.2,
                detach_ratio='soft',
                importance_sampling_level=level,
                turn_indices=turn_indices if level == 'turn' else None
            )
            
            print(f"  ✅ Success!")
            print(f"  📈 Policy loss: {pg_loss.item():.6f}")
            print(f"  📊 Clip fraction: {pg_clipfrac.item():.6f}")
            print(f"  🔄 PPO KL: {ppo_kl.item():.6f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {str(e)}")
    
    # Test edge cases
    print(f"\n🔍 Testing Edge Cases:")
    print("-" * 40)
    
    # Test with empty turn indices
    empty_turn_indices = torch.full((batch_size, 20), -1, dtype=torch.long)
    try:
        pg_loss, _, _ = core_algos.compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=eos_mask,
            cliprange=0.2,
            detach_ratio='soft',
            importance_sampling_level='turn',
            turn_indices=empty_turn_indices
        )
        print(f"  ✅ Empty turn indices handled correctly")
    except Exception as e:
        print(f"  ❌ Empty turn indices failed: {str(e)}")
    
    # Test without turn_indices for turn level (should raise error)
    try:
        pg_loss, _, _ = core_algos.compute_policy_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            eos_mask=eos_mask,
            cliprange=0.2,
            detach_ratio='soft',
            importance_sampling_level='turn',
            turn_indices=None
        )
        print(f"  ❌ Should have raised error for missing turn_indices")
    except ValueError as e:
        print(f"  ✅ Correctly raised error for missing turn_indices: {str(e)}")
    except Exception as e:
        print(f"  ❌ Unexpected error: {str(e)}")
    
    print(f"\n🎉 Turn-level importance sampling test completed!")

if __name__ == "__main__":
    test_turn_importance_sampling()
