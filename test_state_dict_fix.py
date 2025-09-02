#!/usr/bin/env python3
"""
Test script to verify our state_dict fix works with pyreft's parameter structure.
"""
import torch
import sys
import os
sys.path.append('/cluster/home/tdieudonne/clmm/TRACE')

from pyreft import get_reft_model, ReftConfig
from loreft.reft_cl_intervention import ReftCLIntervention
from transformers import AutoModelForCausalLM

def test_state_dict_fix():
    """Test that our state_dict fix works with pyreft's parameter structure"""
    
    print("=== Testing State Dict Fix ===")
    
    # Create a minimal test
    base_model = AutoModelForCausalLM.from_pretrained(
        '/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat', 
        torch_dtype=torch.bfloat16, 
        device_map='cpu'
    )

    def _get_alpha(i): 
        return torch.tensor(0.1)

    reps = [{
        'layer': 4,
        'component': 'block_output',
        'low_rank_dimension': 8,
        'intervention': ReftCLIntervention(
            embed_dim=4096,
            low_rank_dimension=8,
            num_tasks=3,
            get_alpha=_get_alpha,
            eps=1e-8,
            dtype=torch.bfloat16,
        )
    }]

    cfg = ReftConfig(representations=reps)
    reft_model = get_reft_model(base_model, cfg, set_device=False)

    # Test our FIXED state_dict
    intervention = list(reft_model.interventions.values())[0]
    print('Testing FIXED state_dict:')
    custom_state = intervention.state_dict()
    print(f'Custom state keys: {list(custom_state.keys())}')

    # Check for task-specific keys with correct pyreft structure
    task_keys = [k for k in custom_state.keys() if 'tasks.' in k]
    print(f'Task-specific keys: {task_keys}')

    # Check if we have the correct pyreft parameter names
    expected_keys = [
        'tasks.0.rotate_layer.parametrizations.weight.original',
        'tasks.0.learned_source.weight',
        'tasks.0.learned_source.bias',
        'tasks.1.rotate_layer.parametrizations.weight.original',
        'tasks.1.learned_source.weight',
        'tasks.1.learned_source.bias',
        'tasks.2.rotate_layer.parametrizations.weight.original',
        'tasks.2.learned_source.weight',
        'tasks.2.learned_source.bias',
    ]
    
    has_correct_structure = all(key in custom_state for key in expected_keys)
    print(f'Has correct pyreft structure: {has_correct_structure}')
    
    if has_correct_structure:
        print('✅ SUCCESS: State dict has correct pyreft parameter structure!')
        
        # Test loading
        print('\n--- Testing Load ---')
        try:
            missing_keys, unexpected_keys = intervention.load_state_dict(custom_state, strict=False)
            print(f'✅ SUCCESS: Load state dict worked!')
            print(f'Missing keys: {missing_keys}')
            print(f'Unexpected keys: {unexpected_keys}')
            return True
        except Exception as e:
            print(f'❌ FAILED: Load state dict failed: {e}')
            return False
    else:
        print('❌ FAILED: State dict does not have correct pyreft structure')
        return False

if __name__ == "__main__":
    success = test_state_dict_fix()
    if success:
        print(f"\n✅ State dict fix test PASSED")
    else:
        print(f"\n❌ State dict fix test FAILED")
