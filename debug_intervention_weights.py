#!/usr/bin/env python3
"""
Debug script to compare intervention weights across training rounds.
If weights are identical, it means they weren't actually trained.
"""
import os
import torch
import numpy as np

def compare_intervention_weights():
    """Compare intervention weights across all rounds for layer 4"""
    
    base_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers"
    
    # Focus on layer 4 intervention file
    target_file = "intkey_layer_4_comp_block_output_unit_pos_nunit_1#0.bin"
    
    print("=== Intervention Weight Analysis Across Rounds ===")
    print(f"Analyzing: {target_file}")
    print()
    
    # Load weights from all available rounds
    rounds_data = {}
    available_rounds = []
    
    for round_num in range(8):  # 8 rounds (0-7)
        round_path = os.path.join(base_path, str(round_num))
        file_path = os.path.join(round_path, target_file)
        
        if os.path.exists(file_path):
            try:
                weights = torch.load(file_path, map_location="cpu")
                rounds_data[round_num] = weights
                available_rounds.append(round_num)
                print(f"✓ Round {round_num}: Loaded {len(weights)} keys")
            except Exception as e:
                print(f"❌ Round {round_num}: Failed to load - {e}")
        else:
            print(f"⚠️  Round {round_num}: File not found")
    
    if len(available_rounds) < 2:
        print(f"\n❌ Need at least 2 rounds to compare, found {len(available_rounds)}")
        return
    
    print(f"\n📊 Comparing weights across {len(available_rounds)} rounds: {available_rounds}")
    
    # Get first round as baseline
    baseline_round = available_rounds[0]
    baseline_weights = rounds_data[baseline_round]
    
    print(f"\nUsing Round {baseline_round} as baseline")
    print("Weight tensor shapes:")
    for key, tensor in baseline_weights.items():
        if torch.is_tensor(tensor):
            print(f"  {key}: {tuple(tensor.shape)} {tensor.dtype}")
        else:
            print(f"  {key}: {tensor} ({type(tensor).__name__})")
    
    # Compare each round against baseline
    print(f"\n🔍 Comparing all rounds against Round {baseline_round}:")
    
    identical_rounds = []
    different_rounds = []
    
    for round_num in available_rounds[1:]:  # Skip baseline
        round_weights = rounds_data[round_num]
        
        print(f"\n--- Round {round_num} vs Round {baseline_round} ---")
        
        all_identical = True
        
        for key in baseline_weights.keys():
            if key not in round_weights:
                print(f"  ❌ Key '{key}' missing in Round {round_num}")
                all_identical = False
                continue
                
            baseline_tensor = baseline_weights[key]
            round_tensor = round_weights[key]
            
            if (
                torch.is_tensor(baseline_tensor) 
                and torch.is_tensor(round_tensor) 
                and (torch.is_floating_point(baseline_tensor) or torch.is_complex(baseline_tensor))
            ):

                if baseline_tensor.shape != round_tensor.shape:
                    print(f"  ❌ {key}: Shape mismatch {baseline_tensor.shape} vs {round_tensor.shape}")
                    all_identical = False
                    continue
                
                # Check if tensors are identical
                are_identical = torch.allclose(baseline_tensor, round_tensor, atol=1e-8)
                max_diff = torch.max(torch.abs(baseline_tensor - round_tensor)).item()
                mean_abs_diff = torch.mean(torch.abs(baseline_tensor - round_tensor)).item()
                
                if are_identical:
                    print(f"  🟡 {key}: IDENTICAL (max_diff: {max_diff:.2e})")
                else:
                    print(f"  ✅ {key}: DIFFERENT (max_diff: {max_diff:.2e}, mean_abs_diff: {mean_abs_diff:.2e})")
                    all_identical = False
                    
            else:
                # Non-tensor comparison
                if baseline_tensor == round_tensor:
                    print(f"  🟡 {key}: IDENTICAL ({baseline_tensor})")
                else:
                    print(f"  ✅ {key}: DIFFERENT ({baseline_tensor} -> {round_tensor})")
                    all_identical = False
        
        if all_identical:
            identical_rounds.append(round_num)
            print(f"  🚨 Round {round_num}: ALL WEIGHTS IDENTICAL TO BASELINE")
        else:
            different_rounds.append(round_num)
            print(f"  ✅ Round {round_num}: Some weights changed")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"  Baseline: Round {baseline_round}")
    print(f"  Identical to baseline: {identical_rounds} ({len(identical_rounds)} rounds)")
    print(f"  Different from baseline: {different_rounds} ({len(different_rounds)} rounds)")
    
    if len(identical_rounds) > 0:
        print(f"\n🚨 PROBLEM DETECTED:")
        print(f"  {len(identical_rounds)} rounds have identical weights to Round {baseline_round}")
        print(f"  This suggests intervention weights were NOT trained properly!")
        print(f"  Likely cause: Intervention parameters not included in optimizer")
        
        # Check if this is a pattern across all rounds
        if len(identical_rounds) == len(available_rounds) - 1:
            print(f"\n💥 CRITICAL: ALL rounds have identical weights!")
            print(f"  Intervention training completely failed")
        else:
            print(f"\n⚠️  PARTIAL: Some rounds trained, others didn't")
            
    else:
        print(f"\n✅ GOOD: All rounds have different weights")
        print(f"  Intervention training appears to be working")
    
    # Additional check: Look at alpha parameters if available
    print(f"\n🔍 Checking alpha parameters...")
    for round_num in available_rounds:
        weights = rounds_data[round_num]
        if 'active_tasks' in weights:
            print(f"  Round {round_num}: active_tasks = {weights['active_tasks'].item()}")
    
    return len(identical_rounds) == 0

if __name__ == "__main__":
    success = compare_intervention_weights()
    if success:
        print(f"\n✅ Intervention weights appear to be training correctly")
    else:
        print(f"\n❌ Intervention weights are NOT being trained!")
        print(f"   This explains why loaded models have no effect.")
        print(f"   Need to investigate why intervention parameters aren't in optimizer.")
