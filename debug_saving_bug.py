#!/usr/bin/env python3
"""
Debug whether the intervention saving/loading is working correctly.
Test: Do saved intervention files actually contain different weights?
"""
import os
import torch

def debug_intervention_saving():
    """Check if intervention files are being saved with different weights"""
    
    base_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers_2"
    target_file = "intkey_layer_4_comp_block_output_unit_pos_nunit_1#0.bin"
    
    print("=== Debugging Intervention Saving Bug ===")
    print(f"Target file: {target_file}")
    print()
    
    # Load intervention files from multiple rounds
    rounds_data = {}
    
    for round_num in [0, 1, 2, 3, 4, 5, 6, 7]:
        file_path = os.path.join(base_path, str(round_num), target_file)
        if os.path.exists(file_path):
            try:
                weights = torch.load(file_path, map_location="cpu")
                rounds_data[round_num] = weights
                print(f"✓ Round {round_num}: Loaded intervention file")
            except Exception as e:
                print(f"❌ Round {round_num}: Failed to load - {e}")
        else:
            print(f"⚠️  Round {round_num}: File not found")
    
    if len(rounds_data) < 2:
        print(f"\n❌ Need at least 2 rounds to compare")
        return
    
    print(f"\n🔍 Detailed comparison of task-specific weights:")
    
    # According to REFT-CL training regime:
    # Task i parameters should ONLY change during round i, then stay frozen
    
    for task_id in range(8):
        print(f"\n--- Task {task_id} Analysis ---")
        
        # Find task-specific parameters
        first_round = min(rounds_data.keys())
        task_keys = [k for k in rounds_data[first_round].keys() 
                    if f"tasks.{task_id}." in k and torch.is_tensor(rounds_data[first_round][k])]
        
        if not task_keys:
            print(f"  No task {task_id} parameters found")
            continue
            
        print(f"  Task {task_id} has {len(task_keys)} parameter tensors")
        
        # Check each parameter across rounds
        for key in task_keys:
            print(f"\n  Parameter: {key}")
            
            # Track when this parameter changes
            changes = []
            prev_round = None
            
            for round_num in sorted(rounds_data.keys()):
                if key in rounds_data[round_num]:
                    current_tensor = rounds_data[round_num][key]
                    
                    if prev_round is not None:
                        prev_tensor = rounds_data[prev_round][key]
                        
                        if prev_tensor.shape == current_tensor.shape:
                            max_diff = torch.max(torch.abs(prev_tensor - current_tensor)).item()
                            
                            # Should change only at round = task_id
                            expected_change = (round_num == task_id)
                            actually_changed = (max_diff > 1e-6)
                            
                            status = "✅" if (expected_change == actually_changed) else "❌"
                            change_desc = "CHANGED" if actually_changed else "SAME"
                            expected_desc = "SHOULD_CHANGE" if expected_change else "SHOULD_FREEZE"
                            
                            print(f"    Round {prev_round}→{round_num}: {status} {change_desc} (diff={max_diff:.2e}) - {expected_desc}")
                            
                            changes.append({
                                'from': prev_round,
                                'to': round_num, 
                                'changed': actually_changed,
                                'expected': expected_change,
                                'diff': max_diff
                            })
                    
                    prev_round = round_num
            
            # Summary for this parameter
            unexpected_changes = [c for c in changes if c['changed'] != c['expected']]
            if unexpected_changes:
                print(f"    ❌ {len(unexpected_changes)} unexpected changes!")
                for change in unexpected_changes:
                    print(f"      Round {change['from']}→{change['to']}: "
                          f"{'changed' if change['changed'] else 'unchanged'} but "
                          f"{'should change' if change['expected'] else 'should freeze'}")
            else:
                print(f"    ✅ All changes follow expected pattern")
    
    # Overall summary
    print(f"\n🎯 SUMMARY:")
    print(f"Expected training regime:")
    print(f"  - Task 0 params: change at round 0, then freeze")
    print(f"  - Task 1 params: change at round 1, then freeze") 
    print(f"  - Task i params: change at round i, then freeze")
    print(f"  - etc.")
    
    # Check if ANY parameters follow the expected pattern
    total_violations = 0
    total_checks = 0
    
    for task_id in range(8):
        first_round = min(rounds_data.keys())
        task_keys = [k for k in rounds_data[first_round].keys() 
                    if f"tasks.{task_id}." in k and torch.is_tensor(rounds_data[first_round][k])]
        
        for key in task_keys:
            prev_round = None
            for round_num in sorted(rounds_data.keys()):
                if prev_round is not None and key in rounds_data[round_num] and key in rounds_data[prev_round]:
                    current_tensor = rounds_data[round_num][key]
                    prev_tensor = rounds_data[prev_round][key]
                    
                    if prev_tensor.shape == current_tensor.shape:
                        max_diff = torch.max(torch.abs(prev_tensor - current_tensor)).item()
                        expected_change = (round_num == task_id)
                        actually_changed = (max_diff > 1e-6)
                        
                        total_checks += 1
                        if expected_change != actually_changed:
                            total_violations += 1
                
                prev_round = round_num
    
    if total_checks > 0:
        violation_rate = total_violations / total_checks
        print(f"\nViolation rate: {total_violations}/{total_checks} = {violation_rate:.1%}")
        
        if violation_rate > 0.5:
            print(f"🚨 CRITICAL: Training regime is completely broken!")
            print(f"   Likely causes:")
            print(f"   1. Parameter freezing/unfreezing logic is wrong")
            print(f"   2. Saving is overwriting with wrong parameters")
            print(f"   3. DeepSpeed is interfering with parameter states")
        elif violation_rate > 0.1:
            print(f"⚠️  WARNING: Training regime has issues")
        else:
            print(f"✅ Training regime mostly working")
    
    return total_violations == 0

if __name__ == "__main__":
    success = debug_intervention_saving()
    if success:
        print(f"\n✅ Intervention saving/training follows expected regime")
    else:
        print(f"\n❌ Intervention training regime is broken - needs fixing")
