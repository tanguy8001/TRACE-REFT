#!/usr/bin/env python3
"""
Convert old intervention files to the format that pyreft expects.
This handles the mismatch between our saved format and pyreft's expected format.
"""
import torch
import os
import sys
sys.path.append('/cluster/home/tdieudonne/clmm/TRACE')

def convert_intervention_file(old_file_path, new_file_path):
    """Convert an old intervention file to pyreft-compatible format."""
    
    print(f"Converting {old_file_path} -> {new_file_path}")
    
    # Load the old format
    old_state = torch.load(old_file_path, map_location="cpu")
    print(f"Old format keys: {list(old_state.keys())}")
    
    # Create new format
    new_state = {}
    
    # Convert task parameters
    for key, value in old_state.items():
        if key.startswith("tasks."):
            # Extract task number and parameter name
            parts = key.split(".")
            if len(parts) >= 3:
                task_num = parts[1]
                param_name = parts[2]
                
                if param_name == "rotate_layer":
                    # Convert rotate_layer to pyreft's expected format
                    new_key = f"tasks.{task_num}.rotate_layer.parametrizations.weight.original"
                    new_state[new_key] = value.to(torch.bfloat16)  # Match the model's dtype
                elif param_name == "weight":
                    # Convert weight to learned_source.weight
                    new_key = f"tasks.{task_num}.learned_source.weight"
                    new_state[new_key] = value
                elif param_name == "bias":
                    # Convert bias to learned_source.bias
                    new_key = f"tasks.{task_num}.learned_source.bias"
                    new_state[new_key] = value
        else:
            # Keep non-task parameters as-is
            new_state[key] = value
    
    print(f"New format keys: {list(new_state.keys())}")
    
    # Save the new format
    torch.save(new_state, new_file_path)
    print(f"Saved converted file to {new_file_path}")

def convert_all_interventions(base_dir, round_num):
    """Convert all intervention files for a given round."""
    
    round_dir = os.path.join(base_dir, str(round_num))
    if not os.path.exists(round_dir):
        print(f"Round directory not found: {round_dir}")
        return
    
    # Find all intervention files
    intervention_files = [f for f in os.listdir(round_dir) if f.startswith("intkey_") and f.endswith(".bin")]
    
    for int_file in intervention_files:
        old_path = os.path.join(round_dir, int_file)
        new_path = os.path.join(round_dir, f"converted_{int_file}")
        
        try:
            convert_intervention_file(old_path, new_path)
        except Exception as e:
            print(f"Error converting {int_file}: {e}")

if __name__ == "__main__":
    # Convert round 2 interventions
    base_dir = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers"
    convert_all_interventions(base_dir, 2)
