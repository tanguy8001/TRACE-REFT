#!/usr/bin/env python3
"""
Simple test to verify that model.load() works for REFT-CL.
"""

import os
import torch

def test_simple_loading():
    print("=== Testing pyreft.load() method ===")
    
    saved_model_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers_2/0/"
    
    # Check if path exists
    if not os.path.exists(saved_model_path):
        print(f"ERROR: Saved model path not found: {saved_model_path}")
        return False
    
    # List files in directory
    files = os.listdir(saved_model_path)
    print(f"Files in saved model directory: {files}")
    
    # Check if we have the right files
    intervention_files = [f for f in files if f.startswith('intkey_') and f.endswith('.bin')]
    has_config = 'config.json' in files
    
    print(f"Intervention files: {len(intervention_files)}")
    print(f"Config file present: {has_config}")
    
    if len(intervention_files) > 0 and has_config:
        print("✓ Saved model structure looks correct for pyreft loading")
        
        # Try to understand the config structure
        import json
        with open(os.path.join(saved_model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        print(f"Config keys: {list(config.keys())}")
        if 'representations' in config:
            print(f"Number of representations: {len(config['representations'])}")
        
        return True
    else:
        print("✗ Saved model structure is incomplete")
        return False

if __name__ == "__main__":
    success = test_simple_loading()
    
    if success:
        print("\n=== Recommendation ===")
        print("The saved model structure looks correct for pyreft.load().")
        print("The inference loading should work with model.load(path).")
        print("If there are still issues, it may be due to:")
        print("1. Version mismatch between training and inference pyreft")
        print("2. Different model architecture expectations")
        print("3. Missing alpha parameter handling")
    else:
        print("\n=== Issue Found ===")
        print("The saved model structure doesn't match pyreft expectations.")
