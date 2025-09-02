#!/usr/bin/env python3
"""
Test script to verify REFT-CL model loading works correctly.
"""

import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add TRACE to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_reft_loading():
    print("=== Testing REFT-CL Model Loading ===")
    
    # Test parameters (from your training)
    base_model_path = "/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat"
    saved_model_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers_2/0/"
    
    # Check if paths exist
    if not os.path.exists(base_model_path):
        print(f"ERROR: Base model path not found: {base_model_path}")
        return False
        
    if not os.path.exists(saved_model_path):
        print(f"ERROR: Saved model path not found: {saved_model_path}")
        return False
    
    try:
        # Step 1: Load base model
        print("1. Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cpu",
            local_files_only=True
        )
        print("   ✓ Base model loaded successfully")
        
        # Step 2: Set up REFT interventions (similar to inference code)
        print("2. Setting up REFT interventions...")
        from pyreft import get_reft_model, ReftConfig
        from loreft.reft_cl_intervention import ReftCLIntervention
        from model.ReFTCL import AlphaBank
        
        # Configuration (should match training)
        target_layers = [4, 6, 10, 12, 14, 18, 20, 22, 26]  # inferred from saved files
        low_rank = 8  # from folder name REFT-CL_rank8
        eps = 1e-6
        embed_dim = model.config.hidden_size
        num_tasks = 8  # typical
        
        # Build alpha bank
        alpha_bank = AlphaBank(num_tasks=num_tasks, alpha_init=0.0)
        def _get_alpha(i: int):
            return alpha_bank.alphas[i]
        
        # Build representations
        reps = []
        for l in target_layers:
            reps.append({
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": low_rank,
                "intervention": ReftCLIntervention(
                    embed_dim=embed_dim,
                    low_rank_dimension=low_rank,
                    num_tasks=num_tasks,
                    get_alpha=_get_alpha,
                    eps=eps,
                    dtype=torch.float32,
                )
            })
        
        cfg = ReftConfig(representations=reps)
        model = get_reft_model(model, cfg, set_device=False)
        
        # Attach alpha bank
        if not hasattr(model, "reftcl_alpha_bank"):
            model.add_module("reftcl_alpha_bank", alpha_bank)
        print("   ✓ REFT interventions set up successfully")
        
        # Step 3: Load saved weights
        print("3. Loading saved intervention weights...")
        try:
            model.load(saved_model_path)
            print("   ✓ Weights loaded successfully using pyreft.load()")
        except Exception as e:
            print(f"   ✗ pyreft.load() failed: {e}")
            print("   → This is expected if pyreft.load() doesn't work as expected")
            return False
        
        # Step 4: Basic sanity check
        print("4. Running basic sanity check...")
        test_input = "Hello, world!"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            # Test that the model can run forward pass
            base_inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
            outputs = model(base_inputs)
            print(f"   ✓ Forward pass successful, output shape: {outputs.logits.shape}")
        
        print("\n=== REFT-CL Loading Test PASSED ===")
        return True
        
    except Exception as e:
        print(f"\n=== REFT-CL Loading Test FAILED ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reft_loading()
    sys.exit(0 if success else 1)
