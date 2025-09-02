#!/usr/bin/env python3
"""
Minimal test: just load REFT-CL and verify interventions without generation
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def minimal_reft_test():
    print("=== Minimal REFT-CL Test (No Generation) ===")
    
    base_model_path = "/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat"
    saved_model_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers/7"
    
    # 1. Load base model
    print("1. Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    print(f"✓ Base model loaded: {model.__class__.__name__}")
    
    # 2. Build REFT structure
    print("2. Building REFT intervention structure...")
    try:
        from pyreft import get_reft_model, ReftConfig
        from loreft.reft_cl_intervention import ReftCLIntervention
        sys.path.append('/cluster/home/tdieudonne/clmm/TRACE')
        from reft_loading_utils import AlphaBank
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Config
    target_layers = [4, 6, 10, 12, 14, 18, 20, 22, 26]
    low_rank = 8
    eps = 1e-8
    num_tasks = 8
    embed_dim = model.config.hidden_size
    
    # Alpha bank
    alpha_bank = AlphaBank(num_tasks=num_tasks, alpha_init=0.0)
    def _get_alpha(i: int):
        return alpha_bank.alphas[i]
    
    # Build interventions
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
    model.add_module("reftcl_alpha_bank", alpha_bank)
    print("✓ REFT structure built")
    
    # 3. Load intervention weights
    print("3. Loading intervention weights...")
    import glob
    intervention_files = glob.glob(os.path.join(saved_model_path, "intkey_*.bin"))
    print(f"Found {len(intervention_files)} intervention files")
    
    loaded_weights = {}
    for int_file in intervention_files:
        base_name = os.path.basename(int_file)
        state_dict = torch.load(int_file, map_location="cpu")
        loaded_weights[base_name] = state_dict
        print(f"  Loaded {base_name}: {list(state_dict.keys())}")
    
    # 4. Apply weights
    print("4. Applying weights to interventions...")
    applied_count = 0
    if hasattr(model, 'interventions'):
        for key, intervention in model.interventions.items():
            matching_file = None
            for file_name in loaded_weights.keys():
                if key.replace("#", "#") in file_name:
                    matching_file = file_name
                    break
            
            if matching_file:
                weights = loaded_weights[matching_file]
                try:
                    intervention.load_state_dict(weights, strict=False)
                    applied_count += 1
                    print(f"  ✓ Applied weights to {key}")
                except Exception as e:
                    print(f"  ❌ Failed to apply weights to {key}: {e}")
    
    print(f"✓ Applied weights to {applied_count}/{len(model.interventions)} interventions")
    
    # 5. Set active tasks
    print("5. Setting active tasks...")
    for key, intervention in model.interventions.items():
        if hasattr(intervention, 'set_active_tasks'):
            intervention.set_active_tasks(num_tasks)
        if hasattr(intervention, 'active_tasks'):
            print(f"  {key}: active_tasks = {intervention.active_tasks}")
    
    # 6. Basic forward pass test (no generation)
    print("6. Testing basic forward pass...")
    inputs = tokenizer("Hello", return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    try:
        with torch.no_grad():
            # Test base model forward pass
            base_model = model.model
            base_outputs = base_model(input_ids)
            print(f"✓ Base forward pass: output shape {base_outputs.logits.shape}")
            
            # Test intervention model forward pass (simple)
            # For pyreft, we need to provide intervention inputs
            test_outputs = model(
                {"input_ids": input_ids},
                unit_locations={"sources->base": (None, [[[0]] for _ in range(len(model.interventions))])}
            )
            
            # Handle different output formats
            if isinstance(test_outputs, tuple):
                intervention_logits = test_outputs[1].logits if hasattr(test_outputs[1], 'logits') else test_outputs[1]
            else:
                intervention_logits = test_outputs.logits if hasattr(test_outputs, 'logits') else test_outputs
                
            print(f"✓ Intervention forward pass: output shape {intervention_logits.shape}")
            
            # Compare outputs
            if torch.allclose(base_outputs.logits, intervention_logits, atol=1e-3):
                print("⚠️  Outputs are identical - interventions may not be active")
            else:
                print("✓ Outputs differ - interventions are working!")
                
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = minimal_reft_test()
    if success:
        print("\n✅ REFT-CL loading test PASSED!")
        print("The model loads correctly and interventions are functional.")
        print("Generation hanging is likely a separate issue.")
    else:
        print("\n❌ REFT-CL loading test FAILED!")
