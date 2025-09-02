#!/usr/bin/env python3
"""
Simple test to load REFT-CL model: frozen llama + alphas + interventions
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_reft_cl_simple(base_model_path, saved_model_path):
    """
    Simple REFT-CL loading: base model + rebuild interventions + load weights
    """
    # 1. Load base frozen model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Force CPU to avoid CUDA issues
    )
    print(f"✓ Base model loaded: {model.__class__.__name__}")
    
    # 2. Build REFT intervention structure 
    print("Building REFT intervention structure...")
    try:
        from pyreft import get_reft_model, ReftConfig
        from loreft.reft_cl_intervention import ReftCLIntervention
        sys.path.append('/cluster/home/tdieudonne/clmm/TRACE')
        from reft_loading_utils import AlphaBank
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return None
    
    # Training config
    target_layers = [4, 6, 10, 12, 14, 18, 20, 22, 26]
    low_rank = 8
    eps = 1e-8
    num_tasks = 8
    embed_dim = model.config.hidden_size
    
    # Build alpha bank
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
    
    # 3. Manually load intervention weights with CPU mapping
    print("Loading intervention weights...")
    import glob
    intervention_files = glob.glob(os.path.join(saved_model_path, "intkey_*.bin"))
    print(f"Found {len(intervention_files)} intervention files")
    
    if not intervention_files:
        print("❌ No intervention files found")
        return None
    
    # Load each intervention file manually
    loaded_weights = {}
    for int_file in intervention_files:
        # Extract layer info from filename
        base_name = os.path.basename(int_file)
        print(f"Loading {base_name}...")
        
        # Load with CPU mapping
        state_dict = torch.load(int_file, map_location="cpu")
        print(f"  Keys: {list(state_dict.keys())}")
        
        # Store for later application
        loaded_weights[base_name] = state_dict
    
    print("✓ All intervention weights loaded")
    
    # 4. Apply weights to interventions
    print("Applying weights to model interventions...")
    
    # Get intervention keys from model
    if hasattr(model, 'interventions'):
        print(f"Model has {len(model.interventions)} interventions")
        for key, intervention in model.interventions.items():
            print(f"  Intervention key: {key}")
            
            # Find matching loaded weights
            matching_file = None
            for file_name in loaded_weights.keys():
                if key.replace("#", "#") in file_name:
                    matching_file = file_name
                    break
            
            if matching_file:
                print(f"  Loading weights from {matching_file}")
                weights = loaded_weights[matching_file]
                
                # Apply weights to intervention's state dict
                try:
                    intervention.load_state_dict(weights, strict=False)
                    print(f"  ✓ Weights applied to {key}")
                except Exception as e:
                    print(f"  ❌ Failed to apply weights to {key}: {e}")
            else:
                print(f"  ⚠️  No weights found for {key}")
    else:
        print("❌ Model has no interventions")
        return None
    
    # Set active tasks for all interventions (important for REFT-CL)
    print("Setting active tasks for interventions...")
    if hasattr(model, 'interventions'):
        for key, intervention in model.interventions.items():
            if hasattr(intervention, 'set_active_tasks'):
                intervention.set_active_tasks(num_tasks)  # Activate all tasks
                print(f"  Set active tasks to {num_tasks} for {key}")
            if hasattr(intervention, 'active_tasks'):
                print(f"  {key} active_tasks: {intervention.active_tasks}")
    
    print("✓ REFT-CL model loaded successfully!")
    return model

def test_model_generation(model, tokenizer):
    """Test that the loaded model can generate with proper REFT intervention"""
    print("\nTesting model generation...")
    
    # Debug intervention status first
    if hasattr(model, 'interventions'):
        print(f"Model has {len(model.interventions)} interventions")
        for key, intervention in model.interventions.items():
            if hasattr(intervention, 'active_tasks'):
                print(f"  {key}: active_tasks = {intervention.active_tasks}")
    
    prompt = "Hello"  # Start with shorter prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"Input: '{prompt}'")
    print(f"Input shape: {input_ids.shape}")
    print(f"Token IDs: {input_ids.tolist()}")
    
    try:
        with torch.no_grad():
            print("Attempting generation...")
            
            # Try simpler generation first - no interventions
            print("Testing without interventions (should work)...")
            base_model = model.model if hasattr(model, 'model') else model
            simple_output = base_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            simple_text = tokenizer.decode(simple_output[0], skip_special_tokens=True)
            print(f"✓ Base generation works: '{simple_text}'")
            
            # Now try with minimal intervention setup
            print("Testing with interventions...")
            seq_len = input_ids.shape[1]
            last_position = seq_len - 1
            num_interventions = len(model.interventions)
            
            # Simpler unit_locations - just one position per intervention
            unit_locations = [[[last_position]] for _ in range(num_interventions)]
            
            print(f"Unit locations: {len(unit_locations)} interventions, position {last_position}")
            
            generation_args = {
                "base": {"input_ids": input_ids, "attention_mask": attention_mask},
                "unit_locations": {"sources->base": (None, unit_locations)},
                "intervene_on_prompt": True,
                "max_new_tokens": 3,  # Very small to avoid hanging
                "do_sample": False,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.eos_token_id,
            }
            
            print("Calling model.generate with interventions...")
            result = model.generate(**generation_args)
            print(f"Generation returned: {type(result)}")
            
            if isinstance(result, tuple) and len(result) >= 2:
                base_output, intervention_output = result
                print(f"Base output shape: {base_output.shape if hasattr(base_output, 'shape') else 'no shape'}")
                print(f"Intervention output shape: {intervention_output.shape if hasattr(intervention_output, 'shape') else 'no shape'}")
                generated_ids = intervention_output
            else:
                generated_ids = result
                
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            print(f"✓ Intervention generation works: '{generated_text}'")
            return True
            
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Simple REFT-CL Loading Test ===")
    
    base_model_path = "/cluster/scratch/tdieudonne/initial_model/llama-2-7b-chat"
    saved_model_path = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL_rank8_9layers/7"
    
    # Test loading
    model = load_reft_cl_simple(base_model_path, saved_model_path)
    
    if model is not None:
        # Test generation
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        success = test_model_generation(model, tokenizer)
        if success:
            print("\n✅ REFT-CL loading and generation successful!")
        else:
            print("\n❌ Generation test failed")
    else:
        print("\n❌ REFT-CL loading failed")
