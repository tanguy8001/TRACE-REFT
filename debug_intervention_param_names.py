#!/usr/bin/env python3
"""
Investigate intervention parameter names in a trained REFT-CL model
to understand how they appear in model.named_parameters()
"""
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def investigate_param_names():
    """Load a REFT-CL model and inspect parameter names"""
    
    print("=== Investigating REFT-CL Parameter Names ===")
    
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
    
    # 2. Build REFT structure (like we do in training)
    print("2. Building REFT intervention structure...")
    try:
        from pyreft import get_reft_model, ReftConfig
        from loreft.reft_cl_intervention import ReftCLIntervention
        sys.path.append('/cluster/home/tdieudonne/clmm/TRACE')
        from reft_loading_utils import AlphaBank
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Config matching training
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
    
    # 3. Analyze all parameter names
    print("\n3. Analyzing parameter names...")
    
    all_params = list(model.named_parameters())
    print(f"Total parameters: {len(all_params)}")
    
    # Categorize parameters
    alpha_params = []
    intervention_params = []
    base_params = []
    other_params = []
    
    for name, param in all_params:
        if "alpha" in name.lower():
            alpha_params.append((name, param.shape, param.requires_grad))
        elif "intervention" in name.lower():
            intervention_params.append((name, param.shape, param.requires_grad))
        elif name.startswith("model."):
            base_params.append((name, param.shape, param.requires_grad))
        else:
            other_params.append((name, param.shape, param.requires_grad))
    
    print(f"\n📊 Parameter Categories:")
    print(f"  Alpha parameters: {len(alpha_params)}")
    print(f"  Intervention parameters: {len(intervention_params)}")
    print(f"  Base model parameters: {len(base_params)}")
    print(f"  Other parameters: {len(other_params)}")
    
    # Show alpha parameters
    if alpha_params:
        print(f"\n🔸 Alpha Parameters ({len(alpha_params)}):")
        for name, shape, req_grad in alpha_params:
            print(f"  {name}: {shape} (requires_grad={req_grad})")
    else:
        print(f"\n❌ No alpha parameters found in named_parameters()")
    
    # Show intervention parameters  
    if intervention_params:
        print(f"\n🔸 Intervention Parameters ({len(intervention_params)}):")
        for name, shape, req_grad in intervention_params[:20]:  # Show first 20
            print(f"  {name}: {shape} (requires_grad={req_grad})")
        if len(intervention_params) > 20:
            print(f"  ... and {len(intervention_params) - 20} more")
    else:
        print(f"\n❌ No intervention parameters found in named_parameters()")
    
    # Show other parameters (might include interventions under different names)
    if other_params:
        print(f"\n🔸 Other Parameters ({len(other_params)}):")
        for name, shape, req_grad in other_params[:10]:  # Show first 10
            print(f"  {name}: {shape} (requires_grad={req_grad})")
        if len(other_params) > 10:
            print(f"  ... and {len(other_params) - 10} more")
    
    # 4. Check direct access to interventions
    print(f"\n4. Direct intervention access:")
    if hasattr(model, 'interventions'):
        print(f"  model.interventions keys: {list(model.interventions.keys())}")
        
        # Look at first intervention structure
        first_key = list(model.interventions.keys())[0]
        first_intervention = model.interventions[first_key]
        print(f"  First intervention ({first_key}):")
        print(f"    Type: {type(first_intervention).__name__}")
        
        if hasattr(first_intervention, 'named_parameters'):
            intervention_named_params = list(first_intervention.named_parameters())
            print(f"    Named parameters: {len(intervention_named_params)}")
            for name, param in intervention_named_params[:5]:
                print(f"      {name}: {param.shape} (requires_grad={param.requires_grad})")
    
    # 5. Search for intervention-like patterns
    print(f"\n5. Searching for intervention patterns...")
    
    intervention_patterns = []
    for name, param in all_params:
        # Look for patterns that might be interventions
        if any(pattern in name.lower() for pattern in ['rotate', 'weight', 'bias']) and 'layer_' in name:
            intervention_patterns.append((name, param.shape, param.requires_grad))
        elif 'tasks' in name.lower():
            intervention_patterns.append((name, param.shape, param.requires_grad))
    
    if intervention_patterns:
        print(f"  Found intervention-like patterns ({len(intervention_patterns)}):")
        for name, shape, req_grad in intervention_patterns[:15]:
            print(f"    {name}: {shape} (requires_grad={req_grad})")
    else:
        print(f"  No intervention-like patterns found")
    
    # 6. Check what patterns would match in optimizer
    print(f"\n6. Testing optimizer matching patterns...")
    
    # Current intervention patterns from utils.py
    intervention_pattern_prefixes = ("interventions.", "module.interventions.")
    
    matching_params = []
    for name, param in all_params:
        if any(name.startswith(prefix) for prefix in intervention_pattern_prefixes):
            matching_params.append((name, param.shape, param.requires_grad))
    
    if matching_params:
        print(f"  Current patterns match {len(matching_params)} parameters:")
        for name, shape, req_grad in matching_params[:10]:
            print(f"    {name}: {shape} (requires_grad={req_grad})")
    else:
        print(f"  ❌ Current patterns ('interventions.', 'module.interventions.') match 0 parameters")
        print(f"  This explains why interventions aren't in optimizer!")
    
    return intervention_patterns, matching_params

if __name__ == "__main__":
    investigate_param_names()
