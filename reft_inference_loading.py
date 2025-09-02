#!/usr/bin/env python3
"""
REFT-CL inference loading utilities for task-structured intervention files.
This handles the new format where interventions are saved with task prefixes (tasks.0.weight, etc.)
"""
import os
import torch
import json
from typing import Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyreft import get_reft_model, ReftConfig
from loreft.reft_cl_intervention import ReftCLIntervention

class AlphaBank(torch.nn.Module):
    """Standalone AlphaBank for inference loading to avoid circular imports."""
    
    def __init__(self, num_tasks: int, alpha_init: float = 0.05):
        super().__init__()
        self.alphas = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32)) for _ in range(num_tasks)]
        )

def load_reft_cl_model_for_inference(
    base_model_path: str,
    saved_model_path: str, 
    reft_config: Dict[str, Any],
    tokenizer=None
):
    """
    Load a REFT-CL model for inference with the new task-structured intervention files.
    
    Args:
        base_model_path: Path to the base model (e.g., llama-2-7b-chat)
        saved_model_path: Path to the saved REFT-CL model directory
        reft_config: Configuration dict with reft_layers, reft_rank, reft_eps, num_tasks
        tokenizer: Optional tokenizer (will load if not provided)
    
    Returns:
        Loaded REFT-CL model ready for inference
    """
    
    print(f"[REFT-CL] Loading base model from {base_model_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Extract config
    target_layers = reft_config['reft_layers']
    if target_layers.strip() == "all":
        num_layers = base_model.config.num_hidden_layers
        target_layers = list(range(num_layers))
    else:
        target_layers = [int(x) for x in target_layers.split(";") if len(x) > 0]
    
    low_rank = reft_config['reft_rank']
    eps = reft_config['reft_eps']
    num_tasks = reft_config['num_tasks']
    embed_dim = base_model.config.hidden_size
    
    print(f"[REFT-CL] Building intervention structure:")
    print(f"  - Layers: {target_layers}")
    print(f"  - Rank: {low_rank}")
    print(f"  - Tasks: {num_tasks}")
    print(f"  - Embed dim: {embed_dim}")
    
    # Create alpha bank
    alpha_bank = AlphaBank(num_tasks=num_tasks, alpha_init=0.0)
    
    def _get_alpha(i: int):
        return alpha_bank.alphas[i]
    
    # Build intervention representations
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
                dtype=torch.bfloat16,  # Match the base model's dtype
            )
        })
    
    # Create REFT model
    cfg = ReftConfig(representations=reps)
    reft_model = get_reft_model(base_model, cfg, set_device=False)
    
    # Attach alpha bank
    if not hasattr(reft_model, "reftcl_alpha_bank"):
        reft_model.add_module("reftcl_alpha_bank", alpha_bank)
    
    print(f"[REFT-CL] Loading intervention weights from {saved_model_path}")
    
    # Load intervention weights with task structure
    base_ref = reft_model.module if hasattr(reft_model, "module") else reft_model
    
    for key, intervention in getattr(base_ref, "interventions", {}).items():
        # Try converted file first, then fall back to original
        converted_file = os.path.join(saved_model_path, f"converted_intkey_{key}.bin")
        original_file = os.path.join(saved_model_path, f"intkey_{key}.bin")
        
        if os.path.exists(converted_file):
            intervention_file = converted_file
            print(f"[REFT-CL] Using converted intervention file: {intervention_file}")
        else:
            intervention_file = original_file
        
        if not os.path.exists(intervention_file):
            print(f"[REFT-CL] WARNING: Intervention file not found: {intervention_file}")
            continue
            
        print(f"[REFT-CL] Loading intervention {key} from {intervention_file}")
        
        # Load the task-structured state dict
        intervention_state = torch.load(intervention_file, map_location="cpu")
        
        # Load into the intervention using our custom load_state_dict
        try:
            missing_keys, unexpected_keys = intervention.load_state_dict(intervention_state, strict=False)
            if missing_keys:
                print(f"[REFT-CL] Missing keys for {key}: {missing_keys}")
            if unexpected_keys:
                print(f"[REFT-CL] Unexpected keys for {key}: {unexpected_keys}")
            print(f"[REFT-CL] Successfully loaded {key} with {len(intervention_state)} parameters")
        except Exception as e:
            print(f"[REFT-CL] ERROR loading {key}: {e}")
            raise
    
    # Load alpha parameters
    alpha_file = os.path.join(saved_model_path, "reftcl_alphas.bin")
    if os.path.exists(alpha_file):
        print(f"[REFT-CL] Loading alpha parameters from {alpha_file}")
        alpha_state = torch.load(alpha_file, map_location="cpu")
        
        for i, alpha_param in enumerate(reft_model.reftcl_alpha_bank.alphas):
            key = f"alpha_{i}"
            if key in alpha_state:
                alpha_param.data.copy_(alpha_state[key])
                print(f"[REFT-CL] Loaded alpha_{i} = {alpha_state[key].item():.6f}")
            else:
                print(f"[REFT-CL] WARNING: No alpha_{i} found in {alpha_file}")
        
        print(f"[REFT-CL] Loaded {len(alpha_state)} alpha parameters")
    else:
        print(f"[REFT-CL] WARNING: No alpha file found at {alpha_file}")
    
    # Set to eval mode
    reft_model.eval()
    for intervention in getattr(base_ref, "interventions", {}).values():
        intervention.eval()
    
    print(f"[REFT-CL] Model loaded successfully and ready for inference")
    return reft_model, tokenizer

def load_reft_config_from_saved_model(saved_model_path: str) -> Dict[str, Any]:
    """Load REFT configuration from a saved model directory."""
    config_file = os.path.join(saved_model_path, "reft_config.json")
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"[REFT-CL] Loaded config from {config_file}: {config}")
        return config
    else:
        print(f"[REFT-CL] No config file found at {config_file}, using defaults")
        return {
            'reft_layers': '3;9;18;24',
            'reft_rank': 4,
            'reft_eps': 1e-6,
            'num_tasks': 8  # Default to 8 tasks
        }
