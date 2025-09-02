"""
Utility functions for REFT-CL model loading to avoid circular imports.
"""
import torch
from torch import nn


class AlphaBank(nn.Module):
    """Standalone AlphaBank class to avoid circular imports."""
    
    def __init__(self, num_tasks: int, alpha_init: float = 0.1):
        super().__init__()
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32)) for _ in range(num_tasks)]
        )


def load_reft_cl_model(base_model, saved_model_path, reft_config):
    """
    Load a REFT-CL model with proper intervention weights.
    
    Args:
        base_model: Base transformer model
        saved_model_path: Path to saved REFT-CL model
        reft_config: Dict with keys: target_layers, low_rank, eps, num_tasks
    
    Returns:
        Loaded REFT model
    """
    try:
        from pyreft import get_reft_model, ReftConfig
    except Exception:
        from pyreft.utils import get_reft_model  # type: ignore
        from pyreft import ReftConfig  # type: ignore
    from loreft.reft_cl_intervention import ReftCLIntervention
    
    # Extract config
    target_layers = reft_config['target_layers']
    low_rank = reft_config['low_rank']
    eps = reft_config['eps']
    num_tasks = reft_config['num_tasks']
    embed_dim = base_model.config.hidden_size
    
    # Build alpha bank
    alpha_bank = AlphaBank(num_tasks=num_tasks, alpha_init=0.0)
    def _get_alpha(i: int):
        return alpha_bank.alphas[i]
    
    # Build intervention config
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
    model = get_reft_model(base_model, cfg, set_device=False)
    
    # Attach alpha bank
    if not hasattr(model, "reftcl_alpha_bank"):
        model.add_module("reftcl_alpha_bank", alpha_bank)
    
    # Load intervention weights with proper device mapping
    try:
        model.load_intervention(saved_model_path)
    except RuntimeError as e:
        if "CUDA device but torch.cuda.is_available() is False" in str(e):
            print("Attempting to fix CUDA/CPU device mapping issue...")
            # Manually load intervention files with CPU mapping and patch them
            import glob
            import os
            import tempfile
            import shutil
            
            # Create a temporary directory with CPU-mapped intervention files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy config file
                config_file = os.path.join(saved_model_path, "config.json")
                if os.path.exists(config_file):
                    shutil.copy2(config_file, temp_dir)
                
                # Load and re-save intervention files with CPU mapping
                intervention_files = glob.glob(os.path.join(saved_model_path, "intkey_*.bin"))
                for int_file in intervention_files:
                    # Load with CPU mapping
                    state_dict = torch.load(int_file, map_location="cpu")
                    # Save to temp directory
                    temp_file = os.path.join(temp_dir, os.path.basename(int_file))
                    torch.save(state_dict, temp_file)
                
                # Now try loading from the temp directory
                model.load_intervention(temp_dir)
        else:
            raise
    
    return model
