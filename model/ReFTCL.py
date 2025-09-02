import os
import torch
from typing import Dict

from model.base_model import CL_Base_Model
from utils.utils import print_rank_0

try:
    from pyreft import get_reft_model, ReftConfig  # modern API
except Exception:
    try:
        print("Using older pyreft API")
        from pyreft.utils import get_reft_model  # older API
        from pyreft import ReftConfig
    except Exception as e:
        raise ImportError("pyreft get_reft_model/ReftConfig not found; please update pyreft") from e
from loreft.reft_cl_intervention import ReftCLIntervention
import types


class AlphaBank(torch.nn.Module):
    """Holds shared alphas (one per task) so all layers can reference them."""

    def __init__(self, num_tasks: int, alpha_init: float = 0.05):
        super().__init__()
        self.alphas = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32)) for _ in range(num_tasks)]
        )


class ReFTCL(CL_Base_Model):
    """
    CL trainer that composes ReFT interventions per task and shares
    a single alpha per task across all layers. It relies on a custom pyreft intervention
    that accumulates normalized edits online.
    """

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        self.tasks = list(self.train_task_list.keys())
        self.num_tasks = len(self.tasks)
        # shared across layers
        self.alpha_bank = AlphaBank(self.num_tasks, alpha_init=0.1)

        # Inject pyreft model wrapper with our intervention across selected layers
        # We do this once here so parameters are registered before deepspeed init in main.py
        self._attach_reft_interventions()

    def _attach_reft_interventions(self):
        """Wrap self.model with pyreft get_reft_model using our intervention."""
  
        layer_str = self.args.reft_layers
        if layer_str.strip() == "all":
            num_layers = self.model.config.num_hidden_layers
            target_layers = list(range(num_layers))
        else:
            target_layers = [int(x) for x in layer_str.split(";") if len(x) > 0]

        low_rank = int(getattr(self.args, "reft_rank", 4))
        eps = float(getattr(self.args, "reft_eps", 1e-6))
        embed_dim = int(getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None))

        reps = []
        # Provide a getter to share one alpha per task across all layers without duplicating params
        def _get_alpha(i: int):
            return self.alpha_bank.alphas[i]
        for l in target_layers:
            reps.append({
                "layer": l,
                # Use generic block_output to avoid model-specific module paths
                "component": "block_output",
                "low_rank_dimension": low_rank,
                "intervention": ReftCLIntervention(
                    embed_dim=embed_dim,
                    low_rank_dimension=low_rank,
                    num_tasks=self.num_tasks,
                    get_alpha=_get_alpha,
                    eps=eps,
                    dtype=torch.float32,
                )
            })

        cfg = ReftConfig(representations=reps)
        # Replace self.model with reft model in-place
        self.model = get_reft_model(self.model, cfg, set_device=False)
        self.model.print_trainable_parameters()
        n_params = self.model.count_parameters(include_model=False)
        n_params_with_model = self.model.count_parameters(include_model=True)
        print(f"Number of trainable parameters with frozen model: {n_params_with_model}. Number of trainable parameters without frozen model: {n_params}.")

        # By default, activate zero tasks (no edit) until training starts
        base_ref = self.model.module if hasattr(self.model, "module") else self.model
        for inter in getattr(base_ref, "interventions", {}).values():
            if hasattr(inter, "set_active_tasks"):
                inter.set_active_tasks(0)

        # Register alpha bank once at the top-level model so params appear only once
        if not hasattr(self.model, "reftcl_alpha_bank"):
            self.model.add_module("reftcl_alpha_bank", self.alpha_bank)
        # Ensure alpha params appear in named_parameters() prior to DeepSpeed init
        try:
            base_model = self.model
            original_named_parameters = base_model.named_parameters
            def _named_parameters_with_alpha(self_obj, *args, **kwargs):
                seen = set()
                for n, p in original_named_parameters(*args, **kwargs):
                    seen.add(n)
                    yield n, p
                # Derive prefix from args/kwargs if present
                prefix = ""
                try:
                    if 'prefix' in kwargs and isinstance(kwargs['prefix'], str):
                        prefix = kwargs['prefix']
                    elif len(args) >= 1 and isinstance(args[0], str):
                        prefix = args[0]
                except Exception:
                    prefix = ""
                try:
                    alpha_bank = getattr(self_obj, 'reftcl_alpha_bank', None)
                    if alpha_bank is not None and hasattr(alpha_bank, 'alphas'):
                        for i, p in enumerate(alpha_bank.alphas):
                            base_name = f"reftcl_alpha_bank.alphas.{i}"
                            full_name = f"{prefix}.{base_name}" if prefix else base_name
                            if full_name not in seen:
                                yield full_name, p
                except Exception:
                    pass
            base_model.named_parameters = types.MethodType(_named_parameters_with_alpha, base_model)
            print("[DEBUG][ReFTCL] Patched model.named_parameters to include alpha bank params.")
        except Exception as e:
            print(f"[DEBUG][ReFTCL] Failed to patch named_parameters: {e}")
        # Debug: inspect parameter names after attaching alpha bank
        try:
            all_named = list(self.model.named_parameters())
            alpha_like = [n for n, _ in all_named if ("alpha" in n.lower() or "reft" in n.lower())]
            preview = ", ".join(alpha_like[:20]) + (" ..." if len(alpha_like) > 20 else "")
            print(f"[DEBUG][ReFTCL] params containing 'alpha' or 'reft' after attach: {preview if alpha_like else 'NONE'}")
            if hasattr(self.alpha_bank, 'alphas'):
                flags = [p.requires_grad for p in self.alpha_bank.alphas]
                print(f"[DEBUG][ReFTCL] alpha bank size={len(self.alpha_bank.alphas)}, requires_grad flags={flags}")
        except Exception as e:
            print(f"[DEBUG][ReFTCL] Error while listing named_parameters: {e}")

    def _set_active_round(self, t: int):
        # Enable tasks 1..t (1-indexed externally) => internally 0..t-1
        base_ref = self.model.module if hasattr(self.model, "module") else self.model
        for inter in getattr(base_ref, "interventions", {}).values():
            if hasattr(inter, "set_active_tasks"):
                inter.set_active_tasks(t)

    def _debug_verify_saved_interventions(self, save_dir: str, round_idx: int):
        """Debug method to verify intervention weights were actually saved."""
        try:
            import glob
            
            # Check what files were created
            saved_files = glob.glob(os.path.join(save_dir, "*"))
            saved_files = [os.path.basename(f) for f in saved_files]
            print_rank_0(f"[DEBUG][ReFTCL] Round {round_idx} saved files: {saved_files}", self.args.global_rank)
            
            # pyreft saves interventions in separate .bin files, not in pytorch_model.bin
            intervention_files = [f for f in saved_files if f.startswith("intkey_") and f.endswith(".bin")]
            print_rank_0(f"[DEBUG][ReFTCL] Found {len(intervention_files)} intervention files", self.args.global_rank)
            
            if intervention_files:
                # Load and inspect intervention files
                total_intervention_params = 0
                intervention_details = []
                
                for int_file in sorted(intervention_files):
                    int_path = os.path.join(save_dir, int_file)
                    int_state = torch.load(int_path, map_location="cpu")
                    
                    file_params = sum(v.numel() for v in int_state.values() if torch.is_tensor(v))
                    total_intervention_params += file_params
                    
                    # Get a few example parameters from this file
                    param_examples = []
                    for key, tensor in int_state.items():
                        if torch.is_tensor(tensor):
                            param_examples.append(f"{key}: {tuple(tensor.shape)}")
                    
                    intervention_details.append((int_file, file_params, len(int_state), param_examples[:3]))
                
                print_rank_0(f"[DEBUG][ReFTCL] Round {round_idx} intervention verification:", self.args.global_rank)
                print_rank_0(f"  - Intervention files found: {len(intervention_files)}", self.args.global_rank)
                print_rank_0(f"  - Total intervention parameters: {total_intervention_params:,}", self.args.global_rank)
                
                # Show details for each intervention file
                for file_name, params, num_keys, examples in intervention_details:
                    print_rank_0(f"  - {file_name}: {params:,} params, {num_keys} keys", self.args.global_rank)
                    for example in examples:
                        print_rank_0(f"    {example}", self.args.global_rank)
                
                # Verify expected parameter count
                expected_params = self._calculate_expected_intervention_params()
                if total_intervention_params != expected_params:
                    print_rank_0(f"[DEBUG][ReFTCL] WARNING: Expected {expected_params:,} intervention params, found {total_intervention_params:,}", self.args.global_rank)
                else:
                    print_rank_0(f"[DEBUG][ReFTCL] ✓ Intervention parameter count matches expected: {total_intervention_params:,}", self.args.global_rank)
            else:
                print_rank_0(f"[DEBUG][ReFTCL] WARNING: No intervention files found in {save_dir}", self.args.global_rank)
                
            # Also check for alpha parameters in config or separate files
            alpha_info = self._check_alpha_saving(save_dir)
            if alpha_info:
                print_rank_0(f"[DEBUG][ReFTCL] Alpha parameters: {alpha_info}", self.args.global_rank)
                
        except Exception as e:
            print_rank_0(f"[DEBUG][ReFTCL] Error during intervention verification: {e}", self.args.global_rank)

    def _check_alpha_saving(self, save_dir: str) -> str:
        """Check how alpha parameters are saved."""
        try:
            # Check if there's a separate file for alphas or if they're in config
            possible_files = ["pytorch_model.bin", "adapter_model.bin", "alpha_bank.bin"]
            for file_name in possible_files:
                file_path = os.path.join(save_dir, file_name)
                if os.path.exists(file_path):
                    state = torch.load(file_path, map_location="cpu")
                    alpha_keys = [k for k in state.keys() if "alpha" in k.lower()]
                    if alpha_keys:
                        return f"{len(alpha_keys)} alpha params in {file_name}"
            return "Alpha parameters not found in standard locations"
        except Exception:
            return "Error checking alpha parameters"

    def _calculate_expected_intervention_params(self) -> int:
        """Calculate expected number of intervention parameters for verification."""
        # Get intervention config
        layer_str = getattr(self.args, "reft_layers", "3;9;18;24")
        if layer_str.strip() == "all":
            num_layers = self.model.config.num_hidden_layers
            target_layers = list(range(num_layers))
        else:
            target_layers = [int(x) for x in layer_str.split(";") if len(x) > 0]
        
        low_rank = int(getattr(self.args, "reft_rank", 4))
        embed_dim = int(getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None))
        
        # Per task per layer: R (embed_dim x low_rank) + W (low_rank x embed_dim) + b (low_rank)
        params_per_task_per_layer = (embed_dim * low_rank) + (low_rank * embed_dim) + low_rank
        total_expected = len(target_layers) * self.num_tasks * params_per_task_per_layer
        
        print_rank_0(f"[DEBUG][ReFTCL] Expected intervention params calculation:", self.args.global_rank)
        print_rank_0(f"  - Layers: {len(target_layers)} (indices: {target_layers})", self.args.global_rank)
        print_rank_0(f"  - Tasks: {self.num_tasks}", self.args.global_rank)
        print_rank_0(f"  - Rank: {low_rank}, Embed dim: {embed_dim}", self.args.global_rank)
        print_rank_0(f"  - Params per task/layer: {params_per_task_per_layer:,}", self.args.global_rank)
        print_rank_0(f"  - Total expected: {total_expected:,}", self.args.global_rank)
        
        return total_expected

    def save_model(self, round):
        """Override base save to use pyreft's .save() method for intervention weights."""
        if self.args.output_dir is not None:
            print_rank_0('saving model to ' + self.args.output_dir + "/" + str(round) + '...', self.args.global_rank)

        if self.args.global_rank == 0:
            # Use pyreft's save method to properly save intervention weights
            save_dir = os.path.join(self.args.output_dir, str(round))
            os.makedirs(save_dir, exist_ok=True)
            
            # Get the underlying reft model (unwrap from DeepSpeed if needed)
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

            # DON'T use pyreft's save - it doesn't preserve our task structure!
            # Instead, manually save interventions with proper task structure
            for key, intervention in getattr(model_to_save, 'interventions', {}).items():
                print_rank_0(f'[DEBUG] Intervention {key} type: {type(intervention)}', self.args.global_rank)
                print_rank_0(f'[DEBUG] Has state_dict: {hasattr(intervention, "state_dict")}', self.args.global_rank)
                print_rank_0(f'[DEBUG] Has tasks: {hasattr(intervention, "tasks")}', self.args.global_rank)
                
                if hasattr(intervention, 'state_dict'):
                    # Use our custom state_dict that preserves task structure
                    intervention_state = intervention.state_dict()
                    print_rank_0(f'[DEBUG] State dict keys: {list(intervention_state.keys())}', self.args.global_rank)
                    intervention_file = os.path.join(save_dir, f"intkey_{key}.bin")
                    torch.save(intervention_state, intervention_file)
                    print_rank_0(f'Saved intervention {key} with {len(intervention_state)} keys to {intervention_file}', self.args.global_rank)
            
            # Save intervention config for loading later
            if hasattr(model_to_save, 'interventions'):
                config_data = {
                    'reft_layers': getattr(self.args, 'reft_layers', '4;6;10;12;14;18;20;22;26'),
                    'reft_rank': getattr(self.args, 'reft_rank', 8), 
                    'reft_eps': getattr(self.args, 'reft_eps', 1e-8),
                    'num_tasks': self.num_tasks
                }
                config_file = os.path.join(save_dir, "reft_config.json")
                import json
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
                print_rank_0(f'Saved REFT config to {config_file}', self.args.global_rank)
            
            # SEPARATELY save alpha parameters (pyreft doesn't know about them)
            alpha_bank = getattr(model_to_save, 'reftcl_alpha_bank', None)
            if alpha_bank is not None:
                alpha_file = os.path.join(save_dir, "reftcl_alphas.bin")
                alpha_state = {f"alpha_{i}": alpha_bank.alphas[i].data.clone() for i in range(len(alpha_bank.alphas))}
                torch.save(alpha_state, alpha_file)
                print_rank_0(f'Saved {len(alpha_state)} alpha parameters to {alpha_file}', self.args.global_rank)
            else:
                print_rank_0('WARNING: No alpha bank found to save', self.args.global_rank)
            
            print_rank_0(f'Successfully saved REFT-CL model with interventions to {save_dir}', self.args.global_rank)
            
            self._debug_verify_saved_interventions(save_dir, round)
 
        if self.args.zero_stage == 3:
            from utils.utils import save_zero_three_model
            save_zero_three_model(self.model,
                                  self.args.global_rank,
                                  self.args.output_dir,
                                  zero_stage=self.args.zero_stage,
                                  sub_folder=str(round))
        print_rank_0('Successfully saving model after round {}'.format(round), self.args.global_rank)

    def train_continual(self):
        # For each task round, unfreeze (R_t, W_t, b_t) across layers and set alphas 1..t trainable
        for i_task, task in enumerate(self.train_task_list):
            round_idx = i_task + 1
            print_rank_0(f"[REFT-CL] Begin round {round_idx}/{self.num_tasks} - task {task}", self.args.global_rank)

            # Freeze all directions first
            base_ref = self.model.module if hasattr(self.model, "module") else self.model
            for layer_key, inter in getattr(base_ref, "interventions", {}).items():
                if hasattr(inter, "tasks"):
                    print_rank_0(f"[REFT-CL] Round {round_idx}: Freezing all tasks in {layer_key}", self.args.global_rank)
                    for j in range(self.num_tasks):
                        block = inter.tasks[j]
                        param_count = sum(1 for p in block.parameters())
                        for p in block.parameters():
                            p.requires_grad = False
                        print_rank_0(f"  Task {j}: Froze {param_count} parameters", self.args.global_rank)

            # Unfreeze current round directions (R_t, W_t, b_t)
            for layer_key, inter in getattr(base_ref, "interventions", {}).items():
                if hasattr(inter, "tasks"):
                    print_rank_0(f"[REFT-CL] Round {round_idx}: Unfreezing task {i_task} in {layer_key}", self.args.global_rank)
                    block = inter.tasks[i_task]
                    param_count = sum(1 for p in block.parameters())
                    trainable_before = sum(1 for p in block.parameters() if p.requires_grad)
                    for p in block.parameters():
                        p.requires_grad = True
                    trainable_after = sum(1 for p in block.parameters() if p.requires_grad)
                    print_rank_0(f"  Task {i_task}: {param_count} params, {trainable_before}→{trainable_after} trainable", self.args.global_rank)

            # Alphas 1..t are trainable, t+1..T are frozen
            for j, a in enumerate(self.alpha_bank.alphas):
                a.requires_grad = (j <= i_task)
            try:
                flags = [p.requires_grad for p in self.alpha_bank.alphas]
                print(f"[DEBUG][ReFTCL] round {round_idx} alpha requires_grad flags: {flags}")
                # Also preview trainable params that include alpha/reft
                trainable_alpha = [n for n, p in (self.model.named_parameters()) if p.requires_grad and ("alpha" in n.lower() or "reft" in n.lower())]
                print(f"[DEBUG][ReFTCL] trainable alpha-like params this round: {trainable_alpha}")
            except Exception:
                pass

            # Activate tasks up to current round
            self._set_active_round(round_idx)

            # Delegate to base training per task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))

            # Save after each round
            self.save_model(i_task)



