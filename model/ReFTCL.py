import os
import torch
from typing import Dict, List, Union

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
    """Holds shared alphas (one per global task index) so all layers can reference them."""

    def __init__(self, num_tasks: int, alpha_init: float = 0.1):
        super().__init__()
        self.alphas = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32)) for _ in range(num_tasks)]
        )

    def get_alpha(self, global_task_id: int) -> torch.nn.Parameter:
        return self.alphas[int(global_task_id)]


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
        # shared across layers (alphas for all tasks; interventions can pick a subset)
        self.alpha_bank = AlphaBank(self.num_tasks, alpha_init=0.1)

        # Inject pyreft model wrapper with our intervention across selected layers
        # We do this once here so parameters are registered before deepspeed init in main.py
        self._attach_reft_interventions()

    def _attach_reft_interventions(self):
        """Wrap self.model with pyreft get_reft_model using our intervention."""

        # Detect task-specific layer scheme from args (reft_layer_task_1..8)
        task_specific_layers = {}
        using_task_specific = False
        for _i in range(1, 9):
            arg_name = f"reft_layer_task_{_i}"
            task_layer_arg = getattr(self.args, arg_name, None)
            if isinstance(task_layer_arg, str) and len(task_layer_arg.strip()) > 0:
                    task_specific_layers[_i - 1] = [int(x) for x in task_layer_arg.split(";") if len(x) > 0]
                    using_task_specific = True

        if using_task_specific:
            print_rank_0(f"[ReFT-CL] Using task-specific layer scheme (scheme 2)", self.args.global_rank)
            self._attach_task_specific_interventions(task_specific_layers)
        else:
            print_rank_0(f"[ReFT-CL] Using unified layer scheme (scheme 1)", self.args.global_rank)
            self._attach_unified_interventions()

    def _attach_unified_interventions(self):
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
        # Parse optional subset argument: semicolon-separated global indices, e.g., "0;2;5"
        subset_arg = getattr(self.args, "reft_task_subset", None)
        if isinstance(subset_arg, str) and len(subset_arg.strip()) > 0:
            subset_task_ids = [int(x) for x in subset_arg.split(";") if len(x) > 0]
        elif isinstance(subset_arg, (list, tuple)):
            subset_task_ids = [int(x) for x in subset_arg]
        else:
            subset_task_ids = list(range(self.num_tasks))

        # Provide a getter to share one alpha per task across all layers without duplicating params
        def _get_alpha(global_task_id: int):
            return self.alpha_bank.get_alpha(global_task_id)

        for l in target_layers:
            reps.append({
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": low_rank,
                "intervention": ReftCLIntervention(
                    embed_dim=embed_dim,
                    low_rank_dimension=low_rank,
                    task_ids=subset_task_ids,
                    get_alpha=_get_alpha,
                    eps=eps,
                    dtype=torch.float32,
                )
            })

        cfg = ReftConfig(representations=reps)
        self.model = get_reft_model(self.model, cfg, set_device=False)
        self._finalize_model_setup()

    def _attach_task_specific_interventions(self, task_specific_layers: dict):
        low_rank = int(getattr(self.args, "reft_rank", 4))
        eps = float(getattr(self.args, "reft_eps", 1e-6))
        embed_dim = int(getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None))

        # Combine all unique layers
        all_layers = set()
        for _task_id, layers in task_specific_layers.items():
            all_layers.update(layers)

        def _get_alpha(global_task_id: int):
            return self.alpha_bank.get_alpha(global_task_id)

        reps = []
        for l in sorted(all_layers):
            # Determine which tasks map to this layer
            tasks_for_layer = [tid for tid, ls in task_specific_layers.items() if l in ls]
            if not tasks_for_layer:
                continue
            reps.append({
                "layer": l,
                "component": "block_output",
                "low_rank_dimension": low_rank,
                "intervention": ReftCLIntervention(
                    embed_dim=embed_dim,
                    low_rank_dimension=low_rank,
                    task_ids=tasks_for_layer,
                    get_alpha=_get_alpha,
                    eps=eps,
                    dtype=torch.float32,
                )
            })

        cfg = ReftConfig(representations=reps)
        self.model = get_reft_model(self.model, cfg, set_device=False)
        self._finalize_model_setup()

    def _finalize_model_setup(self):
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
            if hasattr(inter, "set_active_by_round"):
                inter.set_active_by_round(t)
            elif hasattr(inter, "set_active_tasks"):
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
        # Determine target layers under both schemes
        task_specific_layers = {}
        using_task_specific = False
        for _i in range(1, 9):
            arg_name = f"reft_layer_task_{_i}"
            task_layer_arg = getattr(self.args, arg_name, None)
            if isinstance(task_layer_arg, str) and len(task_layer_arg.strip()) > 0:
                try:
                    task_specific_layers[_i - 1] = [int(x) for x in task_layer_arg.split(";") if len(x) > 0]
                    using_task_specific = True
                except Exception:
                    pass
        if using_task_specific:
            # Count unique layer-task pairs
            layer_to_tasks = {}
            for tid, layers in task_specific_layers.items():
                for l in layers:
                    layer_to_tasks.setdefault(l, set()).add(tid)
            target_layers = sorted(layer_to_tasks.keys())
            tasks_per_layer = {l: len(layer_to_tasks[l]) for l in target_layers}
        else:
            layer_str = getattr(self.args, "reft_layers", "3;9;18;24")
            if layer_str.strip() == "all":
                num_layers = self.model.config.num_hidden_layers
                target_layers = list(range(num_layers))
            else:
                target_layers = [int(x) for x in layer_str.split(";") if len(x) > 0]
            # Under unified scheme, tasks_per_layer is uniform and equals size of subset
            # Parse optional subset argument
            subset_arg = getattr(self.args, "reft_task_subset", None)
            if isinstance(subset_arg, str) and len(subset_arg.strip()) > 0:
                subset_task_ids = [int(x) for x in subset_arg.split(";") if len(x) > 0]
            elif isinstance(subset_arg, (list, tuple)):
                subset_task_ids = [int(x) for x in subset_arg]
            else:
                subset_task_ids = list(range(self.num_tasks))
            tasks_per_layer = {l: len(subset_task_ids) for l in target_layers}
        
        low_rank = int(getattr(self.args, "reft_rank", 4))
        embed_dim = int(getattr(self.model.config, "hidden_size", None) or getattr(self.model.config, "d_model", None))
        
        # Per task per layer: R (embed_dim x low_rank) + W (low_rank x embed_dim) + b (low_rank)
        params_per_task_per_layer = (embed_dim * low_rank) + (low_rank * embed_dim) + low_rank
        # Sum over layers the number of tasks assigned to that layer
        total_task_layer_pairs = sum(tasks_per_layer[l] for l in tasks_per_layer.keys())
        total_expected = total_task_layer_pairs * params_per_task_per_layer
        
        print_rank_0(f"[DEBUG][ReFTCL] Expected intervention params calculation:", self.args.global_rank)
        print_rank_0(f"  - Layers: {len(tasks_per_layer)} (indices: {list(tasks_per_layer.keys())})", self.args.global_rank)
        print_rank_0(f"  - Task-layer pairs: {total_task_layer_pairs}", self.args.global_rank)
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

            # Use pyreft's save method which properly handles interventions
            model_to_save.save(save_dir)
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
            for inter in getattr(base_ref, "interventions", {}).values():
                if hasattr(inter, "tasks"):
                    print(f"Freezing {len(inter.tasks)} blocks")
                    for block in inter.tasks:
                        for p in block.parameters():
                            p.requires_grad = False

            # Unfreeze current round directions (R_t, W_t, b_t)
            for inter in getattr(base_ref, "interventions", {}).values():
                if hasattr(inter, "tasks") and hasattr(inter, "task_ids"):
                    # If current global task id exists in this subset, unfreeze its local block
                    if i_task in inter.task_ids:
                        local_idx = inter.task_ids.index(i_task)
                        print(f"Unfreezing block {local_idx} for global task {i_task}")
                        block = inter.tasks[local_idx]
                        for p in block.parameters():
                            p.requires_grad = True

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



