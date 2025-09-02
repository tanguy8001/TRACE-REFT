import torch
from torch import nn
from typing import List, Union, Dict

# pyreft base classes and LoReFT blocks
try:
    from pyreft.interventions import (
        SourcelessIntervention,
        TrainableIntervention,
        DistributedRepresentationIntervention,
        LoreftIntervention,
    )
except Exception as _e:
    # Fallback import path if API surface changes
    from pyreft import (
        SourcelessIntervention,
        TrainableIntervention,
        DistributedRepresentationIntervention,
    )
    from pyreft.interventions import LoreftIntervention  # type: ignore


class ReftCLIntervention(
    SourcelessIntervention, TrainableIntervention, DistributedRepresentationIntervention
):
    """
    Continual REFT intervention that accumulates per-task normalized directions
    and scales them by shared task-wise coefficients alpha_i.

    For task i with (R_i, W_i, b_i) coming from a LoReFT-style block:
        dir_i(h) = R_i^T ( W_i h + b_i - R_i h ) / ( || R_i^T ( W_i h + b_i - R_i h ) ||_2 + eps )
    Total edit: h' = h + sum_{i=1..active_tasks} alpha_i * dir_i(h).

    Notes
    - We reuse pyreft's LoreftIntervention modules for (R, W, b) initialization and activation.
    - Alphas are provided via a shared alpha_bank (nn.Module with .alphas ParameterList) so
      they are registered only once at the top-level model and shared across layers.
    - active_tasks can be updated across rounds to include tasks seen so far.
    - Optionally restrict to a subset of tasks using task_ids (list of global task indices).
    """

    def __init__(self, **kwargs):
        # pyreft will populate standard fields like embed_dim via super().__init__
        # Support either explicit subset selection via task_ids, or fallback to num_tasks
        task_ids = kwargs.get("task_ids", None)
        if task_ids is not None:
            self.task_ids = [int(t) for t in task_ids]
        else:
            num_tasks: int = int(kwargs["num_tasks"])  # type: ignore
            self.task_ids = list(range(num_tasks))
        self.eps: float = float(kwargs.get("eps", 1e-6))
        self.low_rank_dimension: int = int(kwargs["low_rank_dimension"])  # required by pyreft
        self.dropout_p: float = float(kwargs.get("dropout", 0.0))
        self.act_fn_name = kwargs.get("act_fn", None)
        self.dtype = kwargs.get("dtype", torch.bfloat16)

        # Initialize bases before assigning modules
        super().__init__(**kwargs, keep_last_dim=True)

        # Shared alpha accessor (callable) to avoid re-registering parameters in each intervention
        # Usage: self._get_alpha(global_task_id) -> nn.Parameter scalar for that task
        self._get_alpha = kwargs["get_alpha"]

        # Build one LoReFT block per task for (R, W, b) with pyreft defaults
        # We keep dropout=0 inside each block;  nal dropout controlled at this level if needed
        loreft_kwargs = {
            "low_rank_dimension": self.low_rank_dimension,
            "dropout": 0.05,
            "act_fn": self.act_fn_name,
            "dtype": self.dtype,
        }
        # Ensure pyreft receives embed_dim
        loreft_kwargs["embed_dim"] = self.embed_dim
        self.tasks = nn.ModuleList([LoreftIntervention(**loreft_kwargs) for _ in range(len(self.task_ids))])

        # Optional top-level dropout after accumulation
        self.output_dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # Track which local blocks are active (indices into self.tasks)
        self.active_indices: List[int] = []

    def set_active_tasks(self, n: int):
        """Activate the first n blocks in the subset order (backward-compatible behavior)."""
        n = max(0, min(n, len(self.tasks)))
        self.active_indices = list(range(n))

    def set_active_by_round(self, t: int):
        """Activate blocks whose global task id is < t (rounds are 1-indexed upstream)."""
        threshold = int(t)
        self.active_indices = [j for j, gid in enumerate(self.task_ids) if gid < threshold]

    def forward(self, base, source=None, subspaces=None):
        # base: [..., embed_dim] (pyreft passes the right shape; keep_last_dim=True)
        h = base
        # Accumulate normalized edits per active task
        total_delta = torch.zeros_like(h)

        # Compute using float32 for numeric stability of norms
        for j in self.active_indices:
            block = self.tasks[j]
            # Ensure input matches block params dtype to avoid dtype mismatch
            target_dtype = block.rotate_layer.weight.dtype
            h_cast = h.to(target_dtype)
            # R h
            rotated_base = block.rotate_layer(h_cast)
            # W h + b
            learned = block.act_fn(block.learned_source(h_cast))
            # (W h + b) - (R h)
            delta_low = learned - rotated_base
            # R^T * ...
            d = torch.matmul(delta_low, block.rotate_layer.weight.T)
            # normalize per token vector
            d32 = d.to(torch.float32)
            denom = torch.norm(d32, dim=-1, keepdim=True).clamp_min(self.eps)
            dir_i = (d32 / denom).to(h.dtype)

            # Map local block index to global task id for shared alpha
            global_task_id = self.task_ids[j]
            alpha_i = self._get_alpha(global_task_id)
            total_delta = total_delta + alpha_i * dir_i

        out = h + total_delta
        return self.output_dropout(out)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to properly save nested LoreftIntervention parameters."""
        if destination is None:
            destination = {}
        
        # Save each task's LoreftIntervention state using pyreft's native state_dict
        for i, task_module in enumerate(self.tasks):
            task_prefix = f"{prefix}tasks.{i}."
            # Use pyreft's native state_dict and manually add the task prefix
            task_state = task_module.state_dict(prefix="", keep_vars=keep_vars)
            for key, value in task_state.items():
                prefixed_key = f"{task_prefix}{key}"
                destination[prefixed_key] = value
        
        # Save other parameters (like dropout)
        if hasattr(self.output_dropout, 'state_dict'):
            dropout_state = self.output_dropout.state_dict(prefix=f"{prefix}output_dropout.", keep_vars=keep_vars)
            destination.update(dropout_state)
        
        # Save module metadata
        destination[f"{prefix}active_indices"] = torch.tensor(self.active_indices, dtype=torch.long)
        destination[f"{prefix}task_ids"] = torch.tensor(self.task_ids, dtype=torch.long)
        destination[f"{prefix}eps"] = torch.tensor(self.eps)
        destination[f"{prefix}low_rank_dimension"] = torch.tensor(self.low_rank_dimension)
        
        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to properly load nested LoreftIntervention parameters."""
        missing_keys = []
        unexpected_keys = []
        
        # Load each task's LoreftIntervention state by directly setting parameters
        for i, task_module in enumerate(self.tasks):
            task_prefix = f"tasks.{i}."
            
            # Extract keys for this task
            task_state = {}
            for key in list(state_dict.keys()):
                if key.startswith(task_prefix):
                    # Remove the task prefix for the individual module
                    module_key = key[len(task_prefix):]
                    task_state[module_key] = state_dict[key]
            
            if task_state:
                try:
                    # Directly set parameters to avoid pyreft's load_state_dict issues
                    for param_name, param_value in task_state.items():
                        if param_name == "learned_source.weight":
                            if hasattr(task_module, 'learned_source') and hasattr(task_module.learned_source, 'weight'):
                                task_module.learned_source.weight.data.copy_(param_value)
                        elif param_name == "learned_source.bias":
                            if hasattr(task_module, 'learned_source') and hasattr(task_module.learned_source, 'bias'):
                                task_module.learned_source.bias.data.copy_(param_value)
                        elif param_name == "rotate_layer.parametrizations.weight.original":
                            if hasattr(task_module, 'rotate_layer') and hasattr(task_module.rotate_layer, 'parametrizations'):
                                task_module.rotate_layer.parametrizations.weight.original.data.copy_(param_value)
                    
                    print(f"[REFT-CL] Successfully loaded task {i} parameters")
                except Exception as e:
                    print(f"[DEBUG] Error loading task {i}: {e}")
                    print(f"[DEBUG] Task {i} state keys: {list(task_state.keys())}")
                    print(f"[DEBUG] Task {i} state dtypes: {[(k, v.dtype if torch.is_tensor(v) else type(v)) for k, v in task_state.items()]}")
                    raise
        
        # Load other parameters
        if hasattr(self.output_dropout, 'load_state_dict'):
            dropout_prefix = "output_dropout."
            dropout_state = {}
            for key in list(state_dict.keys()):
                if key.startswith(dropout_prefix):
                    module_key = key[len(dropout_prefix):]
                    dropout_state[module_key] = state_dict[key]
            if dropout_state:
                missing, unexpected = self.output_dropout.load_state_dict(dropout_state, strict=strict)
                missing_keys.extend([f"{dropout_prefix}{k}" for k in missing])
                unexpected_keys.extend([f"{dropout_prefix}{k}" for k in unexpected])
        
        # Load metadata
        if "active_indices" in state_dict:
            ai = state_dict["active_indices"]
            try:
                self.active_indices = ai.tolist()  # tensor -> list
            except Exception:
                # Be tolerant to older saves where it may be a python list already
                self.active_indices = list(ai)
        if "task_ids" in state_dict:
            tids = state_dict["task_ids"]
            try:
                self.task_ids = [int(x) for x in tids.tolist()]
            except Exception:
                self.task_ids = [int(x) for x in tids]
        if "eps" in state_dict:
            self.eps = float(state_dict["eps"].item())
        if "low_rank_dimension" in state_dict:
            self.low_rank_dimension = int(state_dict["low_rank_dimension"].item())
        
        return missing_keys, unexpected_keys



