import os
import torch
from typing import Dict

from model.base_model import CL_Base_Model
from utils.utils import print_rank_0


class AlphaBank(torch.nn.Module):
    """Holds shared alphas (one per task) so all layers can reference them."""

    def __init__(self, num_tasks: int, alpha_init: float = 0.1):
        super().__init__()
        self.alphas = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.bfloat16)) for _ in range(num_tasks)]
        )


class ReFTCL(CL_Base_Model):
    """
    Continual Learning trainer that composes ReFT interventions per task and shares
    a single alpha per task across all layers. It relies on a custom pyreft intervention
    that accumulates normalized edits online.
    """

    def __init__(self, model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args):
        super().__init__(model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        # Infer num tasks from provided train_task_list
        self.tasks = list(self.train_task_list.keys())
        self.num_tasks = len(self.tasks)

        # Build alpha bank (shared across layers)
        self.alpha_bank = AlphaBank(self.num_tasks, alpha_init=0.1)

        # Inject pyreft model wrapper with our intervention across selected layers
        # We do this once here so parameters are registered before deepspeed init in main.py
        self._attach_reft_interventions()

    def _attach_reft_interventions(self):
        """Wrap self.model with pyreft get_reft_model using our intervention."""
        # Import locations differ across pyreft versions
        try:
            from pyreft import get_reft_model, ReftConfig  # modern API
        except Exception:
            try:
                from pyreft.utils import get_reft_model  # older API
                from pyreft import ReftConfig
            except Exception as e:
                raise ImportError("pyreft get_reft_model/ReftConfig not found; please update pyreft") from e
        from loreft.reft_cl_intervention import ReftCLIntervention

        # Determine target layers from args or defaults
        layer_str = getattr(self.args, "reft_layers", "3;9;18;24")
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
                    dtype=torch.bfloat16,
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

    def _set_active_round(self, t: int):
        # Enable tasks 1..t (1-indexed externally) => internally 0..t-1
        base_ref = self.model.module if hasattr(self.model, "module") else self.model
        for inter in getattr(base_ref, "interventions", {}).values():
            if hasattr(inter, "set_active_tasks"):
                inter.set_active_tasks(t)

    def ensure_orthogonal_params_fp32(self):
        """Cast orthogonal rotation parameters to float32 to keep Householder orgqr in supported dtype.
        Must be called AFTER DeepSpeed initialize, which may cast modules to bfloat16.
        """
        base_ref = self.model.module if hasattr(self.model, "module") else self.model
        for inter in getattr(base_ref, "interventions", {}).values():
            if hasattr(inter, "tasks"):
                for block in inter.tasks:
                    if hasattr(block, "rotate_layer"):
                        rot = block.rotate_layer
                        # Best-effort: cast parametrization's stored base/original and module to float32
                        try:
                            # Parametrizations are stored under .parametrizations.weight[0]
                            if hasattr(rot, "parametrizations") and hasattr(rot.parametrizations, "weight"):
                                ortho_obj = rot.parametrizations.weight[0]
                                if hasattr(ortho_obj, "original") and ortho_obj.original.dtype != torch.float32:
                                    ortho_obj.original = torch.nn.Parameter(ortho_obj.original.detach().to(torch.float32))
                                if hasattr(ortho_obj, "base") and ortho_obj.base.dtype != torch.float32:
                                    ortho_obj.base = torch.nn.Parameter(ortho_obj.base.detach().to(torch.float32))
                        except Exception:
                            pass
                        # Ensure the exposed weight and module params are float32
                        try:
                            rot.to(torch.float32)
                        except Exception:
                            pass

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
                    for j in range(self.num_tasks):
                        block = inter.tasks[j]
                        for p in block.parameters():
                            p.requires_grad = False

            # Unfreeze current round directions (R_t, W_t, b_t)
            for inter in getattr(base_ref, "interventions", {}).values():
                if hasattr(inter, "tasks"):
                    print(f"Unfreezing {len(inter.tasks)} blocks")
                    block = inter.tasks[i_task]
                    for p in block.parameters():
                        p.requires_grad = True

            # Alphas 1..t are trainable, t+1..T are frozen
            for j, a in enumerate(self.alpha_bank.alphas):
                a.requires_grad = (j <= i_task)

            # Activate tasks up to current round
            self._set_active_round(round_idx)

            # Delegate to base training per task
            self.train_one_task(task, i_task, int(self.args.num_train_epochs[i_task]))

            # Save after each round
            self.save_model(i_task)



