import torch
from torch import nn

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
    """

    def __init__(self, **kwargs):
        # pyreft will populate standard fields like embed_dim via super().__init__
        num_tasks: int = kwargs["num_tasks"]
        self.eps: float = float(kwargs.get("eps", 1e-6))
        self.low_rank_dimension: int = int(kwargs["low_rank_dimension"])  # required by pyreft
        self.dropout_p: float = float(kwargs.get("dropout", 0.0))
        self.act_fn_name = kwargs.get("act_fn", None)
        self.dtype = kwargs.get("dtype", torch.bfloat16)

        # Initialize bases before assigning modules
        super().__init__(**kwargs, keep_last_dim=True)

        # Shared alpha accessor (callable) to avoid re-registering parameters in each intervention
        # Usage: self._get_alpha(i) -> nn.Parameter scalar for task i
        self._get_alpha = kwargs["get_alpha"]

        # Build one LoReFT block per task for (R, W, b) with pyreft defaults
        # We keep dropout=0 inside each block; final dropout controlled at this level if needed
        loreft_kwargs = {
            "low_rank_dimension": self.low_rank_dimension,
            "dropout": 0.0,
            "act_fn": self.act_fn_name,
            "dtype": self.dtype,
        }
        # Ensure pyreft receives embed_dim
        loreft_kwargs["embed_dim"] = self.embed_dim
        self.tasks = nn.ModuleList([LoreftIntervention(**loreft_kwargs) for _ in range(num_tasks)])

        # Optional top-level dropout after accumulation
        self.output_dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # Track how many tasks are active (<= num_tasks)
        self.active_tasks: int = num_tasks

    def set_active_tasks(self, n: int):
        self.active_tasks = max(0, min(n, len(self.tasks)))

    def forward(self, base, source=None, subspaces=None):
        # base: [..., embed_dim] (pyreft passes the right shape; keep_last_dim=True)
        h = base
        # Accumulate normalized edits per active task
        total_delta = torch.zeros_like(h)

        # Compute using float32 for numeric stability of norms
        for i in range(self.active_tasks):
            block = self.tasks[i]
            # R h
            rotated_base = block.rotate_layer(h)
            # W h + b
            learned = block.act_fn(block.learned_source(h))
            # (W h + b) - (R h)
            delta_low = learned - rotated_base
            # R^T * ...
            d = torch.matmul(delta_low, block.rotate_layer.weight.T)
            # normalize per token vector
            d32 = d.to(torch.float32)
            denom = torch.norm(d32, dim=-1, keepdim=True).clamp_min(self.eps)
            dir_i = (d32 / denom).to(h.dtype)

            alpha_i = self._get_alpha(i)
            total_delta = total_delta + alpha_i * dir_i

        out = h + total_delta
        return self.output_dropout(out)



