import torch, os
ckpt = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL/7/pytorch_model.bin"
sd = torch.load(ckpt, map_location="cpu")
alphas = [sd[k].item() for k in sorted(sd) if k.startswith("reftcl_alpha_bank.alphas.")]
print("num_alphas:", len(alphas), "values:", alphas)