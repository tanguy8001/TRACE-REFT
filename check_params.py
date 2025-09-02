import torch
import os

base_dir = "/cluster/scratch/tdieudonne/outputs_LLM-CL/cl/REFT-CL"
layer_file = "intkey_layer_18_comp_block_output_unit_pos_nunit_1#0.bin"
keys_to_check = ["weight", "bias", "rotate_layer"]

num_rounds = 8

for rnd in range(num_rounds):
    file_path = os.path.join(base_dir, str(rnd), layer_file)
    if not os.path.exists(file_path):
        print(f"❌ File not found for round {rnd}: {file_path}")
        continue

    state = torch.load(file_path, map_location="cpu")

    print(f"\n=== Round {rnd} ===")
    #print("Keys available:")
    #for k, v in state.items():
    #    if torch.is_tensor(v):
    #        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
    #    else:
    #        print(f"  {k}: {v} ({type(v)})")

    print("\nNorms of main matrices:")
    for key in state.keys():
        for mat in keys_to_check:
            if key.endswith(mat):
                tensor = state[key].float()
                norm_val = tensor.norm().item()
                print(f"  {key}: ||·|| = {norm_val:.6e}")
